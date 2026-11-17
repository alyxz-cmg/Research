#!/usr/bin/env python3
"""
cnn_cifar10_experiment.py (v1.2.3)
==================================
Staged implementation of Synaptic Neural Balance in Convolutional Neural Networks
following HandOff1.md and Baldi et al. (2025), Sections 8.2, 8.3, 12.2, and Appendix B.

Theoretical & Methodological Architecture (v1.2.3):
  1. Two Balancing Granularities (Baldi et al. Sections 8.2 & 8.3):
     - `balance_mode="layer"` (Coordinated Layer Balancing, Section 8.3 literal formulation):
       Uses a single shared scalar multiplier lambda_l > 0 across all neurons/filters in
       convolutional layer l (5 layer-transition balance constraints in our 6-layer CNN).
     - `balance_mode="channel"` (Coordinated Filter/Channel Subset Balancing, Section 8.2
       tied-subset formulation applied to CNN kernels):
       Treats each output feature map c in layer l as a disjoint tied subset sharing
       multiplier lambda_{l,c} > 0 across its H_l x W_l spatial neurons (paired with
       `fc_balance_mode="neuron"`, yielding 32 + 64 + 128 + 128 + 256 = 608 constraints).
     Both modes preserve exact positive homogeneity through ReLU and channel-wise MaxPool2d
     because every spatial position within a pooling window receives the same multiplier.
  2. Two Cost Domains (`--cost-domain {parameter,connection_graph}`):
     - `parameter`: Minimizes unique trainable parameter squared cost
       R_param(W) = sum_l (||W^(l)||_F^2 + ||b^(l)||_2^2).
     - `connection_graph`: Minimizes multiplicity-weighted synaptic graph squared cost
       R_graph(W) (Section 8.2 Eq. 8.1), weighting each kernel tap (u,v) by its exact
       valid non-padded spatial multiplicity M_l^in(u,v) and bias by H_l * W_l across
       coordinated Conv-plus-Pool blocks.
  3. Persisted Stage 1 Verification Report & Hardened Git/Source Provenance:
     - Writes `stage1_verification_report.json` recording all pass/fail results, float64
       and float32 logit invariance errors, convergence iterations, and per-transition
       |log(lam_param) - log(lam_graph)| shifts.
     - Records `source_script_sha256`, `git_commit_full`, `git_working_tree_dirty`, and
       supports `--require-clean-git` to guarantee exact commit-to-artifact provenance.
  4. Strict Stage-Specific Test-Set Discipline:
     - Stage 2 evaluates the official CIFAR-10 test set only in `full` mode (or with
       `--allow-nonconfirmatory-test-eval`).
     - Stage 3 withholds the test set across ALL modes unless `--stage3-confirmatory-test-eval`
       is explicitly passed (generic override cannot unlock Stage 3 test evaluation).
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


# =============================================================================
# 1. Configuration, Strict Determinism & Source/Git Provenance Utilities
# =============================================================================

def get_git_provenance() -> Dict[str, Any]:
    """
    Returns full and short git commit hashes plus an explicit boolean indicating
    whether the working tree has uncommitted modifications.
    """
    info: Dict[str, Any] = {
        "git_commit_short": "unknown",
        "git_commit_full": "unknown",
        "git_working_tree_dirty": True,
        "git_dirty_files": [],
    }
    try:
        short_out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        full_out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").splitlines()
        dirty_lines = [line.strip() for line in status_out if line.strip()]
        info["git_commit_short"] = short_out
        info["git_commit_full"] = full_out
        info["git_working_tree_dirty"] = bool(len(dirty_lines) > 0)
        info["git_dirty_files"] = dirty_lines[:25]
    except Exception:
        pass
    return info


def get_source_script_sha256() -> str:
    """Computes the exact SHA-256 hex digest of this Python script file on disk."""
    try:
        script_path = os.path.abspath(__file__)
        with open(script_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "unknown"


def configure_reproducibility(deterministic: bool = True, warn_only: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


@dataclass
class CNNExperimentConfig:
    version: str = "1.2.3"
    stage: int = 1
    run_mode: str = "smoke"  # "smoke" (seed 42, 5 ep), "pilot" (seeds 43-47, 60 ep), "full" (seeds 48-55, 60 ep)

    # Balancing definitions (Section 8.2 vs. Section 8.3)
    cost_domain: str = "parameter"        # "parameter" (unique parameter L2) or "connection_graph" (Sec 8.2 Eq. 8.1)
    balance_mode: str = "layer"           # "layer" (Sec 8.3 whole-layer) or "channel" (Sec 8.2 per-filter tied subset)
    fc_balance_mode: str = "layer"        # "layer" (whole-layer) or "neuron" (independent FC neuron balance)

    # Disjoint seed tiers (strictly non-overlapping across smoke, pilot, and confirmatory full runs)
    smoke_seeds: List[int] = field(default_factory=lambda: [42])
    pilot_seeds: List[int] = field(default_factory=lambda: [43, 44, 45, 46, 47])
    full_seeds: List[int] = field(default_factory=lambda: [48, 49, 50, 51, 52, 53, 54, 55])

    # Dataset & Split (CIFAR-10: 45,000 train / 5,000 val / 10,000 official test + 5,000 unaugmented train_subset_eval)
    data_dir: str = "./data"
    split_seed: int = 20261117
    train_size: int = 45000
    val_size: int = 5000
    train_subset_eval_size: int = 5000
    batch_size: int = 128
    use_augmentation: bool = True
    aug_padding: int = 4
    aug_pad_mode: str = "reflect"

    # Custom VGG-style Sequential CNN Architecture & Dropout Policy
    conv_channels: Tuple[int, int, int, int] = (32, 64, 128, 128)
    fc_hidden: int = 256
    num_classes: int = 10
    conv_dropout_mode: str = "elementwise"  # "elementwise" (B,C,H,W) or "channel" (Dropout2d B,C,1,1)
    conv_dropout_rate: float = 0.20
    fc_dropout_rate: float = 0.50

    # Optimization (Stateless SGD to avoid momentum-rescaling confounds)
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    epochs_smoke: int = 5
    epochs_main: int = 60
    deterministic: bool = True
    deterministic_warn_only: bool = False
    allow_nonconfirmatory_test_eval: bool = False
    stage3_confirmatory_test_eval: bool = False
    require_clean_git: bool = False
    save_checkpoints: bool = False

    # Numerical & Convergence parameters
    balance_eps: float = 1e-12
    full_balance_max_iters: int = 300
    full_balance_tol_f64: float = 1e-10
    full_balance_tol_f32: float = 1e-5
    logit_tol_f64: float = 1e-12
    logit_tol_f32: float = 1e-5

    # Stage 3 Orthogonal 1D Validation Sweeps (conditional 1D slices on validation data)
    conv_dropout_sweep_1d: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40)
    fc_dropout_sweep_1d: Tuple[float, ...] = (0.00, 0.25, 0.50, 0.65, 0.80)

    # Root directory for run-specific output folders
    output_root: str = "./results_cnn_cifar10"

    def active_seeds(self) -> List[int]:
        if self.run_mode == "smoke":
            return list(self.smoke_seeds)
        elif self.run_mode == "pilot":
            return list(self.pilot_seeds)
        elif self.run_mode == "full":
            return list(self.full_seeds)
        raise ValueError(f"Unknown run_mode: {self.run_mode}")

    def active_epochs(self) -> int:
        return self.epochs_smoke if self.run_mode == "smoke" else self.epochs_main

    def should_evaluate_test_set(self, for_stage: Optional[int] = None) -> bool:
        """
        Enforces strict stage-specific test-set discipline:
          - Stage 2: official test set is withheld in `smoke` and `pilot` modes and
            unlocked in confirmatory `full` mode (or if `allow_nonconfirmatory_test_eval`
            is explicitly enabled for Stage 2).
          - Stage 3: official test set is withheld across ALL modes (`smoke`, `pilot`,
            and `full`) unless `stage3_confirmatory_test_eval` is explicitly True.
            The generic `allow_nonconfirmatory_test_eval` flag cannot unlock Stage 3.
        """
        st = self.stage if for_stage is None else int(for_stage)
        if st == 2:
            return (self.run_mode == "full") or bool(self.allow_nonconfirmatory_test_eval)
        if st == 3:
            return bool(self.stage3_confirmatory_test_eval)
        return False

    def constraint_unit_type_label(self) -> str:
        if self.balance_mode == "layer" and self.fc_balance_mode == "layer":
            return "layer_transition"
        if self.balance_mode == "channel" and self.fc_balance_mode == "neuron":
            return "channel_and_fc_neuron"
        return f"conv_{self.balance_mode}_fc_{self.fc_balance_mode}"

    def config_fingerprint(self) -> str:
        """
        Computes a deterministic 10-character SHA-256 hex fingerprint over both the
        complete serialized configuration AND the on-disk script SHA-256, ensuring
        any code edit or hyperparameter change produces a distinct fingerprint.
        """
        payload = asdict(self)
        payload["active_seeds"] = self.active_seeds()
        payload["active_epochs"] = self.active_epochs()
        payload["effective_evaluate_test_set"] = self.should_evaluate_test_set(self.stage)
        payload["source_script_sha256"] = get_source_script_sha256()
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:10]

    def run_tag(self) -> str:
        return (
            f"stage{self.stage}_{self.run_mode}_{self.cost_domain}"
            f"_conv-{self.balance_mode}_fc-{self.fc_balance_mode}"
            f"_drop-{self.conv_dropout_mode}"
            f"_pc{self.conv_dropout_rate:g}_pf{self.fc_dropout_rate:g}"
            f"_lr{self.learning_rate:g}_ep{self.active_epochs()}"
            f"_{self.config_fingerprint()}"
        )

    def run_output_dir(self) -> str:
        out_dir = os.path.join(self.output_root, self.run_tag())
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def output_path(self, filename: str) -> str:
        return os.path.join(self.run_output_dir(), filename)


@dataclass
class IsolatedRNGStreams:
    """
    Dedicated random number generators for paired-seed experiments.
    - `gen_init`, `gen_loader`, `gen_aug`, and `gen_rescale` live on CPU.
    - `gen_dropout` lives directly on the execution `device` (`cuda` or `cpu`),
      avoiding CPU-to-GPU mask transfers while guaranteeing bitwise-identical
      dropout masks between paired Dropout-On conditions (Groups B and D).
    """
    gen_init: torch.Generator
    gen_loader: torch.Generator
    gen_aug: torch.Generator
    gen_dropout: torch.Generator
    gen_rescale: torch.Generator

    @classmethod
    def from_seed(cls, seed: int, device: torch.device = torch.device("cpu")) -> "IsolatedRNGStreams":
        def _make_cpu(offset: int) -> torch.Generator:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed) * 10007 + offset)
            return g

        dev_str = "cuda" if device.type == "cuda" else "cpu"
        gen_drop = torch.Generator(device=dev_str)
        gen_drop.manual_seed(int(seed) * 10007 + 4)

        return cls(
            gen_init=_make_cpu(1),
            gen_loader=_make_cpu(2),
            gen_aug=_make_cpu(3),
            gen_dropout=gen_drop,
            gen_rescale=_make_cpu(5),
        )


# =============================================================================
# 2. Generator-Controlled Dropout & Custom Sequential VGG-Style CNN
# =============================================================================

class GeneratorDropout(nn.Module):
    """
    Inverted dropout driven by an isolated torch.Generator (`gen_dropout`).
    Supports two explicit mask granularities for 4D convolutional feature maps (B, C, H, W):
      - mode="elementwise": independent mask per spatial activation (B, C, H, W).
      - mode="channel": Spatial/Dropout2d mask per feature map channel (B, C, 1, 1).
    For 2D linear activations (B, D), both modes apply standard elementwise masks (B, D).
    """
    def __init__(self, p: float = 0.5, mode: str = "elementwise"):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        if mode not in ("elementwise", "channel"):
            raise ValueError(f"mode must be 'elementwise' or 'channel', got {mode}")
        self.p = float(p)
        self.mode = mode

    def forward(self, x: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask_shape = (x.shape[0], x.shape[1], 1, 1) if (x.ndim == 4 and self.mode == "channel") else x.shape

        if rng is not None:
            if rng.device.type == x.device.type:
                probs = torch.rand(mask_shape, generator=rng, dtype=x.dtype, device=x.device)
            else:
                probs = torch.rand(mask_shape, generator=rng, dtype=x.dtype, device=rng.device).to(x.device)
        else:
            probs = torch.rand(mask_shape, dtype=x.dtype, device=x.device)

        mask = (probs < keep_prob).to(x.dtype) / keep_prob
        return x * mask


class SequentialCIFARCNN(nn.Module):
    """
    Custom VGG-style sequential ReLU CNN for CIFAR-10 (3x32x32 -> 10 classes).
    Excludes BatchNorm and skip connections so that all hidden layers are positively
    homogeneous and commute with coordinated layer/channel scaling through MaxPool2d.

    Pooling-Block Definition:
      For layers followed by MaxPool2d (`conv2` and `conv4`), the convolution and its
      immediately following spatial pooling operation form one coordinated function-preserving
      block: incoming cost is evaluated on the pre-pooling convolutional connections, and
      outgoing cost is evaluated on the post-pooling connections to the next learned layer.
    """
    def __init__(
        self,
        in_channels: int = 3,
        spatial_size: int = 32,
        conv_channels: Tuple[int, int, int, int] = (32, 64, 128, 128),
        fc_hidden: int = 256,
        num_classes: int = 10,
        use_dropout: bool = False,
        conv_dropout_mode: str = "elementwise",
        conv_dropout_rate: float = 0.20,
        fc_dropout_rate: float = 0.50,
    ):
        super().__init__()
        c1, c2, c3, c4 = conv_channels
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.conv_channels = conv_channels
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes
        self.use_dropout = bool(use_dropout)
        self.conv_dropout_mode = conv_dropout_mode

        self.final_spatial = spatial_size // 4
        if self.final_spatial < 1 or (spatial_size % 4 != 0):
            raise ValueError(f"spatial_size must be divisible by 4, got {spatial_size}")

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(c4 * self.final_spatial * self.final_spatial, fc_hidden, bias=True)
        self.fc2 = nn.Linear(fc_hidden, num_classes, bias=True)

        p_conv = conv_dropout_rate if self.use_dropout else 0.0
        p_fc = fc_dropout_rate if self.use_dropout else 0.0
        self.drop_conv1 = GeneratorDropout(p=p_conv, mode=conv_dropout_mode)
        self.drop_conv2 = GeneratorDropout(p=p_conv, mode=conv_dropout_mode)
        self.drop_fc1 = GeneratorDropout(p=p_fc, mode="elementwise")

        self.param_layers: List[nn.Module] = [
            self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2
        ]
        s0 = spatial_size
        s1 = spatial_size // 2
        self.conv_spatial_shapes: List[Tuple[int, int, int, int]] = [
            (s0, s0, s0, s0),  # conv1: 32x32 -> 32x32
            (s0, s0, s0, s0),  # conv2: 32x32 -> 32x32 (followed by pool -> 16x16)
            (s1, s1, s1, s1),  # conv3: 16x16 -> 16x16
            (s1, s1, s1, s1),  # conv4: 16x16 -> 16x16 (followed by pool -> 8x8)
        ]

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters_with_generator(self, gen: torch.Generator) -> None:
        """Deterministic Kaiming initialization using an isolated CPU generator."""
        with torch.no_grad():
            for m in self.param_layers:
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = math.sqrt(2.0 / max(1, fan_in))
                    w_cpu = torch.randn(m.weight.shape, generator=gen, dtype=m.weight.dtype) * std
                    m.weight.copy_(w_cpu.to(m.weight.device))
                    if m.bias is not None:
                        bound = 1.0 / math.sqrt(max(1, fan_in))
                        b_cpu = torch.empty(m.bias.shape, dtype=m.bias.dtype).uniform_(
                            -bound, bound, generator=gen
                        )
                        m.bias.copy_(b_cpu.to(m.bias.device))

    def forward(self, x: torch.Tensor, dropout_rng: Optional[torch.Generator] = None) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop_conv1(x, rng=dropout_rng)

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop_conv2(x, rng=dropout_rng)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x, rng=dropout_rng)

        logits = self.fc2(x)
        return logits


# =============================================================================
# 3. Stage 1 Core Mathematics: Parameter-Domain & Connection-Graph Balancing
# =============================================================================

def compute_conv2d_spatial_multiplicity(
    conv_layer: nn.Conv2d,
    h_in: int,
    w_in: int,
    h_out: int,
    w_out: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Computes the exact number of valid (non-zero-padded) spatial connections M_in(u, v)
    instantiated by each kernel offset (u, v) of `conv_layer` across an (h_out, w_out)
    output feature map, plus the bias multiplicity (h_out * w_out).
    """
    kh, kw = conv_layer.kernel_size
    sh, sw = conv_layer.stride
    ph, pw = conv_layer.padding
    dh, dw = conv_layer.dilation

    mult = torch.zeros((kh, kw), dtype=torch.float64, device=device)
    oh = torch.arange(h_out, device=device)
    ow = torch.arange(w_out, device=device)

    for u in range(kh):
        in_rows = oh * sh - ph + u * dh
        valid_h = int(((in_rows >= 0) & (in_rows < h_in)).sum().item())
        for v in range(kw):
            in_cols = ow * sw - pw + v * dw
            valid_w = int(((in_cols >= 0) & (in_cols < w_in)).sum().item())
            mult[u, v] = float(valid_h * valid_w)

    bias_mult = float(h_out * w_out)
    return mult, bias_mult


def compute_total_l2_regularizer(
    model: SequentialCIFARCNN, cost_domain: str = "parameter"
) -> float:
    """
    Computes the total squared L2 cost R(W) in float64 precision under either:
      - cost_domain == "parameter": sum of squared unique trainable parameters.
      - cost_domain == "connection_graph": sum of squared weights over all instantiated
        connections in the computational graph (weighted by exact spatial reuse multiplicity).
    """
    if cost_domain not in ("parameter", "connection_graph"):
        raise ValueError(f"Unknown cost_domain: {cost_domain}")

    total = 0.0
    conv_idx = 0
    with torch.no_grad():
        for layer in model.param_layers:
            w = layer.weight.detach().double()
            b = layer.bias.detach().double() if layer.bias is not None else None

            if cost_domain == "parameter" or isinstance(layer, nn.Linear):
                total += float(w.pow(2).sum().item())
                if b is not None:
                    total += float(b.pow(2).sum().item())
            elif isinstance(layer, nn.Conv2d):
                h_in, w_in, h_out, w_out = model.conv_spatial_shapes[conv_idx]
                mult_uv, bias_mult = compute_conv2d_spatial_multiplicity(
                    layer, h_in, w_in, h_out, w_out, device=w.device
                )
                total += float((w.pow(2) * mult_uv.view(1, 1, *mult_uv.shape)).sum().item())
                if b is not None:
                    total += float((b.pow(2) * bias_mult).sum().item())
                conv_idx += 1
    return total


def _get_unit_costs(
    model: SequentialCIFARCNN,
    layer_idx: int,
    mode: str,
    cost_domain: str = "parameter",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (A, B) 1D float64 tensors of incoming and outgoing squared L2 costs for
    hidden layer `layer_idx` (0 <= layer_idx < len(model.param_layers) - 1).
    """
    curr_layer = model.param_layers[layer_idx]
    next_layer = model.param_layers[layer_idx + 1]
    final_spatial = model.final_spatial

    w_in = curr_layer.weight.detach().double()
    b_in = curr_layer.bias.detach().double() if curr_layer.bias is not None else None
    w_out = next_layer.weight.detach().double()

    if cost_domain == "connection_graph" and isinstance(curr_layer, nn.Conv2d):
        h_in, w_in_sp, h_out, w_out_sp = model.conv_spatial_shapes[layer_idx]
        mult_in_uv, bias_mult_in = compute_conv2d_spatial_multiplicity(
            curr_layer, h_in, w_in_sp, h_out, w_out_sp, device=w_in.device
        )
        w_in_sq = w_in.pow(2) * mult_in_uv.view(1, 1, *mult_in_uv.shape)
        b_in_sq = (b_in.pow(2) * bias_mult_in) if b_in is not None else None
    else:
        w_in_sq = w_in.pow(2)
        b_in_sq = b_in.pow(2) if b_in is not None else None

    if cost_domain == "connection_graph" and isinstance(next_layer, nn.Conv2d):
        h_in2, w_in2, h_out2, w_out2 = model.conv_spatial_shapes[layer_idx + 1]
        mult_out_uv, _ = compute_conv2d_spatial_multiplicity(
            next_layer, h_in2, w_in2, h_out2, w_out2, device=w_out.device
        )
        w_out_sq = w_out.pow(2) * mult_out_uv.view(1, 1, *mult_out_uv.shape)
    else:
        w_out_sq = w_out.pow(2)

    if mode == "layer":
        a_val = w_in_sq.sum()
        if b_in_sq is not None:
            a_val = a_val + b_in_sq.sum()
        b_val = w_out_sq.sum()
        return a_val.unsqueeze(0), b_val.unsqueeze(0)

    if isinstance(curr_layer, nn.Conv2d):
        c_out = curr_layer.out_channels
        a_vec = w_in_sq.view(c_out, -1).sum(dim=1)
        if b_in_sq is not None:
            a_vec = a_vec + b_in_sq

        if isinstance(next_layer, nn.Conv2d):
            b_vec = w_out_sq.sum(dim=(0, 2, 3))
        elif isinstance(next_layer, nn.Linear):
            spatial_area = final_spatial * final_spatial
            w_out_reshaped = w_out_sq.view(next_layer.out_features, c_out, spatial_area)
            b_vec = w_out_reshaped.sum(dim=(0, 2))
        else:
            raise TypeError(f"Unsupported downstream layer type: {type(next_layer)}")
        return a_vec, b_vec

    elif isinstance(curr_layer, nn.Linear):
        h_out = curr_layer.out_features
        a_vec = w_in_sq.view(h_out, -1).sum(dim=1)
        if b_in_sq is not None:
            a_vec = a_vec + b_in_sq
        if not isinstance(next_layer, nn.Linear):
            raise TypeError("Linear layer must be followed by a Linear layer.")
        b_vec = w_out_sq.sum(dim=0)
        return a_vec, b_vec

    raise TypeError(f"Unsupported layer type: {type(curr_layer)}")


def _apply_scaling_to_layer_pair(
    curr_layer: nn.Module,
    next_layer: nn.Module,
    lambdas: torch.Tensor,
    mode: str,
    final_spatial: int,
) -> None:
    """
    Applies function-preserving scaling S_lambda between `curr_layer` and `next_layer`:
      W^(l) <- lambda * W^(l), b^(l) <- lambda * b^(l), W^(l+1) <- (1 / lambda) * W^(l+1).
    """
    with torch.no_grad():
        dtype = curr_layer.weight.dtype
        lam = lambdas.to(device=curr_layer.weight.device, dtype=dtype)
        inv_lam = (1.0 / lambdas).to(device=next_layer.weight.device, dtype=dtype)

        if mode == "layer":
            scalar_lam = lam[0]
            scalar_inv = inv_lam[0]
            curr_layer.weight.mul_(scalar_lam)
            if curr_layer.bias is not None:
                curr_layer.bias.mul_(scalar_lam)
            next_layer.weight.mul_(scalar_inv)
            return

        if isinstance(curr_layer, nn.Conv2d):
            c_out = curr_layer.out_channels
            curr_layer.weight.mul_(lam.view(c_out, 1, 1, 1))
            if curr_layer.bias is not None:
                curr_layer.bias.mul_(lam)

            if isinstance(next_layer, nn.Conv2d):
                next_layer.weight.mul_(inv_lam.view(1, c_out, 1, 1))
            elif isinstance(next_layer, nn.Linear):
                spatial_area = final_spatial * final_spatial
                w_next_view = next_layer.weight.view(next_layer.out_features, c_out, spatial_area)
                w_next_view.mul_(inv_lam.view(1, c_out, 1))
        elif isinstance(curr_layer, nn.Linear):
            h_out = curr_layer.out_features
            curr_layer.weight.mul_(lam.view(h_out, 1))
            if curr_layer.bias is not None:
                curr_layer.bias.mul_(lam)
            next_layer.weight.mul_(inv_lam.view(1, h_out))


def _compute_single_domain_diagnostics(
    model: SequentialCIFARCNN,
    conv_mode: str,
    fc_mode: str,
    cost_domain: str,
    eps: float = 1e-12,
) -> Dict[str, float]:
    n_total_units = 0
    n_eligible = 0
    n_zero_excluded = 0
    n_near_zero = 0

    raw_sq_deficit_all = 0.0
    raw_sq_deficit_elig = 0.0
    norm_deficits_elig: List[float] = []
    norm_deficits_all: List[float] = []

    layers = model.param_layers
    with torch.no_grad():
        for idx in range(len(layers) - 1):
            curr_l = layers[idx]
            mode = conv_mode if isinstance(curr_l, nn.Conv2d) else fc_mode
            a_vec, b_vec = _get_unit_costs(model, idx, mode=mode, cost_domain=cost_domain)

            for a_val, b_val in zip(a_vec.tolist(), b_vec.tolist()):
                n_total_units += 1
                diff_sq = (a_val - b_val) ** 2
                raw_sq_deficit_all += diff_sq
                denom = a_val + b_val
                norm_def_u = abs(a_val - b_val) / denom if denom > 0.0 else 0.0
                norm_deficits_all.append(norm_def_u)

                if a_val == 0.0 or b_val == 0.0:
                    n_zero_excluded += 1
                elif a_val <= eps or b_val <= eps:
                    n_near_zero += 1
                else:
                    n_eligible += 1
                    raw_sq_deficit_elig += diff_sq
                    norm_deficits_elig.append(norm_def_u)

    return {
        "n_total_units": int(n_total_units),
        "n_eligible_units": int(n_eligible),
        "n_zero_cost_excluded": int(n_zero_excluded),
        "n_near_zero_excluded": int(n_near_zero),
        "eligible_fraction": float(n_eligible / max(1, n_total_units)),
        "raw_l2_deficit_eligible": float(math.sqrt(raw_sq_deficit_elig)),
        "raw_l2_deficit_all": float(math.sqrt(raw_sq_deficit_all)),
        "norm_deficit_eligible_mean": float(np.mean(norm_deficits_elig)) if norm_deficits_elig else 0.0,
        "norm_deficit_eligible_max": float(np.max(norm_deficits_elig)) if norm_deficits_elig else 0.0,
        "norm_deficit_all_mean": float(np.mean(norm_deficits_all)) if norm_deficits_all else 0.0,
    }


def compute_cnn_balance_diagnostics(
    model: SequentialCIFARCNN,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
    cost_domain: str = "parameter",
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Computes balance diagnostics for the active `cost_domain` AND simultaneously logs
    both mean and max normalized deficits in `parameter` and `connection_graph` domains.
    """
    active_diag = _compute_single_domain_diagnostics(
        model, conv_mode=conv_mode, fc_mode=fc_mode, cost_domain=cost_domain, eps=eps
    )
    param_diag = (
        active_diag
        if cost_domain == "parameter"
        else _compute_single_domain_diagnostics(
            model, conv_mode=conv_mode, fc_mode=fc_mode, cost_domain="parameter", eps=eps
        )
    )
    graph_diag = (
        active_diag
        if cost_domain == "connection_graph"
        else _compute_single_domain_diagnostics(
            model, conv_mode=conv_mode, fc_mode=fc_mode, cost_domain="connection_graph", eps=eps
        )
    )

    out = dict(active_diag)
    out["param_domain_norm_deficit_mean"] = param_diag["norm_deficit_eligible_mean"]
    out["param_domain_norm_deficit_max"] = param_diag["norm_deficit_eligible_max"]
    out["graph_domain_norm_deficit_mean"] = graph_diag["norm_deficit_eligible_mean"]
    out["graph_domain_norm_deficit_max"] = graph_diag["norm_deficit_eligible_max"]
    return out


def partial_balance_cnn_once(
    model: SequentialCIFARCNN,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
    cost_domain: str = "parameter",
    eps: float = 1e-12,
) -> List[torch.Tensor]:
    """
    Single sequential forward balancing sweep from input toward output (Partial Balancing,
    Baldi et al. Section 11). Applies optimal L2 multiplier lambda^* = (B / A)^(1/4).
    """
    layers = model.param_layers
    applied_log_lambdas: List[torch.Tensor] = []

    with torch.no_grad():
        for idx in range(len(layers) - 1):
            curr_l = layers[idx]
            next_l = layers[idx + 1]
            mode = conv_mode if isinstance(curr_l, nn.Conv2d) else fc_mode
            a_vec, b_vec = _get_unit_costs(model, idx, mode=mode, cost_domain=cost_domain)

            lambdas = torch.ones_like(a_vec, dtype=torch.float64)
            eligible_mask = (a_vec > eps) & (b_vec > eps)
            lambdas[eligible_mask] = (b_vec[eligible_mask] / a_vec[eligible_mask]).pow(0.25)

            _apply_scaling_to_layer_pair(curr_l, next_l, lambdas, mode, model.final_spatial)
            applied_log_lambdas.append(torch.log(lambdas).detach().cpu())

    return applied_log_lambdas


def full_balance_cnn_until_convergence(
    model: SequentialCIFARCNN,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
    cost_domain: str = "parameter",
    eps: float = 1e-12,
    tol: float = 1e-7,
    max_iters: int = 300,
) -> Dict[str, Any]:
    iters = 0
    converged = False
    for iters in range(1, max_iters + 1):
        partial_balance_cnn_once(
            model, conv_mode=conv_mode, fc_mode=fc_mode, cost_domain=cost_domain, eps=eps
        )
        diag = compute_cnn_balance_diagnostics(
            model, conv_mode=conv_mode, fc_mode=fc_mode, cost_domain=cost_domain, eps=eps
        )
        if diag["norm_deficit_eligible_max"] <= tol:
            converged = True
            break
    diag = compute_cnn_balance_diagnostics(
        model, conv_mode=conv_mode, fc_mode=fc_mode, cost_domain=cost_domain, eps=eps
    )
    diag["iterations"] = int(iters)
    diag["converged"] = bool(converged)
    return diag


def apply_matched_random_rescaling_cnn(
    model: SequentialCIFARCNN,
    reference_log_lambdas: List[torch.Tensor],
    rng: torch.Generator,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
) -> None:
    layers = model.param_layers
    with torch.no_grad():
        for idx, ref_log_lam in enumerate(reference_log_lambdas):
            curr_l = layers[idx]
            next_l = layers[idx + 1]
            mode = conv_mode if isinstance(curr_l, nn.Conv2d) else fc_mode
            n_units = ref_log_lam.numel()

            if n_units > 1:
                perm = torch.randperm(n_units, generator=rng)
                shuffled = ref_log_lam[perm].clone()
            else:
                shuffled = ref_log_lam.clone()

            signs = torch.randint(0, 2, (n_units,), generator=rng, dtype=torch.int64) * 2 - 1
            random_log_lam = shuffled * signs.double()
            lambdas = torch.exp(random_log_lam)
            _apply_scaling_to_layer_pair(curr_l, next_l, lambdas, mode, model.final_spatial)


# =============================================================================
# 4. Statistical Analysis Pipeline (Stage 2 Factorial & Symmetric Stage 3 Contrasts)
# =============================================================================

def _paired_summary_stats(diffs: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": np.nan, "sd": np.nan, "se": np.nan,
            "ci95_halfwidth": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "t_stat": np.nan, "p_unadjusted": np.nan,
        }
    mean_val = float(np.mean(arr))
    if n < 2:
        return {
            "mean": mean_val, "sd": np.nan, "se": np.nan,
            "ci95_halfwidth": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "t_stat": np.nan, "p_unadjusted": np.nan,
        }
    sd_val = float(np.std(arr, ddof=1))
    se_val = float(sd_val / math.sqrt(n))
    if se_val <= 1e-15:
        return {
            "mean": mean_val, "sd": sd_val, "se": 0.0,
            "ci95_halfwidth": 0.0, "ci95_low": mean_val, "ci95_high": mean_val,
            "t_stat": np.nan, "p_unadjusted": np.nan,
        }
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    halfwidth = t_crit * se_val
    t_stat = mean_val / se_val
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    return {
        "mean": mean_val,
        "sd": sd_val,
        "se": se_val,
        "ci95_halfwidth": halfwidth,
        "ci95_low": mean_val - halfwidth,
        "ci95_high": mean_val + halfwidth,
        "t_stat": t_stat,
        "p_unadjusted": p_val,
    }


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    n_total = len(p_values)
    adjusted = [np.nan] * n_total
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    if not valid:
        return adjusted
    m = len(valid)
    valid_sorted = sorted(valid, key=lambda x: x[1])
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(valid_sorted):
        adj = min(1.0, p_val * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def compute_factorial_interaction_report(
    results_df: pd.DataFrame,
    metrics: List[str],
    primary_metric: str = "final_val_accuracy",
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        piv = results_df.pivot(index="seed", columns="group_code", values=metric)
        if not {"A", "B", "C", "D"}.issubset(set(piv.columns)):
            continue
        piv = piv.dropna(subset=["A", "B", "C", "D"])
        if piv.empty:
            continue
        a = piv["A"].to_numpy(dtype=np.float64)
        b = piv["B"].to_numpy(dtype=np.float64)
        c = piv["C"].to_numpy(dtype=np.float64)
        d = piv["D"].to_numpy(dtype=np.float64)

        did = (d - c) - (b - a)
        main_drop = 0.5 * ((b - a) + (d - c))
        main_bal = 0.5 * ((c - a) + (d - b))
        d_minus_b = d - b
        b_minus_a = b - a
        d_minus_c = d - c
        c_minus_a = c - a

        did_stats = _paired_summary_stats(did)
        db_stats = _paired_summary_stats(d_minus_b)
        ba_stats = _paired_summary_stats(b_minus_a)
        dc_stats = _paired_summary_stats(d_minus_c)
        ca_stats = _paired_summary_stats(c_minus_a)
        md_stats = _paired_summary_stats(main_drop)
        mb_stats = _paired_summary_stats(main_bal)

        rows.append({
            "metric": metric,
            "is_primary_endpoint": bool(metric == primary_metric),
            "n_seeds": int(len(piv)),
            "Group_A_mean": float(np.mean(a)),
            "Group_B_mean": float(np.mean(b)),
            "Group_C_mean": float(np.mean(c)),
            "Group_D_mean": float(np.mean(d)),
            "dropout_without_balance_B_minus_A": ba_stats["mean"],
            "dropout_with_balance_D_minus_C": dc_stats["mean"],
            "balance_without_dropout_C_minus_A": ca_stats["mean"],
            "balance_with_dropout_D_minus_B": db_stats["mean"],
            "D_minus_B_ci95_low": db_stats["ci95_low"],
            "D_minus_B_ci95_high": db_stats["ci95_high"],
            "D_minus_B_p_unadjusted": db_stats["p_unadjusted"],
            "main_effect_dropout": md_stats["mean"],
            "main_effect_dropout_p": md_stats["p_unadjusted"],
            "main_effect_balance": mb_stats["mean"],
            "main_effect_balance_p": mb_stats["p_unadjusted"],
            "interaction_DiD_mean": did_stats["mean"],
            "interaction_DiD_sd": did_stats["sd"],
            "interaction_DiD_se": did_stats["se"],
            "interaction_DiD_ci95_halfwidth": did_stats["ci95_halfwidth"],
            "interaction_DiD_ci95_low": did_stats["ci95_low"],
            "interaction_DiD_ci95_high": did_stats["ci95_high"],
            "interaction_DiD_t_stat": did_stats["t_stat"],
            "interaction_DiD_p_unadjusted": did_stats["p_unadjusted"],
        })

    rep_df = pd.DataFrame(rows)
    if not rep_df.empty:
        sec_mask = ~rep_df["is_primary_endpoint"]
        sec_p = rep_df.loc[sec_mask, "interaction_DiD_p_unadjusted"].tolist()
        rep_df["interaction_DiD_p_holm"] = np.nan
        rep_df.loc[sec_mask, "interaction_DiD_p_holm"] = _holm_bonferroni(sec_p)
    return rep_df


def compute_stage3_random_rescale_contrasts(
    controls_df: pd.DataFrame,
    metrics: List[str],
    primary_metric: str = "final_val_accuracy",
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in controls_df.columns:
            continue
        piv = controls_df.pivot(index="seed", columns="group_code", values=metric)
        if not {"C_ref", "C_rand", "D_ref", "D_rand"}.issubset(set(piv.columns)):
            continue
        piv = piv.dropna(subset=["C_ref", "C_rand", "D_ref", "D_rand"])
        if piv.empty:
            continue

        c_ref = piv["C_ref"].to_numpy(dtype=np.float64)
        c_rand = piv["C_rand"].to_numpy(dtype=np.float64)
        d_ref = piv["D_ref"].to_numpy(dtype=np.float64)
        d_rand = piv["D_rand"].to_numpy(dtype=np.float64)

        delta_c = c_ref - c_rand
        delta_d = d_ref - d_rand
        delta_inter = delta_d - delta_c

        dc_stats = _paired_summary_stats(delta_c)
        dd_stats = _paired_summary_stats(delta_d)
        di_stats = _paired_summary_stats(delta_inter)

        rows.append({
            "metric": metric,
            "is_primary_endpoint": bool(metric == primary_metric),
            "n_seeds": int(len(piv)),
            "C_ref_mean": float(np.mean(c_ref)),
            "C_rand_mean": float(np.mean(c_rand)),
            "D_ref_mean": float(np.mean(d_ref)),
            "D_rand_mean": float(np.mean(d_rand)),
            "delta_C_mean": dc_stats["mean"],
            "delta_C_sd": dc_stats["sd"],
            "delta_C_se": dc_stats["se"],
            "delta_C_ci95_halfwidth": dc_stats["ci95_halfwidth"],
            "delta_C_ci95_low": dc_stats["ci95_low"],
            "delta_C_ci95_high": dc_stats["ci95_high"],
            "delta_C_t_stat": dc_stats["t_stat"],
            "delta_C_p_unadjusted": dc_stats["p_unadjusted"],
            "delta_D_mean": dd_stats["mean"],
            "delta_D_sd": dd_stats["sd"],
            "delta_D_se": dd_stats["se"],
            "delta_D_ci95_halfwidth": dd_stats["ci95_halfwidth"],
            "delta_D_ci95_low": dd_stats["ci95_low"],
            "delta_D_ci95_high": dd_stats["ci95_high"],
            "delta_D_t_stat": dd_stats["t_stat"],
            "delta_D_p_unadjusted": dd_stats["p_unadjusted"],
            "delta_interaction_mean": di_stats["mean"],
            "delta_interaction_sd": di_stats["sd"],
            "delta_interaction_se": di_stats["se"],
            "delta_interaction_ci95_halfwidth": di_stats["ci95_halfwidth"],
            "delta_interaction_ci95_low": di_stats["ci95_low"],
            "delta_interaction_ci95_high": di_stats["ci95_high"],
            "delta_interaction_t_stat": di_stats["t_stat"],
            "delta_interaction_p_unadjusted": di_stats["p_unadjusted"],
        })

    rep_df = pd.DataFrame(rows)
    if not rep_df.empty:
        sec_mask = ~rep_df["is_primary_endpoint"]
        sec_p = rep_df.loc[sec_mask, "delta_interaction_p_unadjusted"].tolist()
        rep_df["delta_interaction_p_holm"] = np.nan
        rep_df.loc[sec_mask, "delta_interaction_p_holm"] = _holm_bonferroni(sec_p)
    return rep_df


# =============================================================================
# 5. Paired-Seed CIFAR-10 Dataset & Vectorized Deterministic Augmentation
# =============================================================================

class DeterministicAugmentedCIFARDataset(Dataset):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.norm = transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base_dataset[self.indices[idx]]
        return self.norm(img), int(label)

    @staticmethod
    def apply_paired_batch_augmentation(
        x: torch.Tensor,
        gen_aug: torch.Generator,
        padding: int = 4,
        pad_mode: str = "reflect",
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        padded = F.pad(x, (padding, padding, padding, padding), mode=pad_mode)
        max_offset = 2 * padding + 1
        top_offsets = torch.randint(0, max_offset, (b,), generator=gen_aug).to(
            device=x.device, dtype=torch.long
        )
        left_offsets = torch.randint(0, max_offset, (b,), generator=gen_aug).to(
            device=x.device, dtype=torch.long
        )
        flip_mask = (torch.rand((b,), generator=gen_aug) < 0.5).to(device=x.device)

        batch_idx = torch.arange(b, device=x.device, dtype=torch.long).view(b, 1, 1, 1)
        ch_idx = torch.arange(c, device=x.device, dtype=torch.long).view(1, c, 1, 1)
        row_idx = top_offsets.view(b, 1, 1, 1) + torch.arange(
            h, device=x.device, dtype=torch.long
        ).view(1, 1, h, 1)
        col_idx = left_offsets.view(b, 1, 1, 1) + torch.arange(
            w, device=x.device, dtype=torch.long
        ).view(1, 1, 1, w)

        patches = padded[batch_idx, ch_idx, row_idx, col_idx]
        flipped = torch.flip(patches, dims=[3])
        return torch.where(flip_mask.view(b, 1, 1, 1), flipped, patches)


def prepare_cifar10_splits(cfg: CNNExperimentConfig):
    to_tensor = transforms.ToTensor()
    full_train = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=to_tensor
    )
    test_set_raw = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=to_tensor
    )

    g_split = torch.Generator(device="cpu")
    g_split.manual_seed(cfg.split_seed)
    perm = torch.randperm(len(full_train), generator=g_split).tolist()
    train_idx = perm[: cfg.train_size]
    val_idx = perm[cfg.train_size : cfg.train_size + cfg.val_size]
    train_subset_eval_idx = train_idx[: cfg.train_subset_eval_size]

    train_ds = DeterministicAugmentedCIFARDataset(full_train, train_idx)
    train_subset_eval_ds = DeterministicAugmentedCIFARDataset(full_train, train_subset_eval_idx)
    val_ds = DeterministicAugmentedCIFARDataset(full_train, val_idx)
    test_ds = DeterministicAugmentedCIFARDataset(
        test_set_raw, list(range(len(test_set_raw)))
    )
    return train_ds, train_subset_eval_ds, val_ds, test_ds


# =============================================================================
# 6. Stage 1 Verification Gate: Mathematical & Numerical Unit Tests + JSON Report
# =============================================================================

def _brute_force_conv2d_graph_cost_reference(
    conv_layer: nn.Conv2d, h_in: int, w_in: int
) -> Tuple[float, int, int]:
    w = conv_layer.weight.detach().double().cpu()
    b = conv_layer.bias.detach().double().cpu() if conv_layer.bias is not None else None
    c_out, c_in, kh, kw = w.shape
    sh, sw = conv_layer.stride
    ph, pw = conv_layer.padding
    dh, dw = conv_layer.dilation

    with torch.no_grad():
        dummy_in = torch.zeros(
            (1, c_in, h_in, w_in), dtype=conv_layer.weight.dtype, device=conv_layer.weight.device
        )
        dummy_out = conv_layer(dummy_in)
        actual_h_out, actual_w_out = int(dummy_out.shape[2]), int(dummy_out.shape[3])

    formula_h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    formula_w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    assert (formula_h_out, formula_w_out) == (actual_h_out, actual_w_out), (
        f"Conv2d output shape mismatch: formula={(formula_h_out, formula_w_out)} "
        f"vs actual forward={(actual_h_out, actual_w_out)}"
    )
    h_out, w_out = actual_h_out, actual_w_out

    per_tap_sq_sum = w.pow(2).sum(dim=(0, 1))
    bias_sq_sum = float(b.pow(2).sum().item()) if b is not None else 0.0

    explicit_total = 0.0
    for oh in range(h_out):
        for ow in range(w_out):
            explicit_total += bias_sq_sum
            for u in range(kh):
                ih = oh * sh - ph + u * dh
                if 0 <= ih < h_in:
                    for v in range(kw):
                        iw = ow * sw - pw + v * dw
                        if 0 <= iw < w_in:
                            explicit_total += float(per_tap_sq_sum[u, v].item())

    ones = torch.ones((1, 1, h_in, w_in), dtype=torch.float64)
    unfolded = F.unfold(
        ones, kernel_size=(kh, kw), dilation=(dh, dw), padding=(ph, pw), stride=(sh, sw)
    )
    unfold_mult = unfolded.sum(dim=2).view(kh, kw)
    unfold_total = float((per_tap_sq_sum * unfold_mult).sum().item()) + bias_sq_sum * (h_out * w_out)
    assert abs(explicit_total - unfold_total) < 1e-9, (
        f"Brute-force vs. unfold mismatch: {explicit_total} vs {unfold_total}"
    )
    return explicit_total, h_out, w_out


def run_stage1_verification_tests(
    cfg: Optional[CNNExperimentConfig] = None,
    save_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the 5 mandatory Stage 1 verification unit tests and writes a structured
    JSON verification artifact (`stage1_verification_report.json`) to disk.
    """
    if cfg is None:
        cfg = CNNExperimentConfig()

    print("=" * 86)
    print("STAGE 1 GATE: MATHEMATICAL FEASIBILITY & NUMERICAL INVARIANCE UNIT TESTS (v1.2.3)")
    print("=" * 86)

    git_info = get_git_provenance()
    report: Dict[str, Any] = {
        "script_version": cfg.version,
        "source_script_sha256": get_source_script_sha256(),
        "git_provenance": git_info,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test1a_logit_invariance": [],
        "test1b_active_dropout_invariance_max_diff": 0.0,
        "test2a_per_transition_multiplier_diffs": {},
        "test2b_monotonic_convergence": [],
        "test3_eligibility_accounting": {},
        "test4_rng_and_policy_checks": {},
        "test5_statistical_pipeline_checks": {},
        "all_passed": False,
    }

    mode_combinations = [
        ("parameter", "layer", "layer"),
        ("parameter", "layer", "neuron"),
        ("parameter", "channel", "layer"),
        ("parameter", "channel", "neuron"),
        ("connection_graph", "layer", "layer"),
        ("connection_graph", "layer", "neuron"),
        ("connection_graph", "channel", "layer"),
        ("connection_graph", "channel", "neuron"),
    ]

    for cost_dom, conv_m, fc_m in mode_combinations:
        rng_streams = IsolatedRNGStreams.from_seed(101)
        model_f64 = SequentialCIFARCNN(
            in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
            fc_hidden=24, num_classes=10, use_dropout=False
        ).double()
        model_f64.reset_parameters_with_generator(rng_streams.gen_init)
        model_f64.eval()

        x_f64 = torch.randn(6, 3, 16, 16, generator=rng_streams.gen_loader, dtype=torch.float64)
        with torch.no_grad():
            logits_before_f64 = model_f64(x_f64).clone()
            partial_balance_cnn_once(
                model_f64, conv_mode=conv_m, fc_mode=fc_m, cost_domain=cost_dom
            )
            logits_after_partial_f64 = model_f64(x_f64).clone()
            conv_diag64 = full_balance_cnn_until_convergence(
                model_f64, conv_mode=conv_m, fc_mode=fc_m, cost_domain=cost_dom,
                tol=cfg.full_balance_tol_f64, max_iters=cfg.full_balance_max_iters
            )
            logits_after_full_f64 = model_f64(x_f64).clone()

        assert conv_diag64["converged"], (
            f"Float64 convergence failed for ({cost_dom}, {conv_m}, {fc_m}): "
            f"max deficit = {conv_diag64['norm_deficit_eligible_max']}"
        )
        diff_partial_f64 = float((logits_before_f64 - logits_after_partial_f64).abs().max().item())
        diff_full_f64 = float((logits_before_f64 - logits_after_full_f64).abs().max().item())
        assert diff_partial_f64 < cfg.logit_tol_f64, f"Float64 partial invariance failed: {diff_partial_f64}"
        assert diff_full_f64 < cfg.logit_tol_f64, f"Float64 full invariance failed: {diff_full_f64}"

        rng_streams32 = IsolatedRNGStreams.from_seed(202)
        model_f32 = SequentialCIFARCNN(
            in_channels=3, spatial_size=32, conv_channels=(16, 32, 64, 64),
            fc_hidden=64, num_classes=10, use_dropout=False
        ).float()
        model_f32.reset_parameters_with_generator(rng_streams32.gen_init)
        model_f32.eval()

        x_f32 = torch.randn(8, 3, 32, 32, generator=rng_streams32.gen_loader, dtype=torch.float32)
        with torch.no_grad():
            logits_before_f32 = model_f32(x_f32).clone()
            conv_diag32 = full_balance_cnn_until_convergence(
                model_f32, conv_mode=conv_m, fc_mode=fc_m, cost_domain=cost_dom,
                tol=cfg.full_balance_tol_f32, max_iters=cfg.full_balance_max_iters
            )
            logits_after_f32 = model_f32(x_f32).clone()

        assert conv_diag32["converged"], (
            f"Float32 convergence failed for ({cost_dom}, {conv_m}, {fc_m}): "
            f"max deficit = {conv_diag32['norm_deficit_eligible_max']}"
        )
        diff_f32 = float((logits_before_f32 - logits_after_f32).abs().max().item())
        assert diff_f32 < cfg.logit_tol_f32, (
            f"Float32 invariance failed ({diff_f32:.2e} >= {cfg.logit_tol_f32:.2e})"
        )
        report["test1a_logit_invariance"].append({
            "cost_domain": cost_dom,
            "conv_mode": conv_m,
            "fc_mode": fc_m,
            "f64_partial_max_diff": diff_partial_f64,
            "f64_full_max_diff": diff_full_f64,
            "f32_full_max_diff": diff_f32,
            "f64_iterations": int(conv_diag64["iterations"]),
            "f32_iterations": int(conv_diag32["iterations"]),
        })
        print(
            f"[PASS] Test 1a ({cost_dom:16s} | conv={conv_m:7s} | fc={fc_m:6s}): "
            f"f64 diff={diff_full_f64:.2e}, f32 diff={diff_f32:.2e} "
            f"(iters={conv_diag64['iterations']})"
        )

    max_drop_diff = 0.0
    for drop_mode in ["elementwise", "channel"]:
        for cost_dom, conv_m, fc_m in mode_combinations:
            rng_d1 = IsolatedRNGStreams.from_seed(777)
            rng_d2 = IsolatedRNGStreams.from_seed(777)
            model_drop = SequentialCIFARCNN(
                in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
                fc_hidden=24, num_classes=10, use_dropout=True, conv_dropout_mode=drop_mode
            ).double()
            model_drop.reset_parameters_with_generator(rng_d1.gen_init)
            model_drop.train()
            x_d = torch.randn(4, 3, 16, 16, generator=rng_d1.gen_loader, dtype=torch.float64)
            with torch.no_grad():
                out_drop_before = model_drop(x_d, dropout_rng=rng_d1.gen_dropout).clone()
                partial_balance_cnn_once(
                    model_drop, conv_mode=conv_m, fc_mode=fc_m, cost_domain=cost_dom
                )
                out_drop_after = model_drop(x_d, dropout_rng=rng_d2.gen_dropout).clone()
            diff_drop = float((out_drop_before - out_drop_after).abs().max().item())
            max_drop_diff = max(max_drop_diff, diff_drop)
            assert diff_drop < cfg.logit_tol_f64, (
                f"Fixed-mask dropout invariance failed ({drop_mode}, {cost_dom}, {conv_m}, {fc_m}): {diff_drop}"
            )
    report["test1b_active_dropout_invariance_max_diff"] = max_drop_diff
    print(
        "[PASS] Test 1b (Active Dropout Fixed-Mask Invariance): Verified across all 8 balance modes "
        "x 2 dropout modes"
    )

    test_conv_specs = [
        {"kernel_size": 3, "padding": 0, "stride": 1, "dilation": 1, "h_in": 10, "w_in": 12},
        {"kernel_size": 3, "padding": 1, "stride": 1, "dilation": 1, "h_in": 16, "w_in": 16},
        {"kernel_size": 3, "padding": 1, "stride": 2, "dilation": 1, "h_in": 15, "w_in": 15},
        {"kernel_size": 3, "padding": 2, "stride": 1, "dilation": 2, "h_in": 14, "w_in": 14},
    ]
    g_ref = torch.Generator(device="cpu").manual_seed(909)
    for spec in test_conv_specs:
        conv_test = nn.Conv2d(
            2, 3,
            kernel_size=spec["kernel_size"],
            padding=spec["padding"],
            stride=spec["stride"],
            dilation=spec["dilation"],
            bias=True,
        ).double()
        with torch.no_grad():
            conv_test.weight.copy_(torch.randn(conv_test.weight.shape, generator=g_ref, dtype=torch.float64))
            conv_test.bias.copy_(torch.randn(conv_test.bias.shape, generator=g_ref, dtype=torch.float64))

        ref_cost, h_out, w_out = _brute_force_conv2d_graph_cost_reference(
            conv_test, spec["h_in"], spec["w_in"]
        )
        mult_uv, bias_mult = compute_conv2d_spatial_multiplicity(
            conv_test, spec["h_in"], spec["w_in"], h_out, w_out, device=torch.device("cpu")
        )
        our_cost = float(
            (conv_test.weight.pow(2) * mult_uv.view(1, 1, *mult_uv.shape)).sum().item()
            + (conv_test.bias.pow(2) * bias_mult).sum().item()
        )
        assert abs(ref_cost - our_cost) < 1e-9, (
            f"Multiplicity graph cost mismatch for {spec}: ref={ref_cost} vs our={our_cost}"
        )

    rng_net_ref = IsolatedRNGStreams.from_seed(910)
    net_ref = SequentialCIFARCNN(
        in_channels=3, spatial_size=16, conv_channels=(4, 6, 8, 8), fc_hidden=12, num_classes=10
    ).double()
    net_ref.reset_parameters_with_generator(rng_net_ref.gen_init)
    indep_net_graph_cost = 0.0
    for idx_c, conv_l in enumerate([net_ref.conv1, net_ref.conv2, net_ref.conv3, net_ref.conv4]):
        h_in, w_in, _, _ = net_ref.conv_spatial_shapes[idx_c]
        c_cost, _, _ = _brute_force_conv2d_graph_cost_reference(conv_l, h_in, w_in)
        indep_net_graph_cost += c_cost
    for fc_l in [net_ref.fc1, net_ref.fc2]:
        indep_net_graph_cost += float(fc_l.weight.pow(2).sum().item() + fc_l.bias.pow(2).sum().item())
    our_net_graph_cost = compute_total_l2_regularizer(net_ref, cost_domain="connection_graph")
    assert abs(indep_net_graph_cost - our_net_graph_cost) < 1e-9, (
        f"Full CNN connection_graph regularizer mismatch: {indep_net_graph_cost} vs {our_net_graph_cost}"
    )

    net_p = SequentialCIFARCNN(
        in_channels=3, spatial_size=16, conv_channels=(4, 6, 8, 8), fc_hidden=12, num_classes=10
    ).double()
    net_g = SequentialCIFARCNN(
        in_channels=3, spatial_size=16, conv_channels=(4, 6, 8, 8), fc_hidden=12, num_classes=10
    ).double()
    net_p.load_state_dict(net_ref.state_dict())
    net_g.load_state_dict(net_ref.state_dict())
    logs_p = partial_balance_cnn_once(net_p, conv_mode="layer", fc_mode="layer", cost_domain="parameter")
    logs_g = partial_balance_cnn_once(net_g, conv_mode="layer", fc_mode="layer", cost_domain="connection_graph")

    transition_names = [
        "L1(conv1->conv2)",
        "L2(conv2->pool->conv3)",
        "L3(conv3->conv4)",
        "L4(conv4->pool->fc1)",
        "L5(fc1->fc2)",
    ]
    per_transition_diffs = [
        float((lp - lg).abs().max().item()) for lp, lg in zip(logs_p, logs_g)
    ]
    assert per_transition_diffs[1] > 0.15, (
        f"Expected large multiplier shift at L2(conv2->pool->conv3), got {per_transition_diffs[1]}"
    )
    assert per_transition_diffs[3] > 0.50, (
        f"Expected large multiplier shift at L4(conv4->pool->fc1), got {per_transition_diffs[3]}"
    )
    assert per_transition_diffs[0] > 1e-5 and per_transition_diffs[2] > 1e-5, (
        f"Expected non-zero border-multiplicity shift on unpooled conv transitions: {per_transition_diffs}"
    )
    report["test2a_per_transition_multiplier_diffs"] = {
        name: val for name, val in zip(transition_names, per_transition_diffs)
    }
    diff_summary_str = ", ".join(
        f"{name}={val:.4f}" for name, val in zip(transition_names, per_transition_diffs)
    )
    print(
        f"[PASS] Test 2a (Independent Graph Reference & Per-Transition Multiplier Shifts):\n"
        f"       |log(lam_param) - log(lam_graph)|: {diff_summary_str}"
    )

    for cost_dom in ["parameter", "connection_graph"]:
        for balance_mode in ["layer", "channel"]:
            fc_mode = "layer" if balance_mode == "layer" else "neuron"
            rng_streams = IsolatedRNGStreams.from_seed(303)
            model_conv = SequentialCIFARCNN(
                in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
                fc_hidden=24, num_classes=10, use_dropout=False
            ).double()
            model_conv.reset_parameters_with_generator(rng_streams.gen_init)

            extreme_scales = [
                torch.tensor([4.0], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
                torch.tensor([3.5], dtype=torch.float64),
                torch.tensor([0.3], dtype=torch.float64),
                torch.tensor([2.5], dtype=torch.float64),
            ]
            for idx, scale in enumerate(extreme_scales):
                _apply_scaling_to_layer_pair(
                    model_conv.param_layers[idx],
                    model_conv.param_layers[idx + 1],
                    scale,
                    mode="layer",
                    final_spatial=model_conv.final_spatial,
                )

            r_initial = compute_total_l2_regularizer(model_conv, cost_domain=cost_dom)
            r_prev = r_initial
            for _ in range(30):
                partial_balance_cnn_once(
                    model_conv, conv_mode=balance_mode, fc_mode=fc_mode, cost_domain=cost_dom
                )
                r_curr = compute_total_l2_regularizer(model_conv, cost_domain=cost_dom)
                assert r_curr <= r_prev + 1e-10, (
                    f"[{cost_dom}, {balance_mode}] Regularizer increased: {r_curr} > {r_prev}"
                )
                r_prev = r_curr

            diag_final = full_balance_cnn_until_convergence(
                model_conv, conv_mode=balance_mode, fc_mode=fc_mode, cost_domain=cost_dom,
                tol=cfg.full_balance_tol_f64, max_iters=cfg.full_balance_max_iters
            )
            r_final = compute_total_l2_regularizer(model_conv, cost_domain=cost_dom)
            assert diag_final["converged"], f"Failed to converge in ({cost_dom}, {balance_mode})"
            assert r_final < r_initial, "Regularizer did not strictly decrease on imbalanced CNN"
            report["test2b_monotonic_convergence"].append({
                "cost_domain": cost_dom,
                "balance_mode": balance_mode,
                "r_initial": r_initial,
                "r_final": r_final,
                "final_max_deficit": float(diag_final["norm_deficit_eligible_max"]),
            })
            print(
                f"[PASS] Test 2b ({cost_dom:16s} | {balance_mode:7s}): Monotonic R(W) reduction "
                f"({r_initial:.2f} -> {r_final:.2f}) & converged to {diag_final['norm_deficit_eligible_max']:.2e}"
            )

    rng_streams = IsolatedRNGStreams.from_seed(404)
    model_zero = SequentialCIFARCNN(
        in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
        fc_hidden=24, num_classes=10, use_dropout=False
    ).double()
    model_zero.reset_parameters_with_generator(rng_streams.gen_init)
    with torch.no_grad():
        model_zero.conv1.weight[0].zero_()
        model_zero.conv1.bias[0].zero_()
        model_zero.conv2.weight[:, 1, :, :].zero_()
        model_zero.conv1.weight[2].fill_(1e-9)
        model_zero.conv1.bias[2].fill_(1e-9)

    diag_z = compute_cnn_balance_diagnostics(
        model_zero, conv_mode="channel", fc_mode="neuron", cost_domain="parameter", eps=1e-12
    )
    assert diag_z["n_zero_cost_excluded"] == 2, f"Expected 2 zero-cost exclusions, got {diag_z['n_zero_cost_excluded']}"
    assert diag_z["n_near_zero_excluded"] == 1, f"Expected 1 near-zero exclusion, got {diag_z['n_near_zero_excluded']}"
    partial_balance_cnn_once(model_zero, conv_mode="channel", fc_mode="neuron", cost_domain="parameter", eps=1e-12)
    assert torch.all(model_zero.conv1.weight[0] == 0.0), "Zero incoming filter was modified"
    assert torch.all(model_zero.conv2.weight[:, 1, :, :] == 0.0), "Zero outgoing slice was modified"
    report["test3_eligibility_accounting"] = {
        "n_total_units": int(diag_z["n_total_units"]),
        "n_zero_cost_excluded": int(diag_z["n_zero_cost_excluded"]),
        "n_near_zero_excluded": int(diag_z["n_near_zero_excluded"]),
    }
    print(
        f"[PASS] Test 3 (Eligibility Diagnostics): Verified {diag_z['n_zero_cost_excluded']} zero-cost "
        f"& {diag_z['n_near_zero_excluded']} near-zero exclusions out of {diag_z['n_total_units']} units"
    )

    test_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        test_devices.append(torch.device("cuda"))

    for dev in test_devices:
        streams_a = IsolatedRNGStreams.from_seed(505, device=dev)
        streams_b = IsolatedRNGStreams.from_seed(505, device=dev)
        streams_d = IsolatedRNGStreams.from_seed(505, device=dev)

        net_a = SequentialCIFARCNN(spatial_size=16, conv_channels=(8, 8, 16, 16), fc_hidden=16, use_dropout=False).to(dev)
        net_b = SequentialCIFARCNN(spatial_size=16, conv_channels=(8, 8, 16, 16), fc_hidden=16, use_dropout=True).to(dev)
        net_d = SequentialCIFARCNN(spatial_size=16, conv_channels=(8, 8, 16, 16), fc_hidden=16, use_dropout=True).to(dev)

        net_a.reset_parameters_with_generator(streams_a.gen_init)
        net_b.reset_parameters_with_generator(streams_b.gen_init)
        net_d.reset_parameters_with_generator(streams_d.gen_init)

        for (na, pa), (nb, pb), (nd, pd_param) in zip(
            net_a.named_parameters(), net_b.named_parameters(), net_d.named_parameters()
        ):
            assert torch.equal(pa, pb) and torch.equal(pb, pd_param), f"Parameter mismatch at init on {na}"

        synth_x = torch.randn(32, 3, 16, 16)
        synth_y = torch.randint(0, 10, (32,))
        ds = torch.utils.data.TensorDataset(synth_x, synth_y)
        loader_a = DataLoader(ds, batch_size=8, shuffle=True, generator=streams_a.gen_loader)
        loader_b = DataLoader(ds, batch_size=8, shuffle=True, generator=streams_b.gen_loader)
        loader_d = DataLoader(ds, batch_size=8, shuffle=True, generator=streams_d.gen_loader)

        net_a.train()
        net_b.train()
        net_d.train()
        for epoch in range(2):
            batches_a = list(loader_a)
            batches_b = list(loader_b)
            batches_d = list(loader_d)
            for (xa, ya), (xb, yb), (xd, yd) in zip(batches_a, batches_b, batches_d):
                assert torch.equal(xa, xb) and torch.equal(xb, xd), f"Batch X diverged at epoch {epoch}"
                assert torch.equal(ya, yb) and torch.equal(yb, yd), f"Batch Y diverged at epoch {epoch}"

                aug_a = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(xa, streams_a.gen_aug)
                _ = torch.rand(10, generator=streams_b.gen_rescale)
                aug_b = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(xb, streams_b.gen_aug)
                aug_d = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(xd, streams_d.gen_aug)
                assert torch.equal(aug_a, aug_b) and torch.equal(aug_b, aug_d), "Paired augmentation diverged"

                _ = net_a(aug_a.to(dev), dropout_rng=None)
                out_b = net_b(aug_b.to(dev), dropout_rng=streams_b.gen_dropout)
                out_d = net_d(aug_d.to(dev), dropout_rng=streams_d.gen_dropout)
                assert torch.equal(out_b, out_d), f"Paired B-vs-D dropout outputs diverged on {dev}"

    assert set(cfg.smoke_seeds).isdisjoint(set(cfg.pilot_seeds)), "Smoke and pilot seeds overlap!"
    assert set(cfg.pilot_seeds).isdisjoint(set(cfg.full_seeds)), "Pilot and full confirmatory seeds overlap!"
    assert set(cfg.smoke_seeds).isdisjoint(set(cfg.full_seeds)), "Smoke and full seeds overlap!"

    cfg_s2_full = CNNExperimentConfig(stage=2, run_mode="full")
    cfg_s3_full = CNNExperimentConfig(stage=3, run_mode="full")
    cfg_s3_leak_attempt = CNNExperimentConfig(
        stage=3, run_mode="full", allow_nonconfirmatory_test_eval=True
    )
    cfg_s3_conf = CNNExperimentConfig(stage=3, run_mode="full", stage3_confirmatory_test_eval=True)
    assert cfg_s2_full.should_evaluate_test_set() is True, "Stage 2 full should evaluate test set"
    assert cfg_s3_full.should_evaluate_test_set() is False, "Stage 3 full must withhold test set by default"
    assert cfg_s3_leak_attempt.should_evaluate_test_set() is False, (
        "Generic allow_nonconfirmatory_test_eval must NOT unlock Stage 3 test set!"
    )
    assert cfg_s3_conf.should_evaluate_test_set() is True, "Stage 3 confirmatory flag should unlock test set"
    assert cfg_s2_full.run_tag() != cfg_s3_full.run_tag(), "Stage 2 and Stage 3 run_tags collided!"
    assert cfg_s3_full.run_tag() != cfg_s3_conf.run_tag(), "Config fingerprint failed to distinguish test policy!"
    report["test4_rng_and_policy_checks"] = {
        "tested_devices": [str(d) for d in test_devices],
        "stage3_generic_override_blocked": True,
        "seed_tiers_disjoint": True,
    }
    print(
        f"[PASS] Test 4 (RNG, B-vs-D Dropout Isolation on {[str(d) for d in test_devices]}, "
        "Strict Stage 3 Test Policy & Config Hash): Verified"
    )

    synth_rows = []
    synth_s3_rows = []
    for s_idx, seed in enumerate([43, 44, 45, 46, 47]):
        jitter = (s_idx - 2) * 0.0004
        synth_rows.extend([
            {"seed": seed, "group_code": "A", "final_val_accuracy": 0.720000 + jitter, "val_loss": 0.850},
            {"seed": seed, "group_code": "B", "final_val_accuracy": 0.710000 + jitter * 1.1, "val_loss": 0.870},
            {"seed": seed, "group_code": "C", "final_val_accuracy": 0.735000 + jitter * 0.9, "val_loss": 0.810},
            {"seed": seed, "group_code": "D", "final_val_accuracy": 0.742000 + jitter * 1.2, "val_loss": 0.790},
        ])
        synth_s3_rows.extend([
            {"seed": seed, "group_code": "C_ref",  "final_val_accuracy": 0.735000 + jitter},
            {"seed": seed, "group_code": "C_rand", "final_val_accuracy": 0.722000 + jitter * 0.8},
            {"seed": seed, "group_code": "D_ref",  "final_val_accuracy": 0.742000 + jitter * 1.1},
            {"seed": seed, "group_code": "D_rand", "final_val_accuracy": 0.711000 + jitter * 0.9},
        ])
    rep = compute_factorial_interaction_report(
        pd.DataFrame(synth_rows), metrics=["final_val_accuracy", "val_loss"], primary_metric="final_val_accuracy"
    )
    row_acc = rep[rep["metric"] == "final_val_accuracy"].iloc[0]
    expected_did = (0.742 - 0.735) - (0.710 - 0.720)
    assert abs(row_acc["interaction_DiD_mean"] - expected_did) < 1e-12

    rep_s3 = compute_stage3_random_rescale_contrasts(
        pd.DataFrame(synth_s3_rows), metrics=["final_val_accuracy"], primary_metric="final_val_accuracy"
    )
    row_s3 = rep_s3.iloc[0]
    expected_s3_inter = (0.742 - 0.711) - (0.735 - 0.722)
    assert abs(row_s3["delta_interaction_mean"] - expected_s3_inter) < 1e-12
    for prefix in ["delta_C", "delta_D", "delta_interaction"]:
        for suffix in ["mean", "sd", "se", "ci95_halfwidth", "ci95_low", "ci95_high", "t_stat", "p_unadjusted"]:
            col_name = f"{prefix}_{suffix}"
            assert col_name in rep_s3.columns, f"Missing symmetric Stage 3 column: {col_name}"

    report["test5_statistical_pipeline_checks"] = {
        "stage2_synthetic_did_pp": float(row_acc["interaction_DiD_mean"] * 100.0),
        "stage3_synthetic_delta_interaction_pp": float(row_s3["delta_interaction_mean"] * 100.0),
    }
    report["all_passed"] = True

    if save_report_path is None:
        os.makedirs(cfg.output_root, exist_ok=True)
        save_report_path = os.path.join(cfg.output_root, "stage1_verification_report.json")
    with open(save_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[PASS] Test 5 (Stage 2 DiD & Symmetric Stage 3 Contrasts): Verified Stage 2 DiD = "
        f"{row_acc['interaction_DiD_mean']*100:+.3f} pp and Stage 3 delta_interaction = "
        f"{row_s3['delta_interaction_mean']*100:+.3f} pp\n"
        f"       Saved Stage 1 verification report to: {save_report_path}"
    )
    print("=" * 86)
    print("ALL STAGE 1 MATHEMATICAL & NUMERICAL VERIFICATION GATES PASSED.")
    print("=" * 86)
    return report


# =============================================================================
# 7. Stage 2 & Stage 3 Training and Evaluation Engine
# =============================================================================

@torch.no_grad()
def evaluate_cnn(model: SequentialCIFARCNN, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb, dropout_rng=None)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_correct += int((preds == yb).sum().item())
        total_samples += int(yb.numel())
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


def train_and_evaluate_condition(
    cfg: CNNExperimentConfig,
    seed: int,
    group_code: str,
    method_name: str,
    use_dropout: bool,
    balance_schedule: str,
    train_ds: DeterministicAugmentedCIFARDataset,
    train_subset_eval_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    device: torch.device,
    conv_dropout_override: Optional[float] = None,
    fc_dropout_override: Optional[float] = None,
    reference_epoch_log_lambdas: Optional[List[List[torch.Tensor]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[List[torch.Tensor]]]:
    rng = IsolatedRNGStreams.from_seed(seed, device=device)
    c_drop = cfg.conv_dropout_rate if conv_dropout_override is None else conv_dropout_override
    f_drop = cfg.fc_dropout_rate if fc_dropout_override is None else fc_dropout_override

    model = SequentialCIFARCNN(
        in_channels=3,
        spatial_size=32,
        conv_channels=cfg.conv_channels,
        fc_hidden=cfg.fc_hidden,
        num_classes=cfg.num_classes,
        use_dropout=use_dropout,
        conv_dropout_mode=cfg.conv_dropout_mode,
        conv_dropout_rate=c_drop,
        fc_dropout_rate=f_drop,
    ).to(device)
    model.reset_parameters_with_generator(rng.gen_init)
    n_params = model.count_trainable_parameters()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=rng.gen_loader,
        num_workers=0,
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.learning_rate, momentum=0.0, weight_decay=cfg.weight_decay
    )

    epochs = cfg.active_epochs()
    cfg_hash = cfg.config_fingerprint()
    r_tag = cfg.run_tag()
    git_info = get_git_provenance()
    script_sha = get_source_script_sha256()
    unit_type_str = cfg.constraint_unit_type_label()
    curve_rows: List[Dict[str, Any]] = []
    recorded_log_lambdas: List[List[torch.Tensor]] = []
    balance_overhead_sec = 0.0
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0

        for xb, yb in train_loader:
            if cfg.use_augmentation:
                xb = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(
                    xb, rng.gen_aug, padding=cfg.aug_padding, pad_mode=cfg.aug_pad_mode
                )
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb, dropout_rng=rng.gen_dropout if use_dropout else None)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_train_loss += float(loss.item()) * int(yb.numel())
                epoch_train_correct += int((logits.argmax(dim=1) == yb).sum().item())
                epoch_train_samples += int(yb.numel())

        t_bal0 = time.perf_counter()
        if balance_schedule == "interleaved":
            step_logs = partial_balance_cnn_once(
                model,
                conv_mode=cfg.balance_mode,
                fc_mode=cfg.fc_balance_mode,
                cost_domain=cfg.cost_domain,
                eps=cfg.balance_eps,
            )
            recorded_log_lambdas.append(step_logs)
        elif balance_schedule == "random_rescaled":
            if reference_epoch_log_lambdas is None or len(reference_epoch_log_lambdas) < epoch:
                raise ValueError("reference_epoch_log_lambdas required for random_rescaled control")
            apply_matched_random_rescaling_cnn(
                model,
                reference_epoch_log_lambdas[epoch - 1],
                rng=rng.gen_rescale,
                conv_mode=cfg.balance_mode,
                fc_mode=cfg.fc_balance_mode,
            )
        balance_overhead_sec += time.perf_counter() - t_bal0

        train_mode_loss = epoch_train_loss / max(1, epoch_train_samples)
        train_mode_acc = epoch_train_correct / max(1, epoch_train_samples)

        train_sub_loss, train_sub_acc = evaluate_cnn(model, train_subset_eval_loader, device)
        val_loss, val_acc = evaluate_cnn(model, val_loader, device)

        diag = compute_cnn_balance_diagnostics(
            model,
            conv_mode=cfg.balance_mode,
            fc_mode=cfg.fc_balance_mode,
            cost_domain=cfg.cost_domain,
            eps=cfg.balance_eps,
        )
        r_param = compute_total_l2_regularizer(model, cost_domain="parameter")
        r_graph = compute_total_l2_regularizer(model, cost_domain="connection_graph")

        curve_rows.append({
            "script_version": cfg.version,
            "stage": int(cfg.stage),
            "config_fingerprint": cfg_hash,
            "run_tag": r_tag,
            "run_mode": cfg.run_mode,
            "seed": int(seed),
            "group_code": group_code,
            "method": method_name,
            "cost_domain": cfg.cost_domain,
            "balance_mode": cfg.balance_mode,
            "fc_balance_mode": cfg.fc_balance_mode,
            "balance_constraint_unit_type": unit_type_str,
            "conv_dropout_mode": cfg.conv_dropout_mode,
            "epoch": int(epoch),
            "train_mode_augmented_loss": float(train_mode_loss),
            "train_mode_augmented_accuracy": float(train_mode_acc),
            "train_subset_eval_loss": float(train_sub_loss),
            "train_subset_eval_accuracy": float(train_sub_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "generalization_gap": float(train_sub_acc - val_acc),
            "train_mode_minus_val_accuracy": float(train_mode_acc - val_acc),
            "parameter_squared_cost": float(r_param),
            "connection_graph_squared_cost": float(r_graph),
            "l2_regularizer_param_norm": float(r_param),
            "l2_regularizer_graph_norm": float(r_graph),
            "norm_deficit_eligible_mean": float(diag["norm_deficit_eligible_mean"]),
            "norm_deficit_eligible_max": float(diag["norm_deficit_eligible_max"]),
            "param_domain_norm_deficit_mean": float(diag["param_domain_norm_deficit_mean"]),
            "param_domain_norm_deficit_max": float(diag["param_domain_norm_deficit_max"]),
            "graph_domain_norm_deficit_mean": float(diag["graph_domain_norm_deficit_mean"]),
            "graph_domain_norm_deficit_max": float(diag["graph_domain_norm_deficit_max"]),
            "n_eligible_balance_constraints": int(diag["n_eligible_units"]),
            "eligible_fraction": float(diag["eligible_fraction"]),
        })

    total_train_sec = time.perf_counter() - t_start
    final_train_sub_loss = curve_rows[-1]["train_subset_eval_loss"]
    final_train_sub_acc = curve_rows[-1]["train_subset_eval_accuracy"]
    final_val_loss = curve_rows[-1]["val_loss"]
    final_val_acc = curve_rows[-1]["val_accuracy"]

    if test_loader is not None:
        final_test_loss, final_test_acc = evaluate_cnn(model, test_loader, device)
    else:
        final_test_loss, final_test_acc = np.nan, np.nan

    final_diag = compute_cnn_balance_diagnostics(
        model,
        conv_mode=cfg.balance_mode,
        fc_mode=cfg.fc_balance_mode,
        cost_domain=cfg.cost_domain,
        eps=cfg.balance_eps,
    )

    if cfg.save_checkpoints:
        ckpt_dir = os.path.join(cfg.run_output_dir(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir,
            f"ckpt_{r_tag}_seed{seed}_{group_code}.pt",
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "config_fingerprint": cfg_hash,
                "source_script_sha256": script_sha,
                "git_provenance": git_info,
                "seed": seed,
                "group_code": group_code,
            },
            ckpt_path,
        )

    early_ep = min(10, max(1, epochs // 3))
    early_val_acc = next(r["val_accuracy"] for r in curve_rows if r["epoch"] == early_ep)

    if len(curve_rows) >= 5:
        val_delta_last5 = abs(curve_rows[-1]["val_accuracy"] - curve_rows[-5]["val_accuracy"])
    else:
        val_delta_last5 = np.nan

    r_param_final = float(compute_total_l2_regularizer(model, "parameter"))
    r_graph_final = float(compute_total_l2_regularizer(model, "connection_graph"))

    summary_row = {
        "script_version": cfg.version,
        "source_script_sha256": script_sha,
        "git_commit": git_info["git_commit_short"],
        "git_commit_full": git_info["git_commit_full"],
        "git_working_tree_dirty": bool(git_info["git_working_tree_dirty"]),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "deterministic_enabled": bool(cfg.deterministic),
        "deterministic_warn_only": bool(cfg.deterministic_warn_only),
        "deterministic_strict": bool(cfg.deterministic and not cfg.deterministic_warn_only),
        "stage": int(cfg.stage),
        "config_fingerprint": cfg_hash,
        "run_tag": r_tag,
        "run_mode": cfg.run_mode,
        "test_set_evaluated": bool(test_loader is not None),
        "split_seed": int(cfg.split_seed),
        "seed": int(seed),
        "group_code": group_code,
        "method": method_name,
        "model_architecture": f"CustomSequentialVGG_Conv{list(cfg.conv_channels)}_FC{cfg.fc_hidden}",
        "model_param_count": int(n_params),
        "cost_domain": cfg.cost_domain,
        "balance_mode": cfg.balance_mode,
        "fc_balance_mode": cfg.fc_balance_mode,
        "balance_constraint_unit_type": unit_type_str,
        "use_dropout": bool(use_dropout),
        "conv_dropout_mode": cfg.conv_dropout_mode,
        "conv_dropout_rate": float(c_drop if use_dropout else 0.0),
        "fc_dropout_rate": float(f_drop if use_dropout else 0.0),
        "balance_schedule": balance_schedule,
        "optimizer": "Stateless_SGD",
        "learning_rate": float(cfg.learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
        "use_augmentation": bool(cfg.use_augmentation),
        "aug_padding": int(cfg.aug_padding),
        "aug_pad_mode": cfg.aug_pad_mode,
        "epochs": int(epochs),
        "early_epoch_idx": int(early_ep),
        "early_val_accuracy": float(early_val_acc),
        "final_train_mode_augmented_accuracy": float(curve_rows[-1]["train_mode_augmented_accuracy"]),
        "final_train_subset_eval_accuracy": float(final_train_sub_acc),
        "final_train_subset_eval_loss": float(final_train_sub_loss),
        "final_val_accuracy": float(final_val_acc),
        "final_val_loss": float(final_val_loss),
        "final_test_accuracy": float(final_test_acc),
        "test_accuracy": float(final_test_acc),
        "final_test_loss": float(final_test_loss),
        "generalization_gap": float(final_train_sub_acc - final_val_acc),
        "train_mode_minus_val_accuracy": float(curve_rows[-1]["train_mode_minus_val_accuracy"]),
        "val_abs_change_last_5_epochs": float(val_delta_last5),
        "parameter_squared_cost": r_param_final,
        "connection_graph_squared_cost": r_graph_final,
        "l2_regularizer_param_norm": r_param_final,
        "l2_regularizer_graph_norm": r_graph_final,
        "norm_deficit_eligible_mean": float(final_diag["norm_deficit_eligible_mean"]),
        "norm_deficit_eligible_max": float(final_diag["norm_deficit_eligible_max"]),
        "param_domain_norm_deficit_mean": float(final_diag["param_domain_norm_deficit_mean"]),
        "param_domain_norm_deficit_max": float(final_diag["param_domain_norm_deficit_max"]),
        "graph_domain_norm_deficit_mean": float(final_diag["graph_domain_norm_deficit_mean"]),
        "graph_domain_norm_deficit_max": float(final_diag["graph_domain_norm_deficit_max"]),
        "raw_l2_deficit_eligible": float(final_diag["raw_l2_deficit_eligible"]),
        "n_eligible_balance_constraints": int(final_diag["n_eligible_units"]),
        "n_eligible_units": int(final_diag["n_eligible_units"]),
        "n_zero_cost_excluded": int(final_diag["n_zero_cost_excluded"]),
        "n_near_zero_excluded": int(final_diag["n_near_zero_excluded"]),
        "training_time_sec": float(total_train_sec),
        "balance_overhead_sec": float(balance_overhead_sec),
    }
    return summary_row, curve_rows, recorded_log_lambdas


def save_provenance_manifest(cfg: CNNExperimentConfig, device: torch.device) -> str:
    git_info = get_git_provenance()
    if cfg.require_clean_git and git_info["git_working_tree_dirty"]:
        raise RuntimeError(
            f"Git working tree is dirty ({git_info['git_dirty_files']}) and "
            "--require-clean-git was specified. Commit your changes before running."
        )
    manifest_path = cfg.output_path(f"stage{cfg.stage}_provenance_manifest.json")
    manifest = {
        "script_version": cfg.version,
        "source_script_sha256": get_source_script_sha256(),
        "stage": int(cfg.stage),
        "config_fingerprint": cfg.config_fingerprint(),
        "run_tag": cfg.run_tag(),
        "git_commit": git_info["git_commit_short"],
        "git_commit_full": git_info["git_commit_full"],
        "git_working_tree_dirty": bool(git_info["git_working_tree_dirty"]),
        "git_dirty_files": git_info["git_dirty_files"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "deterministic_enabled": bool(cfg.deterministic),
        "deterministic_warn_only": bool(cfg.deterministic_warn_only),
        "deterministic_strict": bool(cfg.deterministic and not cfg.deterministic_warn_only),
        "test_set_evaluated": cfg.should_evaluate_test_set(cfg.stage),
        "active_seeds": cfg.active_seeds(),
        "active_epochs": cfg.active_epochs(),
        "config": asdict(cfg),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def run_stage2_cifar10_factorial(cfg: CNNExperimentConfig) -> None:
    cfg.stage = 2
    configure_reproducibility(cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    s1_report_path = cfg.output_path("stage1_verification_report.json")
    run_stage1_verification_tests(cfg, save_report_path=s1_report_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = save_provenance_manifest(cfg, device)
    seeds = cfg.active_seeds()
    eval_test = cfg.should_evaluate_test_set(for_stage=2)

    print(
        f"\nSTAGE 2: CIFAR-10 2x2 FACTORIAL EXPERIMENT\n"
        f"  run_tag          = {cfg.run_tag()}\n"
        f"  config_hash      = {cfg.config_fingerprint()}\n"
        f"  output_dir       = {cfg.run_output_dir()}\n"
        f"  seeds            = {seeds} (epochs={cfg.active_epochs()})\n"
        f"  cost_domain      = {cfg.cost_domain} | balance_mode = {cfg.balance_mode} | fc_mode = {cfg.fc_balance_mode}\n"
        f"  conv_dropout     = {cfg.conv_dropout_mode} (p_conv={cfg.conv_dropout_rate}, p_fc={cfg.fc_dropout_rate})\n"
        f"  evaluate_test    = {eval_test}\n"
        f"  device           = {device}"
    )

    train_ds, train_subset_eval_ds, val_ds, test_ds = prepare_cifar10_splits(cfg)
    train_subset_eval_loader = DataLoader(train_subset_eval_ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0) if eval_test else None

    conditions = [
        ("A", "Group_A_Base (Drop:Off, Bal:Off)", False, "none"),
        ("B", "Group_B_DropoutOnly (Drop:On, Bal:Off)", True, "none"),
        ("C", "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)", False, "interleaved"),
        ("D", "Group_D_Interaction (Drop:On, Bal:Interleaved)", True, "interleaved"),
    ]

    all_summaries: List[Dict[str, Any]] = []
    all_curves: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n--- Running Paired Seed {seed} ---")
        for g_code, m_name, use_drop, bal_sched in conditions:
            summary, curves, _ = train_and_evaluate_condition(
                cfg=cfg,
                seed=seed,
                group_code=g_code,
                method_name=m_name,
                use_dropout=use_drop,
                balance_schedule=bal_sched,
                train_ds=train_ds,
                train_subset_eval_loader=train_subset_eval_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )
            all_summaries.append(summary)
            all_curves.extend(curves)
            test_str = (
                f"{summary['final_test_accuracy']*100:6.2f}%"
                if eval_test
                else "WITHHELD"
            )
            print(
                f"  [{g_code}] {m_name:47s} | "
                f"TrainSub: {summary['final_train_subset_eval_accuracy']*100:6.2f}% | "
                f"Val: {summary['final_val_accuracy']*100:6.2f}% | "
                f"Test: {test_str:>8s} | "
                f"Gap: {summary['generalization_gap']*100:+5.2f} pp | "
                f"Deficit({cfg.cost_domain[:5]}): {summary['norm_deficit_eligible_mean']:.2e}"
            )

    res_path = cfg.output_path("stage2_results.csv")
    curves_path = cfg.output_path("stage2_learning_curves.csv")
    inter_path = cfg.output_path("stage2_interaction_summary.csv")

    res_df = pd.DataFrame(all_summaries)
    curves_df = pd.DataFrame(all_curves)
    res_df.to_csv(res_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    primary_endpoint = "test_accuracy" if eval_test else "final_val_accuracy"
    metrics_to_analyze = [
        "final_val_accuracy",
        "early_val_accuracy",
        "final_train_subset_eval_accuracy",
        "final_val_loss",
        "generalization_gap",
        "parameter_squared_cost",
        "connection_graph_squared_cost",
        "norm_deficit_eligible_mean",
    ]
    if eval_test:
        metrics_to_analyze = ["test_accuracy", "final_test_loss"] + metrics_to_analyze

    inter_df = compute_factorial_interaction_report(
        res_df, metrics=metrics_to_analyze, primary_metric=primary_endpoint
    )
    inter_df.to_csv(inter_path, index=False)

    print("\n" + "=" * 86)
    print("STAGE 2 SUMMARY: CIFAR-10 2x2 FACTORIAL INTERACTION REPORT")
    print("=" * 86)
    p_row = inter_df[inter_df["metric"] == primary_endpoint].iloc[0]
    print(
        f"Primary Reported Endpoint ({primary_endpoint}, n={int(p_row['n_seeds'])} seeds, "
        f"cost_domain={cfg.cost_domain}, balance_mode={cfg.balance_mode}):\n"
        f"  Group A (Base):               {p_row['Group_A_mean']*100:6.2f}%\n"
        f"  Group B (Dropout Only):       {p_row['Group_B_mean']*100:6.2f}%\n"
        f"  Group C (Balance Only):       {p_row['Group_C_mean']*100:6.2f}%\n"
        f"  Group D (Dropout + Balance):  {p_row['Group_D_mean']*100:6.2f}%\n"
        f"  Dropout Effect w/o Balance (B - A): {p_row['dropout_without_balance_B_minus_A']*100:+6.2f} pp\n"
        f"  Dropout Effect w/  Balance (D - C): {p_row['dropout_with_balance_D_minus_C']*100:+6.2f} pp\n"
        f"  Balance Effect w/  Dropout (D - B): {p_row['balance_with_dropout_D_minus_B']*100:+6.2f} pp "
        f"(p={p_row['D_minus_B_p_unadjusted']:.4g})\n"
        f"  Interaction DiD (D-C)-(B-A):        {p_row['interaction_DiD_mean']*100:+6.2f} pp "
        f"(95% CI [{p_row['interaction_DiD_ci95_low']*100:+.2f}, "
        f"{p_row['interaction_DiD_ci95_high']*100:+.2f}] pp, "
        f"p={p_row['interaction_DiD_p_unadjusted']:.4g})\n"
        f"  Saved to:\n"
        f"    - {s1_report_path}\n"
        f"    - {res_path}\n"
        f"    - {curves_path}\n"
        f"    - {inter_path}\n"
        f"    - {manifest_path}"
    )


def run_stage3_controls_and_dropout_sweep(cfg: CNNExperimentConfig) -> None:
    cfg.stage = 3
    configure_reproducibility(cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    s1_report_path = cfg.output_path("stage1_verification_report.json")
    run_stage1_verification_tests(cfg, save_report_path=s1_report_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = save_provenance_manifest(cfg, device)
    seeds = cfg.active_seeds()
    eval_test = cfg.should_evaluate_test_set(for_stage=3)

    print(
        f"\nSTAGE 3: RANDOM-RESCALING MECHANISTIC CONTROLS & 1D VALIDATION DROPOUT SWEEPS\n"
        f"  run_tag          = {cfg.run_tag()}\n"
        f"  config_hash      = {cfg.config_fingerprint()}\n"
        f"  output_dir       = {cfg.run_output_dir()}\n"
        f"  seeds            = {seeds} (epochs={cfg.active_epochs()})\n"
        f"  evaluate_test    = {eval_test} (strictly requires --stage3-confirmatory-test-eval)\n"
        f"  device           = {device}"
    )

    train_ds, train_subset_eval_ds, val_ds, test_ds = prepare_cifar10_splits(cfg)
    train_subset_eval_loader = DataLoader(train_subset_eval_ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0) if eval_test else None

    control_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n--- Stage 3 Controls & Orthogonal 1D Validation Sweeps (Seed {seed}) ---")
        sum_c, _, ref_logs_c = train_and_evaluate_condition(
            cfg, seed, "C_ref", "Group_C_BalanceOnly_Ref", False, "interleaved",
            train_ds, train_subset_eval_loader, val_loader, test_loader, device
        )
        sum_c_rand, _, _ = train_and_evaluate_condition(
            cfg, seed, "C_rand", "Control_NoDrop_RandomRescale", False, "random_rescaled",
            train_ds, train_subset_eval_loader, val_loader, test_loader, device,
            reference_epoch_log_lambdas=ref_logs_c
        )

        sum_d, _, ref_logs_d = train_and_evaluate_condition(
            cfg, seed, "D_ref", "Group_D_Interaction_Ref", True, "interleaved",
            train_ds, train_subset_eval_loader, val_loader, test_loader, device
        )
        sum_d_rand, _, _ = train_and_evaluate_condition(
            cfg, seed, "D_rand", "Control_Dropout_RandomRescale", True, "random_rescaled",
            train_ds, train_subset_eval_loader, val_loader, test_loader, device,
            reference_epoch_log_lambdas=ref_logs_d
        )
        control_rows.extend([sum_c, sum_c_rand, sum_d, sum_d_rand])

        for c_p in cfg.conv_dropout_sweep_1d:
            tag = f"1DConvSweep_pConv{c_p:.2f}_pFC{cfg.fc_dropout_rate:.2f}"
            use_d = (c_p > 0.0) or (cfg.fc_dropout_rate > 0.0)
            sum_b_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"B_{tag}", f"Sweep_DropOnly_{tag}", use_d, "none",
                train_ds, train_subset_eval_loader, val_loader, None, device,
                conv_dropout_override=c_p, fc_dropout_override=cfg.fc_dropout_rate
            )
            sum_d_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"D_{tag}", f"Sweep_DropPlusBal_{tag}", use_d, "interleaved",
                train_ds, train_subset_eval_loader, val_loader, None, device,
                conv_dropout_override=c_p, fc_dropout_override=cfg.fc_dropout_rate
            )
            sweep_rows.extend([sum_b_sw, sum_d_sw])

        for f_p in cfg.fc_dropout_sweep_1d:
            tag = f"1DFCSweep_pConv{cfg.conv_dropout_rate:.2f}_pFC{f_p:.2f}"
            use_d = (cfg.conv_dropout_rate > 0.0) or (f_p > 0.0)
            sum_b_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"B_{tag}", f"Sweep_DropOnly_{tag}", use_d, "none",
                train_ds, train_subset_eval_loader, val_loader, None, device,
                conv_dropout_override=cfg.conv_dropout_rate, fc_dropout_override=f_p
            )
            sum_d_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"D_{tag}", f"Sweep_DropPlusBal_{tag}", use_d, "interleaved",
                train_ds, train_subset_eval_loader, val_loader, None, device,
                conv_dropout_override=cfg.conv_dropout_rate, fc_dropout_override=f_p
            )
            sweep_rows.extend([sum_b_sw, sum_d_sw])

    controls_path = cfg.output_path("stage3_random_rescale_controls.csv")
    contrasts_path = cfg.output_path("stage3_random_rescale_interaction_summary.csv")
    sweep_path = cfg.output_path("stage3_dropout_validation_sweep.csv")

    controls_df = pd.DataFrame(control_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    controls_df.to_csv(controls_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)

    primary_s3_metric = "test_accuracy" if eval_test else "final_val_accuracy"
    s3_metrics = [
        "final_val_accuracy",
        "early_val_accuracy",
        "final_val_loss",
        "generalization_gap",
        "parameter_squared_cost",
        "connection_graph_squared_cost",
        "norm_deficit_eligible_mean",
    ]
    if eval_test:
        s3_metrics = ["test_accuracy", "final_test_loss"] + s3_metrics

    contrasts_df = compute_stage3_random_rescale_contrasts(
        controls_df, metrics=s3_metrics, primary_metric=primary_s3_metric
    )
    contrasts_df.to_csv(contrasts_path, index=False)

    print("\n" + "=" * 86)
    print("STAGE 3 SUMMARY: RANDOM-RESCALING MECHANISTIC CONTRASTS")
    print("=" * 86)
    if not contrasts_df.empty:
        c_row = contrasts_df[contrasts_df["metric"] == primary_s3_metric].iloc[0]
        print(
            f"Endpoint: {primary_s3_metric} (n={int(c_row['n_seeds'])} seeds)\n"
            f"  delta_C (C_ref - C_rand, No Dropout):   {c_row['delta_C_mean']*100:+6.2f} pp "
            f"(95% CI [{c_row['delta_C_ci95_low']*100:+.2f}, {c_row['delta_C_ci95_high']*100:+.2f}] pp, "
            f"p_unadj={c_row['delta_C_p_unadjusted']:.4g})\n"
            f"  delta_D (D_ref - D_rand, With Dropout): {c_row['delta_D_mean']*100:+6.2f} pp "
            f"(95% CI [{c_row['delta_D_ci95_low']*100:+.2f}, {c_row['delta_D_ci95_high']*100:+.2f}] pp, "
            f"p_unadj={c_row['delta_D_p_unadjusted']:.4g})\n"
            f"  delta_interaction (delta_D - delta_C):  {c_row['delta_interaction_mean']*100:+6.2f} pp "
            f"(95% CI [{c_row['delta_interaction_ci95_low']*100:+.2f}, "
            f"{c_row['delta_interaction_ci95_high']*100:+.2f}] pp, "
            f"p_unadj={c_row['delta_interaction_p_unadjusted']:.4g})"
        )
    print(
        f"Stage 3 outputs written to:\n"
        f"  - {s1_report_path}\n"
        f"  - {controls_path}\n"
        f"  - {contrasts_path}\n"
        f"  - {sweep_path}\n"
        f"  - {manifest_path}"
    )


# =============================================================================
# 8. CLI Entrypoint
# =============================================================================

def parse_args() -> CNNExperimentConfig:
    parser = argparse.ArgumentParser(description="CNN Synaptic Balancing Experiments (Stages 1-3, v1.2.3)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="1: Verification gate only; 2: CIFAR-10 2x2 factorial; 3: Controls & 1D sweeps")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "pilot", "full"],
                        help="smoke (seed 42, 5 ep), pilot (seeds 43-47, 60 ep), full (seeds 48-55, 60 ep)")
    parser.add_argument("--cost-domain", type=str, default="parameter",
                        choices=["parameter", "connection_graph"],
                        help="'parameter' (unique trainable L2) or 'connection_graph' (Sec 8.2 Eq. 8.1)")
    parser.add_argument("--balance-mode", type=str, default="layer", choices=["layer", "channel"],
                        help="'layer' (Sec 8.3 whole-layer) or 'channel' (Sec 8.2 per-filter tied subset)")
    parser.add_argument("--fc-balance-mode", type=str, default="layer", choices=["layer", "neuron"],
                        help="Balancing granularity for downstream FC hidden layers")
    parser.add_argument("--conv-dropout-mode", type=str, default="elementwise",
                        choices=["elementwise", "channel"],
                        help="'elementwise' (B,C,H,W) or 'channel' (Dropout2d B,C,1,1)")
    parser.add_argument("--conv-dropout-rate", type=float, default=0.20,
                        help="Baseline convolutional dropout rate (default: 0.20)")
    parser.add_argument("--fc-dropout-rate", type=float, default=0.50,
                        help="Baseline fully connected hidden dropout rate (default: 0.50)")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Stateless SGD learning rate")
    parser.add_argument("--output-root", type=str, default="./results_cnn_cifar10",
                        help="Root folder for configuration-isolated output directories")
    parser.add_argument("--allow-nonconfirmatory-test-eval", action="store_true",
                        help="Override test-set withholding during Stage 2 smoke/pilot runs (default: False)")
    parser.add_argument("--stage3-confirmatory-test-eval", action="store_true",
                        help="Explicitly unlock official test-set evaluation for confirmatory Stage 3 C/D controls")
    parser.add_argument("--require-clean-git", action="store_true",
                        help="Abort if the git working tree has uncommitted changes")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save final model checkpoints")
    parser.add_argument("--deterministic-warn-only", action="store_true",
                        help="Use warn_only=True instead of strict error mode for deterministic algorithms")
    parser.add_argument("--no-deterministic", action="store_true", help="Disable deterministic algorithms")
    args = parser.parse_args()

    cfg = CNNExperimentConfig(
        stage=args.stage,
        run_mode=args.mode,
        cost_domain=args.cost_domain,
        balance_mode=args.balance_mode,
        fc_balance_mode=args.fc_balance_mode,
        conv_dropout_mode=args.conv_dropout_mode,
        conv_dropout_rate=args.conv_dropout_rate,
        fc_dropout_rate=args.fc_dropout_rate,
        learning_rate=args.lr,
        output_root=args.output_root,
        allow_nonconfirmatory_test_eval=args.allow_nonconfirmatory_test_eval,
        stage3_confirmatory_test_eval=args.stage3_confirmatory_test_eval,
        require_clean_git=args.require_clean_git,
        deterministic=not args.no_deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
        save_checkpoints=args.save_checkpoints,
    )
    if args.epochs is not None:
        cfg.epochs_smoke = args.epochs
        cfg.epochs_main = args.epochs
    return cfg


if __name__ == "__main__":
    config = parse_args()
    if config.stage == 1:
        configure_reproducibility(config.deterministic, warn_only=config.deterministic_warn_only)
        run_stage1_verification_tests(config)
    elif config.stage == 2:
        run_stage2_cifar10_factorial(config)
    elif config.stage == 3:
        run_stage3_controls_and_dropout_sweep(config)