#!/usr/bin/env python3
"""
cnn_cifar10_experiment.py (v1.1.0)
==================================
Staged implementation of Synaptic Neural Balance in Convolutional Neural Networks
following HandOff1.md and Baldi et al. (2025), Sections 8.2, 8.3, and Appendix B.

Key Theoretical & Methodological Features in v1.1.0:
  1. Explicit Separation of Cost Domains (`--cost-domain {parameter,connection_graph}`):
     - "parameter" (Primary Baseline): Minimizes the unique-parameter L2 regularizer
       R_param(W) = sum_l (||W^(l)||_F^2 + ||b^(l)||_2^2).
     - "connection_graph" (Baldi et al. Sec 8.2 Eq. 8.1 Multiplicity-Weighted Cost):
       Weights each shared convolutional kernel parameter by its exact spatial connection
       multiplicity M_l^in(u, v) across the valid non-padded receptive fields and accounts
       for spatial resolution changes across MaxPool2d and Flatten -> Linear transitions.
     Both parameter-domain and graph-domain balance deficits are logged simultaneously.
  2. Pooling Commutativity Invariant:
     - Positive homogeneity through MaxPool2d/AvgPool2d holds because all spatial positions
       within each pooling window receive the identical positive multiplier (shared across
       the whole layer in `balance_mode="layer"`, or within each channel in
       `balance_mode="channel"`).
  3. Clean Generalization Gap Evaluation:
     - Evaluates a fixed, unaugmented 5,000-example training subset (`train_eval_loader`)
       in `model.eval()` mode (dropout disabled) alongside `val_loader` at every epoch.
  4. Strict Disjoint Seed Tiers & Device-Native Isolated RNG Streams:
     - smoke_seeds = [42]
     - pilot_seeds = [43, 44, 45, 46, 47]
     - full_seeds  = [48, 49, 50, 51, 52, 53, 54, 55]
     - Device-native dropout generator eliminates CPU->GPU mask transfer overhead while
       maintaining bitwise-identical dropout masks between Groups B and D.
  5. Complete Stage 3 Mechanistic Controls & Orthogonal 1D Validation Sweeps:
     - Compares both Group C vs. C_rand (Dropout Off) and Group D vs. D_rand (Dropout On).
     - Runs orthogonal 1D validation sweeps over conv_dropout_rate and fc_dropout_rate
       using validation data only (no test-set leakage during hyperparameter selection).
"""

import argparse
import copy
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
# 1. Configuration, Determinism & Provenance Utilities
# =============================================================================

def get_git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def configure_reproducibility(deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


@dataclass
class CNNExperimentConfig:
    version: str = "1.1.0"
    stage: int = 1
    run_mode: str = "smoke"  # "smoke" (1 seed, 5 epochs), "pilot" (5 seeds), "full" (8 disjoint seeds)

    # Balancing definitions
    cost_domain: str = "parameter"        # "parameter" (primary) or "connection_graph" (Sec 8.2 Eq. 8.1)
    balance_mode: str = "layer"           # "layer" (Sec 8.3 primary) or "channel" (secondary tied-filter)
    fc_balance_mode: str = "layer"        # "layer" (primary) or "neuron" (independent FC neuron balance)

    # Disjoint seed tiers (no overlap between smoke, pilot, and confirmatory full runs)
    smoke_seeds: List[int] = field(default_factory=lambda: [42])
    pilot_seeds: List[int] = field(default_factory=lambda: [43, 44, 45, 46, 47])
    full_seeds: List[int] = field(default_factory=lambda: [48, 49, 50, 51, 52, 53, 54, 55])

    # Dataset & Split (CIFAR-10: 45,000 train / 5,000 val / 10,000 official test + 5,000 unaugmented train_eval)
    data_dir: str = "./data"
    split_seed: int = 20261117
    train_size: int = 45000
    val_size: int = 5000
    train_eval_size: int = 5000
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
    save_checkpoints: bool = False
    checkpoint_dir: str = "./checkpoints_cnn_cifar10"

    # Numerical & Convergence parameters
    balance_eps: float = 1e-12
    full_balance_max_iters: int = 300
    full_balance_tol_f64: float = 1e-10
    full_balance_tol_f32: float = 1e-5
    logit_tol_f64: float = 1e-12
    logit_tol_f32: float = 1e-5

    # Stage 3 Orthogonal 1D Validation Sweeps (holding the other dropout rate at its baseline value)
    conv_dropout_sweep_1d: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40)
    fc_dropout_sweep_1d: Tuple[float, ...] = (0.00, 0.25, 0.50, 0.65, 0.80)

    # Output files
    results_csv: str = "stage2_cnn_cifar10_results.csv"
    learning_curves_csv: str = "stage2_cnn_cifar10_learning_curves.csv"
    interaction_csv: str = "stage2_cnn_cifar10_interaction_summary.csv"
    provenance_json: str = "stage2_cnn_cifar10_provenance.json"
    stage3_controls_csv: str = "stage3_cnn_cifar10_random_rescale_controls.csv"
    stage3_sweep_csv: str = "stage3_cnn_cifar10_dropout_validation_sweep.csv"

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


@dataclass
class IsolatedRNGStreams:
    """
    Dedicated random number generators for paired-seed experiments.
    - `gen_init`, `gen_loader`, `gen_aug`, and `gen_rescale` live on CPU.
    - `gen_dropout` lives directly on the execution `device` (e.g., CUDA when available),
      avoiding CPU-to-GPU mask transfers while guaranteeing identical dropout masks
      between paired Dropout-On conditions (Groups B and D).
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

    Architecture & Spatial Geometry (for 32x32 input):
      conv1: Conv2d(3, C1, 3, pad=1)   -> ReLU                 [H1xW1 = 32x32, post-op 32x32]
      conv2: Conv2d(C1, C2, 3, pad=1)  -> ReLU -> MaxPool2d(2) [H2xW2 = 32x32, post-op 16x16] -> drop_conv1
      conv3: Conv2d(C2, C3, 3, pad=1)  -> ReLU                 [H3xW3 = 16x16, post-op 16x16]
      conv4: Conv2d(C3, C4, 3, pad=1)  -> ReLU -> MaxPool2d(2) [H4xW4 = 16x16, post-op  8x8 ] -> drop_conv2
      flatten -> fc1: Linear(C4*8*8, H_fc) -> ReLU             [256 units]                    -> drop_fc1
      fc2: Linear(H_fc, num_classes)                           [10 output logits]
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
        # Exact (H_in, W_in, H_out, W_out) for each Conv2d layer prior to any subsequent pooling
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

    For a 3x3 kernel with stride=1, padding=1 on HxW:
      - Center (1,1): H * W valid connections
      - Edges:        H * (W - 1) or (H - 1) * W valid connections
      - Corners:      (H - 1) * (W - 1) valid connections
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
    Computes the total L2 regularizer R(W) in float64 precision under either:
      - cost_domain == "parameter": sum of squared unique trainable parameters.
      - cost_domain == "connection_graph": sum of squared weights over all instantiated
        connections in the computational graph (weighted by spatial reuse multiplicity).
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
    Returns (A, B) 1D float64 tensors of incoming and outgoing L2 costs for hidden
    layer `layer_idx` (0 <= layer_idx < len(model.param_layers) - 1) under:
      - mode: "layer" (Coordinated Layer, Sec 8.3) or "channel"/"neuron" (Granular)
      - cost_domain: "parameter" (Unique-Parameter Frobenius L2) or
                     "connection_graph" (Spatial-Multiplicity-Weighted Graph L2, Sec 8.2 Eq. 8.1)
    """
    curr_layer = model.param_layers[layer_idx]
    next_layer = model.param_layers[layer_idx + 1]
    final_spatial = model.final_spatial

    w_in = curr_layer.weight.detach().double()
    b_in = curr_layer.bias.detach().double() if curr_layer.bias is not None else None
    w_out = next_layer.weight.detach().double()

    # Determine multiplicity weights for incoming parameters of curr_layer
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

    # Determine multiplicity weights for outgoing parameters (in next_layer)
    if cost_domain == "connection_graph" and isinstance(next_layer, nn.Conv2d):
        h_in2, w_in2, h_out2, w_out2 = model.conv_spatial_shapes[layer_idx + 1]
        mult_out_uv, _ = compute_conv2d_spatial_multiplicity(
            next_layer, h_in2, w_in2, h_out2, w_out2, device=w_out.device
        )
        w_out_sq = w_out.pow(2) * mult_out_uv.view(1, 1, *mult_out_uv.shape)
    else:
        # Linear layers have connection multiplicity 1 in both parameter and graph domains
        w_out_sq = w_out.pow(2)

    if mode == "layer":
        a_val = w_in_sq.sum()
        if b_in_sq is not None:
            a_val = a_val + b_in_sq.sum()
        b_val = w_out_sq.sum()
        return a_val.unsqueeze(0), b_val.unsqueeze(0)

    # Granular mode: per-channel for Conv2d, per-neuron for Linear
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
    Applies the function-preserving scaling transformation:
      incoming weights and bias of curr_layer *= lambda
      outgoing weights (in next_layer)        /= lambda
    Because `lambdas > 0` is spatially uniform across every (h, w) coordinate within
    each channel (and across all channels in `layer` mode), this transformation
    commutes strictly with ReLU and channel-wise MaxPool2d/AvgPool2d.
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
    both `parameter` and `connection_graph` deficits for full transparency.
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
    Executes one forward pass of partial balancing across all hidden layers under
    the specified `cost_domain` ("parameter" or "connection_graph").
    Returns the list of applied log-scaling tensors [log(lambda_l)].
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
    """
    Iteratively sweeps partial_balance_cnn_once until the maximum normalized balance
    deficit across all eligible units in `cost_domain` is <= tol.
    Explicitly records `"converged": bool(...)` so callers can assert convergence.
    """
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
    """
    Stage 3 Mechanistic Control: Applies a function-preserving rescaling whose
    per-layer log-magnitudes match the empirical balancing step from the paired
    balanced model (`reference_log_lambdas`), with random sign inversions and
    (when multiple units exist in a layer) random permutations.
    """
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
# 4. Statistical Analysis Pipeline (Full Precision DiD & Main Effects)
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
    primary_metric: str = "test_accuracy",
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        piv = results_df.pivot(index="seed", columns="group_code", values=metric)
        if not {"A", "B", "C", "D"}.issubset(set(piv.columns)):
            continue
        piv = piv.dropna(subset=["A", "B", "C", "D"])
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


# =============================================================================
# 5. Paired-Seed CIFAR-10 Dataset & Deterministic Augmentation Wrapper
# =============================================================================

class DeterministicAugmentedCIFARDataset(Dataset):
    """
    Applies CIFAR-10 normalization and provides a deterministic, paired batch
    augmentation method (`apply_paired_batch_augmentation`) driven by `gen_aug`.
    """
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
        """
        Applies batch-level RandomCrop(H, padding=4) and RandomHorizontalFlip(p=0.5)
        using `gen_aug`, keeping RNG consumption identical across all conditions for
        a given seed and step.
        """
        b, c, h, w = x.shape
        padded = F.pad(x, (padding, padding, padding, padding), mode=pad_mode)
        max_offset = 2 * padding + 1
        top_offsets = torch.randint(0, max_offset, (b,), generator=gen_aug)
        left_offsets = torch.randint(0, max_offset, (b,), generator=gen_aug)
        flip_mask = torch.rand((b,), generator=gen_aug) < 0.5

        out = torch.empty_like(x)
        for i in range(b):
            t_i = int(top_offsets[i].item())
            l_i = int(left_offsets[i].item())
            patch = padded[i, :, t_i : t_i + h, l_i : l_i + w]
            if bool(flip_mask[i].item()):
                patch = torch.flip(patch, dims=[2])
            out[i] = patch
        return out


def prepare_cifar10_splits(cfg: CNNExperimentConfig):
    """
    Loads CIFAR-10 and creates:
      - `train_ds`: 45,000 training examples (used with paired augmentation during training)
      - `train_eval_ds`: fixed 5,000-example unaugmented training subset for clean
        apples-to-apples `generalization_gap = eval_train_accuracy - val_accuracy`
      - `val_ds`: 5,000 validation examples
      - `test_ds`: official 10,000 test examples
    """
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
    train_eval_idx = train_idx[: cfg.train_eval_size]

    train_ds = DeterministicAugmentedCIFARDataset(full_train, train_idx)
    train_eval_ds = DeterministicAugmentedCIFARDataset(full_train, train_eval_idx)
    val_ds = DeterministicAugmentedCIFARDataset(full_train, val_idx)
    test_ds = DeterministicAugmentedCIFARDataset(
        test_set_raw, list(range(len(test_set_raw)))
    )
    return train_ds, train_eval_ds, val_ds, test_ds


# =============================================================================
# 6. Stage 1 Verification Gate: Mathematical & Numerical Unit Tests
# =============================================================================

def run_stage1_verification_tests(cfg: Optional[CNNExperimentConfig] = None) -> None:
    """
    Executes the 5 mandatory Stage 1 verification unit tests before any CIFAR-10
    training is permitted.
    """
    if cfg is None:
        cfg = CNNExperimentConfig()

    print("=" * 82)
    print("STAGE 1 GATE: MATHEMATICAL FEASIBILITY & NUMERICAL INVARIANCE UNIT TESTS (v1.1.0)")
    print("=" * 82)

    mode_combinations = [
        ("parameter", "layer", "layer"),
        ("parameter", "channel", "layer"),
        ("parameter", "channel", "neuron"),
        ("connection_graph", "layer", "layer"),
        ("connection_graph", "channel", "layer"),
        ("connection_graph", "channel", "neuron"),
    ]

    # -------------------------------------------------------------------------
    # Test 1: Exact Function Preservation (Logit Invariance) in float64 & float32
    #         across all (cost_domain, balance_mode, fc_balance_mode) combinations,
    #         plus active-dropout fixed-mask invariance (elementwise & channel).
    # -------------------------------------------------------------------------
    for cost_dom, conv_m, fc_m in mode_combinations:
        # 1a. Float64 eval() logit invariance
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

        # 1b. Float32 eval() logit invariance on 32x32 input
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
        print(
            f"[PASS] Test 1a ({cost_dom:16s} | conv={conv_m:7s} | fc={fc_m:6s}): "
            f"f64 diff={diff_full_f64:.2e}, f32 diff={diff_f32:.2e} "
            f"(iters={conv_diag64['iterations']})"
        )

    # 1c. Active-dropout fixed-mask logit invariance (both elementwise and channel dropout)
    for drop_mode in ["elementwise", "channel"]:
        rng_d1 = IsolatedRNGStreams.from_seed(777)
        rng_d2 = IsolatedRNGStreams.from_seed(777)
        model_drop = SequentialCIFARCNN(
            in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
            fc_hidden=24, num_classes=10, use_dropout=True, conv_dropout_mode=drop_mode
        ).double()
        model_drop.reset_parameters_with_generator(rng_d1.gen_init)
        model_drop.train()  # Enable active dropout masks
        x_d = torch.randn(4, 3, 16, 16, generator=rng_d1.gen_loader, dtype=torch.float64)
        with torch.no_grad():
            out_drop_before = model_drop(x_d, dropout_rng=rng_d1.gen_dropout).clone()
            partial_balance_cnn_once(model_drop, conv_mode="channel", fc_mode="neuron", cost_domain="parameter")
            out_drop_after = model_drop(x_d, dropout_rng=rng_d2.gen_dropout).clone()
        diff_drop = float((out_drop_before - out_drop_after).abs().max().item())
        assert diff_drop < cfg.logit_tol_f64, f"Fixed-mask dropout invariance failed ({drop_mode}): {diff_drop}"
    print("[PASS] Test 1b (Active Dropout Fixed-Mask Invariance): Verified for elementwise & channel modes")

    # -------------------------------------------------------------------------
    # Test 2: Monotonic Cost Reduction in Chosen Regularizer Domain &
    #         Explicit Demonstration of Parameter vs. Graph Cost Multipliers
    # -------------------------------------------------------------------------
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
                assert r_curr <= r_prev + 1e-11, (
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
            print(
                f"[PASS] Test 2 ({cost_dom:16s} | {balance_mode:7s}): Monotonic R(W) reduction "
                f"({r_initial:.2f} -> {r_final:.2f}) & converged to {diag_final['norm_deficit_eligible_max']:.2e}"
            )

    # -------------------------------------------------------------------------
    # Test 3: Dead / Zero-Cost & Near-Zero-Cost Unit Exclusion Diagnostics
    # -------------------------------------------------------------------------
    rng_streams = IsolatedRNGStreams.from_seed(404)
    model_zero = SequentialCIFARCNN(
        in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
        fc_hidden=24, num_classes=10, use_dropout=False
    ).double()
    model_zero.reset_parameters_with_generator(rng_streams.gen_init)
    with torch.no_grad():
        # Zero incoming filter 0 in conv1; zero outgoing slice for filter 1 in conv2; near-zero filter 2
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
    print(
        f"[PASS] Test 3 (Eligibility Diagnostics): Verified {diag_z['n_zero_cost_excluded']} zero-cost "
        f"& {diag_z['n_near_zero_excluded']} near-zero exclusions out of {diag_z['n_total_units']} units"
    )

    # -------------------------------------------------------------------------
    # Test 4: Isolated RNG Streams (Init, Shuffling, Paired Augmentation & Dropout)
    # -------------------------------------------------------------------------
    streams_a = IsolatedRNGStreams.from_seed(505)
    streams_b = IsolatedRNGStreams.from_seed(505)
    net_a = SequentialCIFARCNN(spatial_size=16, conv_channels=(8, 8, 16, 16), fc_hidden=16, use_dropout=False)
    net_b = SequentialCIFARCNN(spatial_size=16, conv_channels=(8, 8, 16, 16), fc_hidden=16, use_dropout=True)
    net_a.reset_parameters_with_generator(streams_a.gen_init)
    net_b.reset_parameters_with_generator(streams_b.gen_init)

    for (na, pa), (nb, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
        assert torch.equal(pa, pb), f"Parameter mismatch at init on {na}"

    synth_x = torch.randn(32, 3, 16, 16)
    synth_y = torch.randint(0, 10, (32,))
    ds = torch.utils.data.TensorDataset(synth_x, synth_y)
    loader_a = DataLoader(ds, batch_size=8, shuffle=True, generator=streams_a.gen_loader)
    loader_b = DataLoader(ds, batch_size=8, shuffle=True, generator=streams_b.gen_loader)

    net_a.train()
    net_b.train()
    for epoch in range(2):
        for (xa, ya), (xb, yb) in zip(loader_a, loader_b):
            assert torch.equal(xa, xb) and torch.equal(ya, yb), f"Batch order diverged at epoch {epoch}"
            aug_a = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(xa, streams_a.gen_aug)
            # Consume dropout and rescale RNG on branch B before/during augmentation
            _ = torch.rand(10, generator=streams_b.gen_rescale)
            aug_b = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(xb, streams_b.gen_aug)
            assert torch.equal(aug_a, aug_b), f"Paired batch augmentation diverged at epoch {epoch}"
            _ = net_a(aug_a, dropout_rng=None)
            _ = net_b(aug_b, dropout_rng=streams_b.gen_dropout)

    # Also verify seed sets are strictly disjoint
    assert set(cfg.smoke_seeds).isdisjoint(set(cfg.pilot_seeds)), "Smoke and pilot seeds overlap!"
    assert set(cfg.pilot_seeds).isdisjoint(set(cfg.full_seeds)), "Pilot and full confirmatory seeds overlap!"
    assert set(cfg.smoke_seeds).isdisjoint(set(cfg.full_seeds)), "Smoke and full seeds overlap!"
    print(
        "[PASS] Test 4 (RNG & Seed Isolation): Identical init, minibatch order, and paired "
        "batch augmentation verified; smoke/pilot/full seed sets strictly disjoint"
    )

    # -------------------------------------------------------------------------
    # Test 5: Full-Precision Factorial Interaction (DiD) & Main-Effect Pipeline
    # -------------------------------------------------------------------------
    synth_rows = []
    for s_idx, seed in enumerate([43, 44, 45, 46, 47]):
        jitter = (s_idx - 2) * 0.0004
        synth_rows.extend([
            {"seed": seed, "group_code": "A", "test_accuracy": 0.720000 + jitter, "val_loss": 0.850},
            {"seed": seed, "group_code": "B", "test_accuracy": 0.710000 + jitter * 1.1, "val_loss": 0.870},
            {"seed": seed, "group_code": "C", "test_accuracy": 0.735000 + jitter * 0.9, "val_loss": 0.810},
            {"seed": seed, "group_code": "D", "test_accuracy": 0.742000 + jitter * 1.2, "val_loss": 0.790},
        ])
    rep = compute_factorial_interaction_report(
        pd.DataFrame(synth_rows), metrics=["test_accuracy", "val_loss"], primary_metric="test_accuracy"
    )
    row_acc = rep[rep["metric"] == "test_accuracy"].iloc[0]
    expected_did = (0.742 - 0.735) - (0.710 - 0.720)
    assert abs(row_acc["interaction_DiD_mean"] - expected_did) < 1e-12
    print(
        f"[PASS] Test 5 (Factorial DiD Pipeline): Verified full-precision DiD = "
        f"{row_acc['interaction_DiD_mean']*100:+.3f} pp, 95% CI "
        f"[{row_acc['interaction_DiD_ci95_low']*100:+.3f}, {row_acc['interaction_DiD_ci95_high']*100:+.3f}] pp"
    )
    print("=" * 82)
    print("ALL STAGE 1 MATHEMATICAL & NUMERICAL VERIFICATION GATES PASSED.")
    print("=" * 82)


# =============================================================================
# 7. Stage 2 & Stage 3 Training and Evaluation Engine
# =============================================================================

@torch.no_grad()
def evaluate_cnn(model: SequentialCIFARCNN, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluates loss and accuracy in `model.eval()` mode (dropout off, no augmentation)."""
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
    balance_schedule: str,  # "none", "interleaved", "random_rescaled"
    train_ds: DeterministicAugmentedCIFARDataset,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    device: torch.device,
    conv_dropout_override: Optional[float] = None,
    fc_dropout_override: Optional[float] = None,
    reference_epoch_log_lambdas: Optional[List[List[torch.Tensor]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[List[torch.Tensor]]]:
    """
    Trains one CNN condition for `cfg.active_epochs()` with isolated RNG streams
    and stateless SGD. Evaluates clean unaugmented `train_eval_loader` in `model.eval()`
    at every epoch to compute a true apples-to-apples `generalization_gap`.
    """
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

        # Clean apples-to-apples evaluation in model.eval() (no dropout, no augmentation)
        eval_train_loss, eval_train_acc = evaluate_cnn(model, train_eval_loader, device)
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
            "run_mode": cfg.run_mode,
            "seed": int(seed),
            "group_code": group_code,
            "method": method_name,
            "cost_domain": cfg.cost_domain,
            "balance_mode": cfg.balance_mode,
            "fc_balance_mode": cfg.fc_balance_mode,
            "conv_dropout_mode": cfg.conv_dropout_mode,
            "epoch": int(epoch),
            "train_mode_augmented_loss": float(train_mode_loss),
            "train_mode_augmented_accuracy": float(train_mode_acc),
            "eval_train_loss": float(eval_train_loss),
            "eval_train_accuracy": float(eval_train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "generalization_gap": float(eval_train_acc - val_acc),
            "train_mode_minus_val_accuracy": float(train_mode_acc - val_acc),
            "l2_regularizer_param_norm": float(r_param),
            "l2_regularizer_graph_norm": float(r_graph),
            "norm_deficit_eligible_mean": float(diag["norm_deficit_eligible_mean"]),
            "norm_deficit_eligible_max": float(diag["norm_deficit_eligible_max"]),
            "param_domain_norm_deficit_mean": float(diag["param_domain_norm_deficit_mean"]),
            "graph_domain_norm_deficit_mean": float(diag["graph_domain_norm_deficit_mean"]),
            "eligible_fraction": float(diag["eligible_fraction"]),
        })

    total_train_sec = time.perf_counter() - t_start
    final_eval_train_loss = curve_rows[-1]["eval_train_loss"]
    final_eval_train_acc = curve_rows[-1]["eval_train_accuracy"]
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
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            cfg.checkpoint_dir,
            f"ckpt_{cfg.run_mode}_seed{seed}_{group_code}_{cfg.cost_domain}_{cfg.balance_mode}.pt",
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "seed": seed,
                "group_code": group_code,
            },
            ckpt_path,
        )

    early_ep = min(10, max(1, epochs // 3))
    early_val_acc = next(r["val_accuracy"] for r in curve_rows if r["epoch"] == early_ep)

    # 5-epoch validation plateau diagnostic (checks if val_accuracy changed <= 0.50 pp over last 5 epochs)
    if len(curve_rows) >= 5:
        val_delta_last5 = abs(curve_rows[-1]["val_accuracy"] - curve_rows[-5]["val_accuracy"])
    else:
        val_delta_last5 = np.nan

    summary_row = {
        "script_version": cfg.version,
        "git_commit": get_git_commit_hash(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "deterministic_algorithms": bool(cfg.deterministic),
        "run_mode": cfg.run_mode,
        "split_seed": int(cfg.split_seed),
        "seed": int(seed),
        "group_code": group_code,
        "method": method_name,
        "model_architecture": f"CustomSequentialVGG_Conv{list(cfg.conv_channels)}_FC{cfg.fc_hidden}",
        "model_param_count": int(n_params),
        "cost_domain": cfg.cost_domain,
        "balance_mode": cfg.balance_mode,
        "fc_balance_mode": cfg.fc_balance_mode,
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
        "final_eval_train_accuracy": float(final_eval_train_acc),
        "final_eval_train_loss": float(final_eval_train_loss),
        "final_val_accuracy": float(final_val_acc),
        "final_val_loss": float(final_val_loss),
        "final_test_accuracy": float(final_test_acc),
        "test_accuracy": float(final_test_acc),
        "final_test_loss": float(final_test_loss),
        "generalization_gap": float(final_eval_train_acc - final_val_acc),
        "train_mode_minus_val_accuracy": float(curve_rows[-1]["train_mode_minus_val_accuracy"]),
        "val_abs_change_last_5_epochs": float(val_delta_last5),
        "l2_regularizer_param_norm": float(compute_total_l2_regularizer(model, "parameter")),
        "l2_regularizer_graph_norm": float(compute_total_l2_regularizer(model, "connection_graph")),
        "norm_deficit_eligible_mean": float(final_diag["norm_deficit_eligible_mean"]),
        "norm_deficit_eligible_max": float(final_diag["norm_deficit_eligible_max"]),
        "param_domain_norm_deficit_mean": float(final_diag["param_domain_norm_deficit_mean"]),
        "graph_domain_norm_deficit_mean": float(final_diag["graph_domain_norm_deficit_mean"]),
        "raw_l2_deficit_eligible": float(final_diag["raw_l2_deficit_eligible"]),
        "n_eligible_units": int(final_diag["n_eligible_units"]),
        "n_zero_cost_excluded": int(final_diag["n_zero_cost_excluded"]),
        "n_near_zero_excluded": int(final_diag["n_near_zero_excluded"]),
        "training_time_sec": float(total_train_sec),
        "balance_overhead_sec": float(balance_overhead_sec),
    }
    return summary_row, curve_rows, recorded_log_lambdas


def save_provenance_manifest(cfg: CNNExperimentConfig, device: torch.device) -> None:
    manifest = {
        "script_version": cfg.version,
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "config": asdict(cfg),
    }
    with open(cfg.provenance_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_stage2_cifar10_factorial(cfg: CNNExperimentConfig) -> None:
    """
    Executes the Stage 1 verification gate first, then runs the Stage 2 2x2
    factorial experiment on CIFAR-10 across paired seeds.
    """
    configure_reproducibility(cfg.deterministic)
    run_stage1_verification_tests(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_provenance_manifest(cfg, device)
    seeds = cfg.active_seeds()
    print(
        f"\nSTAGE 2: CIFAR-10 2x2 FACTORIAL EXPERIMENT "
        f"(mode={cfg.run_mode}, seeds={seeds}, epochs={cfg.active_epochs()}, "
        f"cost_domain={cfg.cost_domain}, balance_mode={cfg.balance_mode}, "
        f"conv_dropout={cfg.conv_dropout_mode}, device={device})"
    )

    train_ds, train_eval_ds, val_ds, test_ds = prepare_cifar10_splits(cfg)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

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
                train_eval_loader=train_eval_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )
            all_summaries.append(summary)
            all_curves.extend(curves)
            print(
                f"  [{g_code}] {m_name:47s} | "
                f"EvalTrain: {summary['final_eval_train_accuracy']*100:6.2f}% | "
                f"Val: {summary['final_val_accuracy']*100:6.2f}% | "
                f"Test: {summary['final_test_accuracy']*100:6.2f}% | "
                f"Gap: {summary['generalization_gap']*100:+5.2f} pp | "
                f"Deficit({cfg.cost_domain[:5]}): {summary['norm_deficit_eligible_mean']:.2e}"
            )

    res_df = pd.DataFrame(all_summaries)
    curves_df = pd.DataFrame(all_curves)
    res_df.to_csv(cfg.results_csv, index=False)
    curves_df.to_csv(cfg.learning_curves_csv, index=False)

    metrics_to_analyze = [
        "test_accuracy",
        "final_val_accuracy",
        "early_val_accuracy",
        "final_eval_train_accuracy",
        "final_test_loss",
        "final_val_loss",
        "generalization_gap",
        "l2_regularizer_param_norm",
        "l2_regularizer_graph_norm",
        "norm_deficit_eligible_mean",
    ]
    inter_df = compute_factorial_interaction_report(
        res_df, metrics=metrics_to_analyze, primary_metric="test_accuracy"
    )
    inter_df.to_csv(cfg.interaction_csv, index=False)

    print("\n" + "=" * 82)
    print("STAGE 2 SUMMARY: CIFAR-10 2x2 FACTORIAL INTERACTION REPORT")
    print("=" * 82)
    acc_row = inter_df[inter_df["metric"] == "test_accuracy"].iloc[0]
    print(
        f"Primary Endpoint (CIFAR-10 Test Accuracy, n={int(acc_row['n_seeds'])} seeds, "
        f"cost_domain={cfg.cost_domain}, balance_mode={cfg.balance_mode}):\n"
        f"  Group A (Base):               {acc_row['Group_A_mean']*100:6.2f}%\n"
        f"  Group B (Dropout Only):       {acc_row['Group_B_mean']*100:6.2f}%\n"
        f"  Group C (Balance Only):       {acc_row['Group_C_mean']*100:6.2f}%\n"
        f"  Group D (Dropout + Balance):  {acc_row['Group_D_mean']*100:6.2f}%\n"
        f"  Dropout Effect w/o Balance (B - A): {acc_row['dropout_without_balance_B_minus_A']*100:+6.2f} pp\n"
        f"  Dropout Effect w/  Balance (D - C): {acc_row['dropout_with_balance_D_minus_C']*100:+6.2f} pp\n"
        f"  Balance Effect w/  Dropout (D - B): {acc_row['balance_with_dropout_D_minus_B']*100:+6.2f} pp "
        f"(p={acc_row['D_minus_B_p_unadjusted']:.4g})\n"
        f"  Interaction DiD (D-C)-(B-A):        {acc_row['interaction_DiD_mean']*100:+6.2f} pp "
        f"(95% CI [{acc_row['interaction_DiD_ci95_low']*100:+.2f}, "
        f"{acc_row['interaction_DiD_ci95_high']*100:+.2f}] pp, "
        f"p={acc_row['interaction_DiD_p_unadjusted']:.4g})"
    )


def run_stage3_controls_and_dropout_sweep(cfg: CNNExperimentConfig) -> None:
    """
    Stage 3:
      1. Full 2x2 Mechanistic Control comparing Balance vs. Matched Random Rescaling
         both WITHOUT dropout (C_ref vs. C_rand) and WITH dropout (D_ref vs. D_rand).
      2. Orthogonal 1D Dropout Probability Sweeps (1D Conv sweep + 1D FC sweep)
         evaluated strictly on the validation set (`test_loader=None`) to avoid
         test-set selection leakage.
    """
    configure_reproducibility(cfg.deterministic)
    run_stage1_verification_tests(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = cfg.active_seeds()
    train_ds, train_eval_ds, val_ds, test_ds = prepare_cifar10_splits(cfg)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    control_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n--- Stage 3 Controls & Orthogonal 1D Validation Sweeps (Seed {seed}) ---")
        # 1a. Dropout Off: Group C (Balance) vs. C_rand (Matched Random Rescaling)
        sum_c, _, ref_logs_c = train_and_evaluate_condition(
            cfg, seed, "C_ref", "Group_C_BalanceOnly_Ref", False, "interleaved",
            train_ds, train_eval_loader, val_loader, test_loader, device
        )
        sum_c_rand, _, _ = train_and_evaluate_condition(
            cfg, seed, "C_rand", "Control_NoDrop_RandomRescale", False, "random_rescaled",
            train_ds, train_eval_loader, val_loader, test_loader, device,
            reference_epoch_log_lambdas=ref_logs_c
        )

        # 1b. Dropout On: Group D (Balance) vs. D_rand (Matched Random Rescaling)
        sum_d, _, ref_logs_d = train_and_evaluate_condition(
            cfg, seed, "D_ref", "Group_D_Interaction_Ref", True, "interleaved",
            train_ds, train_eval_loader, val_loader, test_loader, device
        )
        sum_d_rand, _, _ = train_and_evaluate_condition(
            cfg, seed, "D_rand", "Control_Dropout_RandomRescale", True, "random_rescaled",
            train_ds, train_eval_loader, val_loader, test_loader, device,
            reference_epoch_log_lambdas=ref_logs_d
        )
        control_rows.extend([sum_c, sum_c_rand, sum_d, sum_d_rand])

        # 2a. Orthogonal 1D Conv Dropout Sweep (holding fc_dropout_rate fixed at cfg.fc_dropout_rate)
        for c_p in cfg.conv_dropout_sweep_1d:
            tag = f"1DConvSweep_pConv{c_p:.2f}_pFC{cfg.fc_dropout_rate:.2f}"
            use_d = (c_p > 0.0) or (cfg.fc_dropout_rate > 0.0)
            sum_b_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"B_{tag}", f"Sweep_DropOnly_{tag}", use_d, "none",
                train_ds, train_eval_loader, val_loader, None, device,
                conv_dropout_override=c_p, fc_dropout_override=cfg.fc_dropout_rate
            )
            sum_d_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"D_{tag}", f"Sweep_DropPlusBal_{tag}", use_d, "interleaved",
                train_ds, train_eval_loader, val_loader, None, device,
                conv_dropout_override=c_p, fc_dropout_override=cfg.fc_dropout_rate
            )
            sweep_rows.extend([sum_b_sw, sum_d_sw])

        # 2b. Orthogonal 1D FC Dropout Sweep (holding conv_dropout_rate fixed at cfg.conv_dropout_rate)
        for f_p in cfg.fc_dropout_sweep_1d:
            tag = f"1DFCSweep_pConv{cfg.conv_dropout_rate:.2f}_pFC{f_p:.2f}"
            use_d = (cfg.conv_dropout_rate > 0.0) or (f_p > 0.0)
            sum_b_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"B_{tag}", f"Sweep_DropOnly_{tag}", use_d, "none",
                train_ds, train_eval_loader, val_loader, None, device,
                conv_dropout_override=cfg.conv_dropout_rate, fc_dropout_override=f_p
            )
            sum_d_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"D_{tag}", f"Sweep_DropPlusBal_{tag}", use_d, "interleaved",
                train_ds, train_eval_loader, val_loader, None, device,
                conv_dropout_override=cfg.conv_dropout_rate, fc_dropout_override=f_p
            )
            sweep_rows.extend([sum_b_sw, sum_d_sw])

    pd.DataFrame(control_rows).to_csv(cfg.stage3_controls_csv, index=False)
    pd.DataFrame(sweep_rows).to_csv(cfg.stage3_sweep_csv, index=False)
    print(
        f"\nStage 3 outputs written to:\n"
        f"  - {cfg.stage3_controls_csv}\n"
        f"  - {cfg.stage3_sweep_csv}"
    )


# =============================================================================
# 8. CLI Entrypoint
# =============================================================================

def parse_args() -> CNNExperimentConfig:
    parser = argparse.ArgumentParser(description="CNN Synaptic Balancing Experiments (Stages 1-3, v1.1.0)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="1: Verification gate only; 2: CIFAR-10 2x2 factorial; 3: Controls & 1D sweeps")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "pilot", "full"],
                        help="smoke (seed 42, 5 ep), pilot (seeds 43-47, 60 ep), full (seeds 48-55, 60 ep)")
    parser.add_argument("--cost-domain", type=str, default="parameter",
                        choices=["parameter", "connection_graph"],
                        help="'parameter' (unique trainable L2, primary) or 'connection_graph' (Sec 8.2 Eq. 8.1)")
    parser.add_argument("--balance-mode", type=str, default="layer", choices=["layer", "channel"],
                        help="'layer' for Coordinated Layer Balancing (Sec 8.3 primary) or 'channel' (secondary)")
    parser.add_argument("--fc-balance-mode", type=str, default="layer", choices=["layer", "neuron"],
                        help="Balancing granularity for downstream FC hidden layers")
    parser.add_argument("--conv-dropout-mode", type=str, default="elementwise",
                        choices=["elementwise", "channel"],
                        help="'elementwise' (B,C,H,W) or 'channel' (Dropout2d B,C,1,1)")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Stateless SGD learning rate")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save final model checkpoints")
    parser.add_argument("--no-deterministic", action="store_true", help="Disable strict deterministic algorithms")
    args = parser.parse_args()

    cfg = CNNExperimentConfig(
        stage=args.stage,
        run_mode=args.mode,
        cost_domain=args.cost_domain,
        balance_mode=args.balance_mode,
        fc_balance_mode=args.fc_balance_mode,
        conv_dropout_mode=args.conv_dropout_mode,
        learning_rate=args.lr,
        deterministic=not args.no_deterministic,
        save_checkpoints=args.save_checkpoints,
    )
    if args.epochs is not None:
        cfg.epochs_smoke = args.epochs
        cfg.epochs_main = args.epochs
    return cfg


if __name__ == "__main__":
    config = parse_args()
    if config.stage == 1:
        configure_reproducibility(config.deterministic)
        run_stage1_verification_tests(config)
    elif config.stage == 2:
        run_stage2_cifar10_factorial(config)
    elif config.stage == 3:
        run_stage3_controls_and_dropout_sweep(config)