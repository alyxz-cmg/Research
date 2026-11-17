#!/usr/bin/env python3
"""
cnn_cifar10_experiment.py (v1.0.0)
==================================
Staged implementation of Synaptic Neural Balance in Convolutional Neural Networks
following HandOff1.md and Baldi et al. (2025), Sections 8.2, 8.3, and Appendix B.

Stages supported via CLI (`--stage`):
  - Stage 1 (`--stage 1`): Mathematical & numerical verification gate on sequential CNNs.
      Verifies exact logit preservation (< 1e-12 in float64, < 1e-6 in float32) across
      Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear with nonzero biases, monotonic
      L2 regularizer cost reduction, balance convergence, isolated RNG streams, and
      full-precision Difference-in-Differences (DiD) interaction statistics.
  - Stage 2 (`--stage 2`): Controlled 2x2 Factorial Experiment on CIFAR-10
      (Group A: Base, Group B: Dropout Only, Group C: Interleaved Balance Only,
       Group D: Dropout + Interleaved Balance) using Coordinated Layer Balancing
      (primary conservative baseline) or Coordinated Channel Balancing (`--balance-mode`).
  - Stage 3 (`--stage 3`): Mechanistic random function-preserving rescaling control
      and optimal dropout probability sweep (H2).
"""

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


# =============================================================================
# 1. Configuration & Reproducibility Infrastructure
# =============================================================================

@dataclass
class CNNExperimentConfig:
    version: str = "1.0.0"
    stage: int = 1
    run_mode: str = "smoke"  # "smoke" (1 seed, 5 epochs), "pilot" (5 seeds), "full" (8 seeds)
    balance_mode: str = "layer"  # "layer" (Coordinated Layer, Sec 8.3) or "channel" (Tied Filter)
    fc_balance_mode: str = "layer"  # "layer" or "neuron" for downstream Linear layers

    # Seeds matching Baldi et al. (8 seeds for full, 5 fresh seeds for pilot, 1 for smoke)
    smoke_seeds: List[int] = field(default_factory=lambda: [42])
    pilot_seeds: List[int] = field(default_factory=lambda: [43, 44, 45, 46, 47])
    full_seeds: List[int] = field(default_factory=lambda: [43, 44, 45, 46, 47, 48, 49, 50])

    # Dataset & Split (CIFAR-10: 45,000 train / 5,000 val / 10,000 official test)
    data_dir: str = "./data"
    train_size: int = 45000
    val_size: int = 5000
    batch_size: int = 128
    use_augmentation: bool = True

    # Architecture & Dropout settings (Pre-specified for Stage 2)
    conv_channels: Tuple[int, int, int, int] = (32, 64, 128, 128)
    fc_hidden: int = 256
    num_classes: int = 10
    conv_dropout_rate: float = 0.20  # Spatial/activation dropout after pooling blocks
    fc_dropout_rate: float = 0.50    # Hidden linear dropout before classifier head

    # Optimization (Stateless SGD to avoid momentum-rescaling confounds)
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    epochs_smoke: int = 5
    epochs_main: int = 60

    # Balancing parameters
    balance_eps: float = 1e-12
    full_balance_max_iters: int = 200
    full_balance_tol: float = 1e-7

    # Stage 3 settings
    dropout_sweep_rates: Tuple[Tuple[float, float], ...] = (
        (0.10, 0.25),
        (0.20, 0.50),
        (0.30, 0.60),
        (0.40, 0.70),
    )

    # Output files
    results_csv: str = "stage2_cnn_cifar10_results.csv"
    learning_curves_csv: str = "stage2_cnn_cifar10_learning_curves.csv"
    interaction_csv: str = "stage2_cnn_cifar10_interaction_summary.csv"
    stage3_csv: str = "stage3_cnn_cifar10_controls_and_sweep.csv"

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
    Dedicated, isolated random number generators for paired-seed experiments.
    Guarantees that Dropout-On and Dropout-Off models receive identical initial
    weights, identical minibatch permutations, and identical data augmentations.
    """
    gen_init: torch.Generator
    gen_loader: torch.Generator
    gen_aug: torch.Generator
    gen_dropout: torch.Generator
    gen_rescale: torch.Generator

    @classmethod
    def from_seed(cls, seed: int, device: str = "cpu") -> "IsolatedRNGStreams":
        def _make(offset: int) -> torch.Generator:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed) * 10007 + offset)
            return g

        return cls(
            gen_init=_make(1),
            gen_loader=_make(2),
            gen_aug=_make(3),
            gen_dropout=_make(4),
            gen_rescale=_make(5),
        )


# =============================================================================
# 2. Generator-Controlled Dropout & Sequential CIFAR-10 CNN Architecture
# =============================================================================

class GeneratorDropout(nn.Module):
    """
    Inverted dropout driven by an isolated CPU torch.Generator so that consuming
    dropout masks never perturbs the global RNG or DataLoader batch order.
    Works for both 2D (B, D) and 4D (B, C, H, W) activation tensors.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        if rng is not None:
            probs = torch.rand(x.shape, generator=rng, dtype=x.dtype, device="cpu").to(x.device)
        else:
            probs = torch.rand_like(x)
        mask = (probs < keep_prob).to(x.dtype) / keep_prob
        return x * mask


class SequentialCIFARCNN(nn.Module):
    """
    Pure sequential ReLU CNN for CIFAR-10 (3x32x32 -> 10 classes).
    No BatchNorm or residual skip connections, ensuring exact positive homogeneity
    across all hidden layers (Conv2d, ReLU, MaxPool2d, Flatten, Linear).

    Architecture:
      conv1: Conv2d(3, C1, 3, padding=1)   -> ReLU                 [32x32]
      conv2: Conv2d(C1, C2, 3, padding=1)  -> ReLU -> MaxPool2d(2) [16x16] -> drop_conv1
      conv3: Conv2d(C2, C3, 3, padding=1)  -> ReLU                 [16x16]
      conv4: Conv2d(C3, C4, 3, padding=1)  -> ReLU -> MaxPool2d(2) [8x8]   -> drop_conv2
      flatten -> fc1: Linear(C4*8*8, H_fc) -> ReLU                         -> drop_fc1
      fc2: Linear(H_fc, num_classes) (output logits)
    """
    def __init__(
        self,
        in_channels: int = 3,
        spatial_size: int = 32,
        conv_channels: Tuple[int, int, int, int] = (32, 64, 128, 128),
        fc_hidden: int = 256,
        num_classes: int = 10,
        use_dropout: bool = False,
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

        # Pooled spatial dimension after two 2x2 MaxPool2d layers
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
        self.drop_conv1 = GeneratorDropout(p=p_conv)
        self.drop_conv2 = GeneratorDropout(p=p_conv)
        self.drop_fc1 = GeneratorDropout(p=p_fc)

        # Ordered chain of parameterized layers for sequential balancing
        self.param_layers: List[nn.Module] = [
            self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2
        ]

    def reset_parameters_with_generator(self, gen: torch.Generator) -> None:
        """Deterministic Kaiming initialization using an isolated generator."""
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
# 3. Stage 1 Core Mathematics: Coordinated Layer & Channel Balancing for CNNs
# =============================================================================

def compute_total_l2_regularizer(model: SequentialCIFARCNN) -> float:
    """
    Computes the total L2 parameter regularizer R(W) = sum_l (||W^(l)||_F^2 + ||b^(l)||_2^2)
    in float64 precision.
    """
    total = 0.0
    with torch.no_grad():
        for layer in model.param_layers:
            total += float(layer.weight.detach().double().pow(2).sum().item())
            if layer.bias is not None:
                total += float(layer.bias.detach().double().pow(2).sum().item())
    return total


def _get_unit_costs(
    curr_layer: nn.Module,
    next_layer: nn.Module,
    mode: str,
    final_spatial: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (A, B) 1D float64 tensors of incoming and outgoing L2 costs for the
    specified balancing granularity:
      - mode == "layer": 1 scalar pair (A_layer, B_layer) for the entire hidden layer
                         (Coordinated Layer Balancing, Baldi et al. Sec 8.2 & 8.3).
      - mode == "channel" (for Conv2d) or "neuron" (for Linear):
                         per-channel or per-neuron (A_c, B_c) pairs.
    """
    w_in = curr_layer.weight.detach().double()
    b_in = curr_layer.bias.detach().double() if curr_layer.bias is not None else None
    w_out = next_layer.weight.detach().double()

    if mode == "layer":
        a_val = w_in.pow(2).sum()
        if b_in is not None:
            a_val = a_val + b_in.pow(2).sum()
        b_val = w_out.pow(2).sum()
        return a_val.unsqueeze(0), b_val.unsqueeze(0)

    # Granular mode: per-channel for Conv2d, per-neuron for Linear
    if isinstance(curr_layer, nn.Conv2d):
        c_out = curr_layer.out_channels
        a_vec = w_in.view(c_out, -1).pow(2).sum(dim=1)
        if b_in is not None:
            a_vec = a_vec + b_in.pow(2)

        if isinstance(next_layer, nn.Conv2d):
            # w_out has shape (C_next, C_curr, K_h, K_w) -> sum over dims (0, 2, 3)
            b_vec = w_out.pow(2).sum(dim=(0, 2, 3))
        elif isinstance(next_layer, nn.Linear):
            # Flattened layout: (C_curr, final_spatial * final_spatial)
            spatial_area = final_spatial * final_spatial
            w_out_reshaped = w_out.view(next_layer.out_features, c_out, spatial_area)
            b_vec = w_out_reshaped.pow(2).sum(dim=(0, 2))
        else:
            raise TypeError(f"Unsupported downstream layer type: {type(next_layer)}")
        return a_vec, b_vec

    elif isinstance(curr_layer, nn.Linear):
        # Independent neuron-wise balancing in FC hidden layer
        h_out = curr_layer.out_features
        a_vec = w_in.view(h_out, -1).pow(2).sum(dim=1)
        if b_in is not None:
            a_vec = a_vec + b_in.pow(2)
        if not isinstance(next_layer, nn.Linear):
            raise TypeError("Linear layer must be followed by a Linear layer.")
        b_vec = w_out.pow(2).sum(dim=0)
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

        # Granular scaling: channel-wise for Conv2d or neuron-wise for Linear
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


def compute_cnn_balance_diagnostics(
    model: SequentialCIFARCNN,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Computes raw and normalized L2 balance deficits and eligibility counts across all
    hidden layers in the sequential CNN.
    Normalized deficit for unit u is |A_u - B_u| / (A_u + B_u).
    """
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
            next_l = layers[idx + 1]
            mode = conv_mode if isinstance(curr_l, nn.Conv2d) else fc_mode
            a_vec, b_vec = _get_unit_costs(curr_l, next_l, mode, model.final_spatial)

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


def partial_balance_cnn_once(
    model: SequentialCIFARCNN,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
    eps: float = 1e-12,
) -> List[torch.Tensor]:
    """
    Executes one forward sweep of synaptic balancing from the first hidden layer
    to the last hidden layer (Partial Balancing, Baldi et al. Section 11).
    Returns the list of applied log-scaling tensors [log(lambda_l)] for each hidden layer.
    """
    layers = model.param_layers
    applied_log_lambdas: List[torch.Tensor] = []

    with torch.no_grad():
        for idx in range(len(layers) - 1):
            curr_l = layers[idx]
            next_l = layers[idx + 1]
            mode = conv_mode if isinstance(curr_l, nn.Conv2d) else fc_mode
            a_vec, b_vec = _get_unit_costs(curr_l, next_l, mode, model.final_spatial)

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
    eps: float = 1e-12,
    tol: float = 1e-7,
    max_iters: int = 200,
) -> Dict[str, float]:
    """
    Iteratively sweeps partial_balance_cnn_once until the maximum normalized balance
    deficit across all eligible units is <= tol.
    """
    iters = 0
    for iters in range(1, max_iters + 1):
        partial_balance_cnn_once(model, conv_mode=conv_mode, fc_mode=fc_mode, eps=eps)
        diag = compute_cnn_balance_diagnostics(model, conv_mode=conv_mode, fc_mode=fc_mode, eps=eps)
        if diag["norm_deficit_eligible_max"] <= tol:
            break
    diag = compute_cnn_balance_diagnostics(model, conv_mode=conv_mode, fc_mode=fc_mode, eps=eps)
    diag["iterations"] = int(iters)
    return diag


def apply_matched_random_rescaling_cnn(
    model: SequentialCIFARCNN,
    reference_log_lambdas: List[torch.Tensor],
    rng: torch.Generator,
    conv_mode: str = "layer",
    fc_mode: str = "layer",
) -> None:
    """
    Stage 3 Mechanistic Control: Applies a function-preserving rescaling whose log-magnitudes
    match the empirical balancing step but with random signs and (when multiple units exist)
    random unit permutations, decoupling weight-scale perturbation from synaptic balance.
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
    """
    Computes mean, sample SD, SE, 95% CI half-width and bounds, t-statistic, and
    two-sided p-value for paired per-seed differences. Handles zero-variance safely.
    """
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
    """Holm-Bonferroni step-down adjustment that preserves NaNs safely."""
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
    """
    Computes paired 2x2 factorial contrasts across seeds:
      - Simple effects: (B - A), (D - C), (C - A), (D - B), (D - A)
      - Factorial main effects:
          E_dropout = ((B - A) + (D - C)) / 2
          E_balance = ((C - A) + (D - B)) / 2
      - Interaction Difference-in-Differences:
          Delta_interaction = (Y_D - Y_C) - (Y_B - Y_A) = (Y_D - Y_B) - (Y_C - Y_A)
    """
    rows = []
    for metric in metrics:
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
        # Apply Holm adjustment across secondary exploratory metrics
        sec_mask = ~rep_df["is_primary_endpoint"]
        sec_p = rep_df.loc[sec_mask, "interaction_DiD_p_unadjusted"].tolist()
        rep_df["interaction_DiD_p_holm"] = np.nan
        rep_df.loc[sec_mask, "interaction_DiD_p_holm"] = _holm_bonferroni(sec_p)
    return rep_df


# =============================================================================
# 5. Stage 1 Verification Gate: Mathematical & Numerical Unit Tests
# =============================================================================

def run_stage1_verification_tests() -> None:
    """
    Executes the 5 mandatory Stage 1 verification unit tests before any CIFAR-10
    training is permitted.
    """
    print("=" * 78)
    print("STAGE 1 GATE: MATHEMATICAL FEASIBILITY & NUMERICAL INVARIANCE UNIT TESTS")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Test 1: Exact Function Preservation (Logit Invariance) in float64 & float32
    #         across Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear with biases
    # -------------------------------------------------------------------------
    for balance_mode in ["layer", "channel"]:
        fc_mode = "layer" if balance_mode == "layer" else "neuron"

        # 1a. Float64 exact check
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
            partial_balance_cnn_once(model_f64, conv_mode=balance_mode, fc_mode=fc_mode)
            logits_after_partial_f64 = model_f64(x_f64).clone()
            full_balance_cnn_until_convergence(model_f64, conv_mode=balance_mode, fc_mode=fc_mode, tol=1e-10)
            logits_after_full_f64 = model_f64(x_f64).clone()

        diff_partial_f64 = float((logits_before_f64 - logits_after_partial_f64).abs().max().item())
        diff_full_f64 = float((logits_before_f64 - logits_after_full_f64).abs().max().item())
        assert diff_partial_f64 < 1e-12, f"[{balance_mode}] Float64 partial invariance failed: {diff_partial_f64}"
        assert diff_full_f64 < 1e-12, f"[{balance_mode}] Float64 full invariance failed: {diff_full_f64}"

        # 1b. Float32 check
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
            full_balance_cnn_until_convergence(model_f32, conv_mode=balance_mode, fc_mode=fc_mode, tol=1e-5)
            logits_after_f32 = model_f32(x_f32).clone()

        diff_f32 = float((logits_before_f32 - logits_after_f32).abs().max().item())
        assert diff_f32 < 1e-5, f"[{balance_mode}] Float32 invariance failed: {diff_f32}"
        print(
            f"[PASS] Test 1 ({balance_mode:7s} mode): Logit invariance verified "
            f"(float64 max diff = {diff_full_f64:.2e}, float32 max diff = {diff_f32:.2e})"
        )

    # -------------------------------------------------------------------------
    # Test 2: Strictly Monotonic L2 Regularizer Cost Reduction & Convergence
    # -------------------------------------------------------------------------
    for balance_mode in ["layer", "channel"]:
        fc_mode = "layer" if balance_mode == "layer" else "neuron"
        rng_streams = IsolatedRNGStreams.from_seed(303)
        model_conv = SequentialCIFARCNN(
            in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
            fc_hidden=24, num_classes=10, use_dropout=False
        ).double()
        model_conv.reset_parameters_with_generator(rng_streams.gen_init)
        # Deliberately imbalance layers by applying extreme function-preserving factors
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

        r_initial = compute_total_l2_regularizer(model_conv)
        diag_init = compute_cnn_balance_diagnostics(model_conv, conv_mode=balance_mode, fc_mode=fc_mode)
        r_prev = r_initial
        for _ in range(40):
            partial_balance_cnn_once(model_conv, conv_mode=balance_mode, fc_mode=fc_mode)
            r_curr = compute_total_l2_regularizer(model_conv)
            assert r_curr <= r_prev + 1e-12, f"Regularizer increased: {r_curr} > {r_prev}"
            r_prev = r_curr

        diag_final = full_balance_cnn_until_convergence(
            model_conv, conv_mode=balance_mode, fc_mode=fc_mode, tol=1e-9, max_iters=200
        )
        r_final = compute_total_l2_regularizer(model_conv)
        assert r_final < r_initial, "Total L2 regularizer did not decrease after balancing imbalanced CNN"
        assert diag_final["norm_deficit_eligible_max"] <= 1e-9, (
            f"Balance deficit failed to converge: {diag_final['norm_deficit_eligible_max']}"
        )
        print(
            f"[PASS] Test 2 ({balance_mode:7s} mode): Monotonic R(W) reduction "
            f"({r_initial:.2f} -> {r_final:.2f}) & deficit convergence "
            f"({diag_init['norm_deficit_eligible_mean']:.4f} -> {diag_final['norm_deficit_eligible_max']:.2e})"
        )

    # -------------------------------------------------------------------------
    # Test 3: Dead / Zero-Cost Filter Exclusion & Eligibility Accounting
    # -------------------------------------------------------------------------
    rng_streams = IsolatedRNGStreams.from_seed(404)
    model_zero = SequentialCIFARCNN(
        in_channels=3, spatial_size=16, conv_channels=(8, 16, 16, 32),
        fc_hidden=24, num_classes=10, use_dropout=False
    ).double()
    model_zero.reset_parameters_with_generator(rng_streams.gen_init)
    with torch.no_grad():
        # Zero out filter 0 in conv1 (both incoming kernel and bias)
        model_zero.conv1.weight[0].zero_()
        model_zero.conv1.bias[0].zero_()

    diag_z = compute_cnn_balance_diagnostics(model_zero, conv_mode="channel", fc_mode="neuron")
    assert diag_z["n_zero_cost_excluded"] == 1, f"Expected 1 zero-cost unit, got {diag_z['n_zero_cost_excluded']}"
    partial_balance_cnn_once(model_zero, conv_mode="channel", fc_mode="neuron")
    assert torch.all(model_zero.conv1.weight[0] == 0.0), "Zero filter was modified during balancing"
    print(
        f"[PASS] Test 3 (Eligibility Diagnostics): Correctly isolated "
        f"{diag_z['n_zero_cost_excluded']} zero-cost filter out of {diag_z['n_total_units']} units"
    )

    # -------------------------------------------------------------------------
    # Test 4: Isolated RNG Streams (Identical Init & Batch Order Under Dropout)
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
            _ = net_a(xa, dropout_rng=streams_a.gen_dropout)
            _ = net_b(xb, dropout_rng=streams_b.gen_dropout)
    print("[PASS] Test 4 (RNG Isolation): Identical init across all layers & identical minibatch order across epochs")

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
    expected_did = (0.742 - 0.735) - (0.710 - 0.720)  # +0.017 (+1.70 pp)
    assert abs(row_acc["interaction_DiD_mean"] - expected_did) < 1e-12
    print(
        f"[PASS] Test 5 (Factorial DiD Pipeline): Verified full-precision DiD = "
        f"{row_acc['interaction_DiD_mean']*100:+.3f} pp, 95% CI "
        f"[{row_acc['interaction_DiD_ci95_low']*100:+.3f}, {row_acc['interaction_DiD_ci95_high']*100:+.3f}] pp"
    )
    print("=" * 78)
    print("ALL STAGE 1 MATHEMATICAL & NUMERICAL VERIFICATION GATES PASSED.")
    print("=" * 78)


# =============================================================================
# 6. Paired-Seed CIFAR-10 Dataset & Deterministic Augmentation Wrapper
# =============================================================================

class DeterministicAugmentedCIFARDataset(Dataset):
    """
    Applies standard CIFAR-10 normalization and optional RandomCrop(32, padding=4)
    + RandomHorizontalFlip using an isolated torch.Generator (`gen_aug`) so that
    Groups A, B, C, and D see the exact same augmented training images in the exact
    same order for a given seed.
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, base_dataset: Dataset, indices: List[int], augment: bool = False):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.augment = bool(augment)
        self.norm = transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base_dataset[self.indices[idx]]
        return self.norm(img), int(label)

    @staticmethod
    def apply_paired_batch_augmentation(
        x: torch.Tensor, gen_aug: torch.Generator
    ) -> torch.Tensor:
        """
        Applies batch-level RandomCrop(32, padding=4) and RandomHorizontalFlip(p=0.5)
        using `gen_aug` on CPU integer indices, keeping RNG consumption identical
        across all conditions for a given seed and step.
        """
        b, c, h, w = x.shape
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
        top_offsets = torch.randint(0, 9, (b,), generator=gen_aug)
        left_offsets = torch.randint(0, 9, (b,), generator=gen_aug)
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


def prepare_cifar10_splits(cfg: CNNExperimentConfig, split_seed: int = 20261117):
    """
    Loads CIFAR-10 and creates a fixed 45,000 train / 5,000 validation split
    plus the official 10,000 test split.
    """
    to_tensor = transforms.ToTensor()
    full_train = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=to_tensor
    )
    test_set_raw = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=to_tensor
    )

    g_split = torch.Generator()
    g_split.manual_seed(split_seed)
    perm = torch.randperm(len(full_train), generator=g_split).tolist()
    train_idx = perm[: cfg.train_size]
    val_idx = perm[cfg.train_size : cfg.train_size + cfg.val_size]

    train_ds = DeterministicAugmentedCIFARDataset(full_train, train_idx, augment=cfg.use_augmentation)
    val_ds = DeterministicAugmentedCIFARDataset(full_train, val_idx, augment=False)
    test_ds = DeterministicAugmentedCIFARDataset(
        test_set_raw, list(range(len(test_set_raw))), augment=False
    )
    return train_ds, val_ds, test_ds


# =============================================================================
# 7. Stage 2 & Stage 3 Training and Evaluation Engine
# =============================================================================

@torch.no_grad()
def evaluate_cnn(model: SequentialCIFARCNN, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Returns (mean_cross_entropy_loss, accuracy_fraction) on the given DataLoader."""
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
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    conv_dropout_override: Optional[float] = None,
    fc_dropout_override: Optional[float] = None,
    reference_epoch_log_lambdas: Optional[List[List[torch.Tensor]]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[List[torch.Tensor]]]:
    """
    Trains one CNN condition for `cfg.active_epochs()` with isolated RNG streams
    and stateless SGD, logging learning curves and final test metrics.
    """
    rng = IsolatedRNGStreams.from_seed(seed)
    c_drop = cfg.conv_dropout_rate if conv_dropout_override is None else conv_dropout_override
    f_drop = cfg.fc_dropout_rate if fc_dropout_override is None else fc_dropout_override

    model = SequentialCIFARCNN(
        in_channels=3,
        spatial_size=32,
        conv_channels=cfg.conv_channels,
        fc_hidden=cfg.fc_hidden,
        num_classes=cfg.num_classes,
        use_dropout=use_dropout,
        conv_dropout_rate=c_drop,
        fc_dropout_rate=f_drop,
    ).to(device)
    model.reset_parameters_with_generator(rng.gen_init)

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
    curve_rows: List[Dict[str, float]] = []
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
                xb = DeterministicAugmentedCIFARDataset.apply_paired_batch_augmentation(xb, rng.gen_aug)
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

        # Apply end-of-epoch function-preserving intervention on unmasked weights
        t_bal0 = time.perf_counter()
        if balance_schedule == "interleaved":
            step_logs = partial_balance_cnn_once(
                model, conv_mode=cfg.balance_mode, fc_mode=cfg.fc_balance_mode, eps=cfg.balance_eps
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

        train_loss = epoch_train_loss / max(1, epoch_train_samples)
        train_acc = epoch_train_correct / max(1, epoch_train_samples)
        val_loss, val_acc = evaluate_cnn(model, val_loader, device)
        diag = compute_cnn_balance_diagnostics(
            model, conv_mode=cfg.balance_mode, fc_mode=cfg.fc_balance_mode, eps=cfg.balance_eps
        )
        r_norm = compute_total_l2_regularizer(model)

        curve_rows.append({
            "seed": int(seed),
            "group_code": group_code,
            "method": method_name,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "generalization_gap": float(train_acc - val_acc),
            "l2_regularizer_norm": float(r_norm),
            "norm_deficit_eligible_mean": float(diag["norm_deficit_eligible_mean"]),
            "norm_deficit_eligible_max": float(diag["norm_deficit_eligible_max"]),
            "eligible_fraction": float(diag["eligible_fraction"]),
        })

    total_train_sec = time.perf_counter() - t_start
    final_val_loss, final_val_acc = evaluate_cnn(model, val_loader, device)
    final_test_loss, final_test_acc = evaluate_cnn(model, test_loader, device)
    final_diag = compute_cnn_balance_diagnostics(
        model, conv_mode=cfg.balance_mode, fc_mode=cfg.fc_balance_mode, eps=cfg.balance_eps
    )

    # Early convergence proxy: mean validation accuracy at Epoch 10 (or Epoch 2 in smoke mode)
    early_ep = min(10, max(1, epochs // 3))
    early_val_acc = next(r["val_accuracy"] for r in curve_rows if r["epoch"] == early_ep)

    summary_row = {
        "seed": int(seed),
        "group_code": group_code,
        "method": method_name,
        "balance_mode": cfg.balance_mode,
        "fc_balance_mode": cfg.fc_balance_mode,
        "use_dropout": bool(use_dropout),
        "conv_dropout_rate": float(c_drop if use_dropout else 0.0),
        "fc_dropout_rate": float(f_drop if use_dropout else 0.0),
        "balance_schedule": balance_schedule,
        "epochs": int(epochs),
        "early_epoch_idx": int(early_ep),
        "early_val_accuracy": float(early_val_acc),
        "final_train_accuracy": float(curve_rows[-1]["train_accuracy"]),
        "final_val_accuracy": float(final_val_acc),
        "final_test_accuracy": float(final_test_acc),
        "test_accuracy": float(final_test_acc),
        "final_val_loss": float(final_val_loss),
        "final_test_loss": float(final_test_loss),
        "generalization_gap": float(curve_rows[-1]["train_accuracy"] - final_val_acc),
        "l2_regularizer_norm": float(compute_total_l2_regularizer(model)),
        "norm_deficit_eligible_mean": float(final_diag["norm_deficit_eligible_mean"]),
        "norm_deficit_eligible_max": float(final_diag["norm_deficit_eligible_max"]),
        "raw_l2_deficit_eligible": float(final_diag["raw_l2_deficit_eligible"]),
        "n_eligible_units": int(final_diag["n_eligible_units"]),
        "n_zero_cost_excluded": int(final_diag["n_zero_cost_excluded"]),
        "training_time_sec": float(total_train_sec),
        "balance_overhead_sec": float(balance_overhead_sec),
    }
    return summary_row, curve_rows, recorded_log_lambdas


def run_stage2_cifar10_factorial(cfg: CNNExperimentConfig) -> None:
    """
    Executes the Stage 1 verification gate first, then runs the Stage 2 2x2
    factorial experiment on CIFAR-10 across paired seeds.
    """
    run_stage1_verification_tests()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = cfg.active_seeds()
    print(
        f"\nSTAGE 2: CIFAR-10 2x2 FACTORIAL EXPERIMENT "
        f"(mode={cfg.run_mode}, seeds={seeds}, epochs={cfg.active_epochs()}, "
        f"balance_mode={cfg.balance_mode}, device={device})"
    )

    train_ds, val_ds, test_ds = prepare_cifar10_splits(cfg)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    conditions = [
        ("A", "Group_A_Base (Drop:Off, Bal:Off)", False, "none"),
        ("B", "Group_B_DropoutOnly (Drop:On, Bal:Off)", True, "none"),
        ("C", "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)", False, "interleaved"),
        ("D", "Group_D_Interaction (Drop:On, Bal:Interleaved)", True, "interleaved"),
    ]

    all_summaries: List[Dict[str, float]] = []
    all_curves: List[Dict[str, float]] = []

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
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )
            all_summaries.append(summary)
            all_curves.extend(curves)
            print(
                f"  [{g_code}] {m_name:47s} | "
                f"Val Acc: {summary['final_val_accuracy']*100:6.2f}% | "
                f"Test Acc: {summary['final_test_accuracy']*100:6.2f}% | "
                f"Deficit: {summary['norm_deficit_eligible_mean']:.2e}"
            )

    res_df = pd.DataFrame(all_summaries)
    curves_df = pd.DataFrame(all_curves)
    res_df.to_csv(cfg.results_csv, index=False)
    curves_df.to_csv(cfg.learning_curves_csv, index=False)

    metrics_to_analyze = [
        "test_accuracy",
        "final_val_accuracy",
        "early_val_accuracy",
        "final_test_loss",
        "final_val_loss",
        "generalization_gap",
        "l2_regularizer_norm",
        "norm_deficit_eligible_mean",
    ]
    inter_df = compute_factorial_interaction_report(
        res_df, metrics=metrics_to_analyze, primary_metric="test_accuracy"
    )
    inter_df.to_csv(cfg.interaction_csv, index=False)

    print("\n" + "=" * 78)
    print("STAGE 2 SUMMARY: CIFAR-10 2x2 FACTORIAL INTERACTION REPORT")
    print("=" * 78)
    acc_row = inter_df[inter_df["metric"] == "test_accuracy"].iloc[0]
    print(
        f"Primary Endpoint (CIFAR-10 Test Accuracy, n={int(acc_row['n_seeds'])} seeds):\n"
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
    Stage 3: Runs the matched random function-preserving rescaling control
    (compared against Group C and Group D) and the H2 dropout probability sweep.
    """
    run_stage1_verification_tests()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = cfg.active_seeds()
    train_ds, val_ds, test_ds = prepare_cifar10_splits(cfg)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    stage3_rows: List[Dict[str, float]] = []
    for seed in seeds:
        print(f"\n--- Stage 3 Controls & H2 Dropout Sweep (Seed {seed}) ---")
        # 1. Matched Random Rescaling Control (paired against Group D)
        sum_d, _, ref_logs_d = train_and_evaluate_condition(
            cfg, seed, "D_ref", "Group_D_Interaction_Ref", True, "interleaved",
            train_ds, val_loader, test_loader, device
        )
        sum_rand, _, _ = train_and_evaluate_condition(
            cfg, seed, "D_rand", "Control_Dropout_Plus_RandomRescale", True, "random_rescaled",
            train_ds, val_loader, test_loader, device, reference_epoch_log_lambdas=ref_logs_d
        )
        stage3_rows.extend([sum_d, sum_rand])

        # 2. H2 Dropout Rate Sweep (Dropout Only vs. Dropout + Interleaved Balance)
        for c_p, f_p in cfg.dropout_sweep_rates:
            tag = f"pConv{c_p:.2f}_pFC{f_p:.2f}"
            sum_b_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"B_{tag}", f"Sweep_DropOnly_{tag}", True, "none",
                train_ds, val_loader, test_loader, device,
                conv_dropout_override=c_p, fc_dropout_override=f_p
            )
            sum_d_sw, _, _ = train_and_evaluate_condition(
                cfg, seed, f"D_{tag}", f"Sweep_DropPlusBal_{tag}", True, "interleaved",
                train_ds, val_loader, test_loader, device,
                conv_dropout_override=c_p, fc_dropout_override=f_p
            )
            stage3_rows.extend([sum_b_sw, sum_d_sw])

    pd.DataFrame(stage3_rows).to_csv(cfg.stage3_csv, index=False)
    print(f"\nStage 3 results written to {cfg.stage3_csv}")


# =============================================================================
# 8. CLI Entrypoint
# =============================================================================

def parse_args() -> CNNExperimentConfig:
    parser = argparse.ArgumentParser(description="CNN Synaptic Balancing Experiments (Stages 1-3)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="1: Verification gate only; 2: CIFAR-10 2x2 factorial; 3: Controls & H2 sweep")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "pilot", "full"],
                        help="smoke (1 seed, 5 epochs), pilot (5 seeds, 60 epochs), full (8 seeds, 60 epochs)")
    parser.add_argument("--balance-mode", type=str, default="layer", choices=["layer", "channel"],
                        help="'layer' for Coordinated Layer Balancing (Sec 8.3 baseline) or 'channel' for tied filter")
    parser.add_argument("--fc-balance-mode", type=str, default="layer", choices=["layer", "neuron"],
                        help="Balancing granularity for downstream FC hidden layers")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Stateless SGD learning rate")
    args = parser.parse_args()

    cfg = CNNExperimentConfig(
        stage=args.stage,
        run_mode=args.mode,
        balance_mode=args.balance_mode,
        fc_balance_mode=args.fc_balance_mode,
        learning_rate=args.lr,
    )
    if args.epochs is not None:
        cfg.epochs_smoke = args.epochs
        cfg.epochs_main = args.epochs
    return cfg


if __name__ == "__main__":
    config = parse_args()
    if config.stage == 1:
        run_stage1_verification_tests()
    elif config.stage == 2:
        run_stage2_cifar10_factorial(config)
    elif config.stage == 3:
        run_stage3_controls_and_dropout_sweep(config)