"""
Minimal Diagnostic Experiment v2.2.8 (Locked 5-Seed Fresh Confirmatory Harness):
Synaptic Balance x Dropout for SNN Computational Efficiency
=================================================================================
Draws on and adapts methodological concepts from:
  - Baldi et al. (2024/2025), "A Theory of Synaptic Neural Balance: From Local to Global Order"
    (L2 neuron-wise balancing equations, bias as clamped input synapse, and stateless SGD setup)
  - Alexos, Domingo, & Baldi (AAAI 2025), "Improving Deep Learning Speed and Performance..."
    (Pre-training full balance and interleaved per-epoch partial balancing schedules)
  - Rueckauer et al. (2017), "Conversion of Continuous-Valued Deep Networks..."
    (Subtraction-reset IF conversion and tied layer-wise 99.9th-percentile normalization)
  - Sorbaro et al. (2020), "Optimizing the Energy Consumption of SNNs..."
    (Multi-spike IF event counting and an inspired fan-out-weighted L1 activation penalty proxy;
     note this harness does not implement Sorbaro's target quadratic loss, QReLU, or Sinabs)
  - NeuroBench (2025), Workload Metrics
    (Manually calculated Dense vs. binary Effective ACs and non-binary Effective MACs)

Formally Locked Inferential Architecture (Frozen Prior to Fresh-Seed Execution):
  1. Single Primary Confirmatory Endpoint (`hypothesis_family_role == "primary_confirmatory_endpoint"`):
     - Metric: `hidden_event_count_synops_proxy_per_sample` (emitted spike events x nonzero fan-out)
     - Condition: `threshold_mode == "recalibrated"` at `inference_steps == 50`
     - Tested at alpha = 0.05 (unadjusted primary test) via conjunctive gate
       `confirmatory_interaction_efficiency_success`:
         (a) Negative DiD interaction: (Y_D - Y_C) - (Y_B - Y_A) < 0 with p_unadjusted < 0.05
         (b) Practical D-vs-B improvement (`practical_D_vs_B_efficiency_success`):
             Y_D - Y_B < 0 with p_{D-B} < 0.05 AND one-sided 95% LCB(Acc_D - Acc_B) > -0.01
         (c) Baseline accuracy safeguard (`passes_stat_acc_noninferiority_D_vs_A`):
             one-sided 95% LCB(Acc_D - Acc_A) > -0.01
       (Note: The 1.0 percentage-point margin delta = 0.01 is a study-specific engineering
       tolerance pre-specified prior to multi-seed data collection.)
  2. Secondary Exploratory Holm Family (`hypothesis_family_role == "exploratory_holm_family"`):
     - Consists of the 130 unique secondary exploratory comparisons across the 10
       (threshold_mode, T) combinations:
         * 10 horizons x 13 SNN-level metrics = 130 SNN cells, minus the 1 primary cell = 129 SNN cells
         * Plus 1 canonical global `ann_test_accuracy` comparison (at the primary reference row)
         * Total exploratory Holm family size: `M_exploratory = 130`.
     - The other 9 repeated `ann_test_accuracy` rows are retained for table lookup with
       `hypothesis_family_role == "duplicate_ann_display_only_excluded_from_holm"` and
       excluded from inflating the Holm divisor.

Summary of v2.2.8 Final Additions:
  - Added `windowed_early_stop_event_count_synops_per_sample` (exact hidden event-count SynOps
    accumulated up to each example's online windowed stopping step `win_step[b]`).
  - Separated the primary confirmatory test from the 130-comparison exploratory Holm family
    and deduplicated `ann_test_accuracy` in `_holm_bonferroni_adjust`.
  - Added `+ 1e-12` numerical tolerance to `val_acc_plateau_reached` (`win_span <= tol + 1e-12`).
  - Expanded Test 7 to verify: (a) Negative-Path Test A where `Y_D - Y_B < 0` has `p > 0.05`
    while `DiD < 0` has `p < 0.001`; (b) Negative-Path Test B where `D` passes `D vs B` accuracy
    non-inferiority but fails `D vs A` accuracy non-inferiority; (c) Negative-Path Test C where
    `D` fails `D vs B` accuracy non-inferiority; and (d) the `n_seeds == 0` missing-horizon
    placeholder row behavior.
  - Set `run_single_seed_smoke_first = False` and `use_fresh_confirmatory_seeds = True` so
    default execution runs the 5 fresh confirmatory seeds `[43, 44, 45, 46, 47]` at 60 epochs.
"""

import copy
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# =====================================================================
# 1. Configuration & Isolated Reproducibility Utilities
# =====================================================================

@dataclass
class ExperimentConfigV2:
    dataset_name: str = "MNIST"
    allow_synthetic_fallback: bool = False
    data_dir: str = "./data"
    # Inspired by Baldi et al. (2025) Sec 11.2-11.3 SGD setup; note Baldi's 3-layer FCN used (784, 25, 128, 10)
    layer_sizes: Tuple[int, ...] = (784, 256, 128, 10)
    dropout_rate: float = 0.5  # Inspired by Sorbaro et al. (2020) 50% rate; applied after hidden ReLUs
    dropout_type: str = "inverted_hidden_relu_isolated_rng"
    balance_norm: int = 2  # Primary: p=2 (L2)
    # Baldi et al. Section 3 treats bias as an incoming synapse from a unit clamped at 1
    include_bias_in_cost: bool = True
    balance_eps: float = 1e-12
    full_balance_tol: float = 0.01
    full_balance_max_iters: int = 200
    # Locked to 60 epochs following the Seed-42 exploratory learning-curve diagnostic
    epochs: int = 60
    val_plateau_window: int = 5  # Rolling window length (epochs) for sustained plateau check
    val_plateau_delta_tol: float = 0.005  # Max-minus-min span tolerance across `val_plateau_window`
    batch_size: int = 128
    # Matches Baldi et al. (Sections 10, 11.2-11.6) stateless SGD with lr=0.001
    optimizer_name: str = "SGD"  # "SGD" or "Adam"
    learning_rate: float = 0.001
    sgd_momentum: float = 0.0
    optimizer_state_mode: str = "covector"  # "covector", "contravariant", "reset", or "none"
    # Sorbaro-inspired fan-out-weighted L1 activation regularizer proxy
    synoploss_alpha: float = 1e-5
    tune_synoploss_on_val: bool = True
    synoploss_val_acc_tolerance: float = 0.015
    synoploss_alpha_grid: Tuple[float, ...] = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
    # SNN conversion, spike emission mode & simulation settings
    input_encoding: str = "analog_current"  # "analog_current" (Option B) or "rate_spikes" (Option A)
    spike_mode: str = "multi_spike"  # "multi_spike" (integer event count) or "binary" (0/1 step count)
    calib_percentile: float = 99.9  # Rueckauer et al. (2017) tied layer-wise activation percentile
    timesteps_budget: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    stability_window_k: int = 5
    # Confirmatory seed configuration:
    # Seed 42 was used to select the 60-epoch training budget. Default execution now runs
    # the 5 fresh confirmatory seeds `[43, 44, 45, 46, 47]` (`run_single_seed_smoke_first=False`).
    exploratory_pilot_seed: int = 42
    run_single_seed_smoke_first: bool = False
    use_fresh_confirmatory_seeds: bool = True
    use_main_8_seeds: bool = False
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    fresh_5_seeds: Tuple[int, ...] = (43, 44, 45, 46, 47)
    main_8_seeds: Tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49)
    fresh_8_seeds: Tuple[int, ...] = (43, 44, 45, 46, 47, 48, 49, 50)
    run_tier2_controls: bool = False  # Locked to False for the core 4-group confirmatory experiment
    # Pre-specified primary confirmatory outcome
    primary_confirmatory_metric: str = "hidden_event_count_synops_proxy_per_sample"
    primary_confirmatory_threshold_mode: str = "recalibrated"
    primary_confirmatory_timestep: int = 50
    acc_noninferiority_margin: float = 0.01  # Study-specific engineering tolerance (1.0 percentage point)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_csv: str = "phase3_v2_2_8_minimal_experiment_results.csv"
    learning_curves_csv: str = "phase3_v2_2_8_learning_curves.csv"
    interaction_csv: str = "phase3_v2_2_8_interaction_summary.csv"
    output_plot: str = "tradeoff_pareto_v2_2_8_accuracy_vs_ops.png"
    gap_plot: str = "ann_to_snn_convergence_gap_v2_2_8.png"

    def __post_init__(self) -> None:
        if (
            self.spike_mode == "binary"
            and self.primary_confirmatory_metric == "hidden_event_count_synops_proxy_per_sample"
        ):
            self.primary_confirmatory_metric = "hidden_neurobench_eff_ACs_per_sample"

    def get_active_seeds(self) -> List[int]:
        """
        Returns the active seed list without mutating `self.seeds`:
          - When `run_single_seed_smoke_first=True`, returns `[self.seeds[0]]` (Seed 42).
          - When `run_single_seed_smoke_first=False`:
              * If `use_fresh_confirmatory_seeds=True`, returns 5 or 8 fresh seeds excluding
                `exploratory_pilot_seed` (42) if custom `self.seeds` was not overridden.
              * Otherwise returns the configured 5-seed or 8-seed list including Seed 42.
        """
        if self.run_single_seed_smoke_first:
            return [self.seeds[0]]
        if self.use_main_8_seeds:
            if self.use_fresh_confirmatory_seeds:
                return list(self.fresh_8_seeds)
            return list(self.main_8_seeds) if len(self.seeds) < 8 else list(self.seeds)
        if self.use_fresh_confirmatory_seeds and self.seeds == [42, 43, 44, 45, 46]:
            return list(self.fresh_5_seeds)
        return list(self.seeds)


def _sync_device(device: str) -> None:
    """Synchronizes CUDA before/after timing blocks (introduces a stream-completion barrier)."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _float_or_nan(val: Optional[float]) -> float:
    """Preserves full float64 precision for finite numbers and maps None/NaN/Inf to `np.nan`."""
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


def set_global_seed(seed: int) -> None:
    """Pins global seeds for deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IsolatedRNGStreams:
    """
    Maintains strictly separated random number generators for:
      1. Model weight initialization (`gen_init`)
      2. DataLoader batch shuffling (`gen_loader`)
      3. Dropout mask sampling (`gen_dropout`)
      4. Random rescaling negative control (`gen_rescale`)
      5. SNN Poisson/rate input spike encoding (`gen_snn_input`)
    """
    def __init__(self, base_seed: int, device: str = "cpu"):
        self.base_seed = base_seed
        self.device = device
        self.gen_init = torch.Generator(device="cpu").manual_seed(base_seed + 1000)
        self.gen_loader = torch.Generator(device="cpu").manual_seed(base_seed + 2000)
        self.gen_rescale = torch.Generator(device="cpu").manual_seed(base_seed + 3000)
        gen_dev = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.gen_dropout = torch.Generator(device=gen_dev).manual_seed(base_seed + 4000)
        self.gen_snn_input = torch.Generator(device=gen_dev).manual_seed(base_seed + 5000)


# =====================================================================
# 2. Architecture with Generator-Isolated Inverted Dropout
# =====================================================================

class GeneratorDropout(nn.Module):
    """
    Inverted dropout layer that draws Bernoulli masks exclusively from an isolated
    `torch.Generator` and optionally caches `last_mask` for direct mask verification.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.record_last_mask: bool = False
        self.last_mask: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            if self.record_last_mask:
                self.last_mask = torch.ones_like(x)
            return x
        keep_prob = 1.0 - self.p
        mask = torch.empty_like(x).bernoulli_(keep_prob, generator=generator)
        if self.record_last_mask:
            self.last_mask = mask.detach().clone()
        return x * mask / keep_prob


class FullyConnectedReLU(nn.Module):
    """
    Feed-forward ReLU (BiLU) network with deterministic generator-isolated initialization
    and inverted dropout after hidden ReLU activations.
    """
    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        dropout_rate: float = 0.0,
        init_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.linears = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True)
            for i in range(len(layer_sizes) - 1)
        ])
        self.dropouts = nn.ModuleList([
            GeneratorDropout(p=dropout_rate) for _ in range(len(layer_sizes) - 2)
        ])
        self.fanouts = [layer_sizes[i + 2] for i in range(len(layer_sizes) - 2)]

        if init_generator is not None:
            self._reset_parameters_deterministically(init_generator)

    @torch.no_grad()
    def _reset_parameters_deterministically(self, gen: torch.Generator) -> None:
        for lin in self.linears:
            bound = 1.0 / math.sqrt(lin.in_features)
            w_cpu = torch.empty_like(lin.weight, device="cpu").uniform_(-bound, bound, generator=gen)
            lin.weight.copy_(w_cpu.to(lin.weight.device))
            if lin.bias is not None:
                b_cpu = torch.empty_like(lin.bias, device="cpu").uniform_(-bound, bound, generator=gen)
                lin.bias.copy_(b_cpu.to(lin.bias.device))

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        dropout_generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = x.view(x.size(0), -1)
        hidden_acts = []
        for i, linear in enumerate(self.linears[:-1]):
            x = F.relu(linear(x))
            x = self.dropouts[i](x, generator=dropout_generator)
            if return_activations:
                hidden_acts.append(x)
        logits = self.linears[-1](x)
        if return_activations:
            return logits, hidden_acts
        return logits, None


# =====================================================================
# 3. Phase 1: Synaptic Balancing, Eligibility & Manifold Diagnostics
# =====================================================================

@torch.no_grad()
def compute_balance_metrics(
    model: FullyConnectedReLU,
    p: int = 2,
    eps: float = 1e-12,
    include_bias_in_cost: bool = True,
) -> Dict[str, float]:
    """
    Computes weight-space balance eligibility (`n_active_eligible`, where A_j > eps
    and B_j > eps) and raw/normalized deficits across eligible and all hidden neurons.
    """
    raw_deficit_all = 0.0
    raw_deficit_eligible = 0.0
    norm_deficits_all = []
    norm_deficits_eligible = []
    max_ratio_deviation_eligible = 0.0

    n_total = 0
    n_active_eligible = 0
    n_zero_cost = 0
    n_near_zero_cost = 0
    n_bias_only = 0

    for layer_idx in range(len(model.linears) - 1):
        w_in = model.linears[layer_idx].weight
        b_in = model.linears[layer_idx].bias
        w_out = model.linears[layer_idx + 1].weight

        if p == 2:
            w_in_cost = torch.sum(w_in ** 2, dim=1)
            b_in_cost = (b_in ** 2) if b_in is not None else torch.zeros_like(w_in_cost)
            A = w_in_cost + b_in_cost if include_bias_in_cost else w_in_cost
            B = torch.sum(w_out ** 2, dim=0)
        elif p == 1:
            w_in_cost = torch.sum(torch.abs(w_in), dim=1)
            b_in_cost = torch.abs(b_in) if b_in is not None else torch.zeros_like(w_in_cost)
            A = w_in_cost + b_in_cost if include_bias_in_cost else w_in_cost
            B = torch.sum(torch.abs(w_out), dim=0)
        else:
            raise ValueError(f"Unsupported norm p={p}. Use p=1 or p=2.")

        H = A.numel()
        n_total += H

        bias_only_mask = (w_in_cost <= eps) & (b_in_cost > eps)
        n_bias_only += int(bias_only_mask.sum().item())

        zero_mask = (A == 0.0) | (B == 0.0)
        eligible_mask = (A > eps) & (B > eps)
        near_zero_mask = (~zero_mask) & (~eligible_mask)

        n_zero_cost += int(zero_mask.sum().item())
        n_active_eligible += int(eligible_mask.sum().item())
        n_near_zero_cost += int(near_zero_mask.sum().item())

        diff_sq = (A - B) ** 2
        raw_deficit_all += float(torch.sum(diff_sq).item())
        if eligible_mask.any():
            raw_deficit_eligible += float(torch.sum(diff_sq[eligible_mask]).item())

        d_j = torch.abs(A - B) / (A + B + eps)
        norm_deficits_all.append(d_j.detach().cpu().numpy())

        if eligible_mask.any():
            norm_deficits_eligible.append(d_j[eligible_mask].detach().cpu().numpy())
            norm_ratio = (A[eligible_mask] / B[eligible_mask]) ** (1.0 / p)
            dev = float(torch.max(torch.abs(norm_ratio - 1.0)).item())
            max_ratio_deviation_eligible = max(max_ratio_deviation_eligible, dev)

    all_d = np.concatenate(norm_deficits_all) if norm_deficits_all else np.array([0.0])
    elig_d = (
        np.concatenate(norm_deficits_eligible)
        if norm_deficits_eligible
        else np.array([0.0])
    )

    return {
        "n_total_neurons": int(n_total),
        "n_active_eligible": int(n_active_eligible),
        "eligible_neuron_fraction": float(n_active_eligible / max(1, n_total)),
        "n_zero_cost": int(n_zero_cost),
        "n_near_zero_cost": int(n_near_zero_cost),
        "n_bias_only_neurons": int(n_bias_only),
        "raw_deficit_all": float(raw_deficit_all),
        "raw_deficit_eligible": float(raw_deficit_eligible),
        "mean_norm_deficit_all": float(np.mean(all_d)),
        "max_norm_deficit_all": float(np.max(all_d)),
        "mean_norm_deficit_eligible": float(np.mean(elig_d)),
        "max_norm_deficit_eligible": float(np.max(elig_d)),
        "max_ratio_deviation_eligible": float(max_ratio_deviation_eligible),
    }


@torch.no_grad()
def compute_edge_log_ratio_norm(
    cum_log_lambdas: List[torch.Tensor],
    layer_sizes: Tuple[int, ...],
    include_bias_edges: bool = True,
) -> Tuple[float, float]:
    """
    Computes two diagnostic transformation magnitude summaries:
      1. Neuron-vector L2 norm: sqrt(sum_j (log Lambda_j)^2)
      2. Directed edge-wise log-ratio L2 norm across all synaptic connections (i -> j):
         L_{ij} = log(Lambda_i) - log(Lambda_j), where visible input, output, and clamped
         bias source units have multiplier Lambda = 1 (log Lambda = 0).
    """
    if not cum_log_lambdas:
        return 0.0, 0.0
    all_neuron_logs = torch.cat([t.view(-1).cpu().float() for t in cum_log_lambdas])
    neuron_l2_norm = float(torch.norm(all_neuron_logs, p=2).item())

    full_layer_logs = [torch.zeros(layer_sizes[0], dtype=torch.float32)]
    for t in cum_log_lambdas:
        full_layer_logs.append(t.view(-1).cpu().float())
    full_layer_logs.append(torch.zeros(layer_sizes[-1], dtype=torch.float32))

    edge_sq_sum = 0.0
    for l_idx in range(len(full_layer_logs) - 1):
        log_pre = full_layer_logs[l_idx].unsqueeze(1)
        log_post = full_layer_logs[l_idx + 1].unsqueeze(0)
        edge_diff = log_pre - log_post
        edge_sq_sum += float(torch.sum(edge_diff ** 2).item())

    if include_bias_edges:
        edge_sq_sum += float(torch.sum(all_neuron_logs ** 2).item())

    return neuron_l2_norm, math.sqrt(edge_sq_sum)


@torch.no_grad()
def _rescale_optimizer_param_state(
    optimizer: Optional[torch.optim.Optimizer],
    param: torch.Tensor,
    scale_tensor: torch.Tensor,
    mode: str = "covector",
) -> None:
    """
    Coordinate-consistent transformation of stored gradient moments when a
    parameter tensor is reparameterized by `param' = scale_tensor * param`.
    """
    if optimizer is None or mode == "none":
        return
    if param not in optimizer.state:
        return
    state = optimizer.state[param]
    if not state:
        return

    if mode == "reset":
        for buf_key in ("momentum_buffer", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if buf_key in state and state[buf_key] is not None:
                state[buf_key].zero_()
        return

    if mode == "covector":
        m_factor = 1.0 / scale_tensor
        v_factor = 1.0 / (scale_tensor ** 2)
    elif mode == "contravariant":
        m_factor = scale_tensor
        v_factor = scale_tensor ** 2
    else:
        raise ValueError(f"Unknown optimizer_state_mode: {mode}")

    if "momentum_buffer" in state and state["momentum_buffer"] is not None:
        state["momentum_buffer"].mul_(m_factor)
    if "exp_avg" in state and state["exp_avg"] is not None:
        state["exp_avg"].mul_(m_factor)
    if "exp_avg_sq" in state and state["exp_avg_sq"] is not None:
        state["exp_avg_sq"].mul_(v_factor)
    if "max_exp_avg_sq" in state and state["max_exp_avg_sq"] is not None:
        state["max_exp_avg_sq"].mul_(v_factor)


@torch.no_grad()
def partial_balance_step(
    model: FullyConnectedReLU,
    p: int = 2,
    eps: float = 1e-12,
    include_bias_in_cost: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_state_mode: str = "covector",
) -> List[torch.Tensor]:
    """
    Applies one deterministic sweep of independent neuron-wise synaptic balancing from input
    to output hidden layers on the full unmasked parameters (Baldi et al., Sections 3, 5, 11).
    """
    log_lambdas = []
    for layer_idx in range(len(model.linears) - 1):
        lin_in = model.linears[layer_idx]
        lin_out = model.linears[layer_idx + 1]

        w_in = lin_in.weight
        b_in = lin_in.bias
        w_out = lin_out.weight

        if p == 2:
            A = torch.sum(w_in ** 2, dim=1)
            if include_bias_in_cost and b_in is not None:
                A = A + b_in ** 2
            B = torch.sum(w_out ** 2, dim=0)
            exponent = 0.25
        elif p == 1:
            A = torch.sum(torch.abs(w_in), dim=1)
            if include_bias_in_cost and b_in is not None:
                A = A + torch.abs(b_in)
            B = torch.sum(torch.abs(w_out), dim=0)
            exponent = 0.5
        else:
            raise ValueError(f"Unsupported norm p={p}. Use p=1 or p=2.")

        valid = (A > eps) & (B > eps)
        lam = torch.ones_like(A)
        lam[valid] = (B[valid] / A[valid]) ** exponent
        log_lambdas.append(torch.log(lam).detach().cpu())

        s_in_w = lam.unsqueeze(1)
        s_out_w = (1.0 / lam).unsqueeze(0)

        w_in.mul_(s_in_w)
        _rescale_optimizer_param_state(optimizer, w_in, s_in_w, mode=optimizer_state_mode)

        if b_in is not None:
            b_in.mul_(lam)
            _rescale_optimizer_param_state(optimizer, b_in, lam, mode=optimizer_state_mode)

        w_out.mul_(s_out_w)
        _rescale_optimizer_param_state(optimizer, w_out, s_out_w, mode=optimizer_state_mode)

    return log_lambdas


@torch.no_grad()
def full_balance_network(
    model: FullyConnectedReLU,
    p: int = 2,
    tol: float = 0.01,
    max_iters: int = 200,
    eps: float = 1e-12,
    include_bias_in_cost: bool = True,
) -> Tuple[int, bool, List[torch.Tensor]]:
    """
    Iteratively applies `partial_balance_step` until the incoming/outgoing norm
    ratios of eligible neurons converge within `tol` of 1.0 (Baldi et al., Section 11).
    Returns `(sweeps_used, converged_within_tol, cum_log_lambdas)`.
    """
    cum_log_lambdas = [
        torch.zeros(model.layer_sizes[l + 1], dtype=torch.float32)
        for l in range(len(model.linears) - 1)
    ]
    for it in range(1, max_iters + 1):
        step_log_lam = partial_balance_step(
            model, p=p, eps=eps, include_bias_in_cost=include_bias_in_cost
        )
        for l in range(len(cum_log_lambdas)):
            cum_log_lambdas[l] += step_log_lam[l]
        metrics = compute_balance_metrics(
            model, p=p, eps=eps, include_bias_in_cost=include_bias_in_cost
        )
        if metrics["max_ratio_deviation_eligible"] <= tol:
            return it, True, cum_log_lambdas
    return max_iters, False, cum_log_lambdas


@torch.no_grad()
def matched_random_function_preserving_rescale(
    model: FullyConnectedReLU,
    gen: torch.Generator,
    reference_log_lambdas: Optional[List[torch.Tensor]] = None,
    fallback_log_std: float = 0.25,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_state_mode: str = "covector",
) -> List[torch.Tensor]:
    """
    Neuron-magnitude-matched random function-preserving rescaling control.
    Permutes and randomly flips the signs of the empirical neuron log-scaling factors
    within each hidden layer (preserving the layer-wise neuron-vector L2 norm while
    ablating the local balance direction A_j = B_j).
    """
    applied_log_lambdas = []
    for layer_idx in range(len(model.linears) - 1):
        lin_in = model.linears[layer_idx]
        lin_out = model.linears[layer_idx + 1]
        H = lin_in.out_features
        dev = lin_in.weight.device

        if reference_log_lambdas is not None and len(reference_log_lambdas) > layer_idx:
            ref = reference_log_lambdas[layer_idx].view(-1).cpu()
            perm = torch.randperm(H, generator=gen)
            signs = (torch.empty(H).bernoulli_(0.5, generator=gen) * 2.0 - 1.0)
            log_lam_cpu = ref[perm] * signs
        else:
            log_lam_cpu = torch.empty(H).normal_(0.0, fallback_log_std, generator=gen)

        applied_log_lambdas.append(log_lam_cpu.clone())
        lam = torch.exp(log_lam_cpu).to(dev)

        s_in_w = lam.unsqueeze(1)
        s_out_w = (1.0 / lam).unsqueeze(0)

        lin_in.weight.mul_(s_in_w)
        _rescale_optimizer_param_state(optimizer, lin_in.weight, s_in_w, mode=optimizer_state_mode)
        if lin_in.bias is not None:
            lin_in.bias.mul_(lam)
            _rescale_optimizer_param_state(optimizer, lin_in.bias, lam, mode=optimizer_state_mode)
        lin_out.weight.mul_(s_out_w)
        _rescale_optimizer_param_state(optimizer, lin_out.weight, s_out_w, mode=optimizer_state_mode)

    return applied_log_lambdas


# =====================================================================
# 4. Phase 4: Validated ANN-to-SNN Conversion & Exact Int64 Accounting
# =====================================================================

class ConvertedSNN(nn.Module):
    """
    Integrate-and-Fire (IF) SNN with subtraction (soft) membrane reset.
    All integer event-count and fan-out products are accumulated in exact 64-bit
    integer arithmetic (`torch.int64` and Python `int`) before dividing by `n`:
      1. `hidden_event_count_synops_proxy_per_sample`: sum_{t=1}^T sum_j s_j(t) * F_j^{nonzero}
      2. `windowed_early_stop_event_count_synops_per_sample`: sum_{t=1}^{t*_stop} sum_j s_j(t) * F_j^{nonzero}
         (Exact hidden event-count SynOps accumulated up to each sample's online stopping step).
      3. `hidden_neurobench_eff_ACs_per_sample`: sum_{t,j} 1[s_j(t) == 1] * F_j^{nonzero}
      4. `hidden_neurobench_eff_MACs_per_sample`: sum_{t,j} 1[s_j(t) > 1] * F_j^{nonzero}
      5. `hidden_binary_active_step_ACs_per_sample`: sum_{t,j} 1[s_j(t) > 0] * F_j^{nonzero}
      6. `hidden_dense_SynOps_per_sample`: T * sum_l (H_l * F_l).
    """
    def __init__(
        self,
        ann: FullyConnectedReLU,
        threshold_mode: str = "recalibrated",
        input_encoding: str = "analog_current",
        spike_mode: str = "multi_spike",
        calib_loader: Optional[DataLoader] = None,
        percentile: float = 99.9,
        fixed_threshold: float = 1.0,
    ):
        super().__init__()
        if spike_mode not in ("multi_spike", "binary"):
            raise ValueError(f"Unsupported spike_mode: {spike_mode}")
        self.layer_sizes = ann.layer_sizes
        self.num_hidden = len(ann.linears) - 1
        self.threshold_mode = threshold_mode
        self.input_encoding = input_encoding
        self.spike_mode = spike_mode
        self.linears = nn.ModuleList([
            nn.Linear(lin.in_features, lin.out_features, bias=(lin.bias is not None))
            for lin in ann.linears
        ])

        with torch.no_grad():
            for snn_lin, ann_lin in zip(self.linears, ann.linears):
                snn_lin.weight.copy_(ann_lin.weight)
                if ann_lin.bias is not None:
                    snn_lin.bias.copy_(ann_lin.bias)

            if threshold_mode == "recalibrated":
                if calib_loader is None:
                    raise ValueError("calib_loader is required when threshold_mode='recalibrated'")
                layer_scales = self._compute_layer_scales(ann, calib_loader, percentile=percentile)
                prev_scale = 1.0
                for l in range(self.num_hidden):
                    curr_scale = max(layer_scales[l], 1e-8)
                    self.linears[l].weight.mul_(prev_scale / curr_scale)
                    if self.linears[l].bias is not None:
                        self.linears[l].bias.div_(curr_scale)
                    prev_scale = curr_scale
                self.linears[-1].weight.mul_(prev_scale)
                thresholds = [1.0] * self.num_hidden
            elif threshold_mode == "fixed":
                thresholds = [fixed_threshold] * self.num_hidden
            else:
                raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")

        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))

        w_first = self.linears[0].weight.data
        self.input_effective_fanout = (w_first.abs() > 1e-12).to(torch.int64).sum(dim=0)
        self.input_dense_fanout = int(self.linears[0].out_features)
        self.input_dense_macs_single_pass = int(
            self.linears[0].in_features * self.linears[0].out_features
        )

        self.effective_fanouts: List[torch.Tensor] = []
        self.hidden_dense_ops_single_step = 0
        for l in range(self.num_hidden):
            w_next = self.linears[l + 1].weight.data
            nonzero_fanout = (w_next.abs() > 1e-12).to(torch.int64).sum(dim=0)
            self.effective_fanouts.append(nonzero_fanout)
            self.hidden_dense_ops_single_step += int(
                self.linears[l + 1].in_features * self.linears[l + 1].out_features
            )

    @staticmethod
    @torch.no_grad()
    def _compute_layer_scales(
        ann: FullyConnectedReLU, calib_loader: DataLoader, percentile: float = 99.9
    ) -> List[float]:
        ann.eval()
        device = next(ann.parameters()).device
        acts_per_layer = [[] for _ in range(len(ann.linears) - 1)]
        for xb, _ in calib_loader:
            xb = xb.to(device)
            _, hidden_acts = ann(xb, return_activations=True)
            for l, act in enumerate(hidden_acts):
                acts_per_layer[l].append(act.detach().cpu())

        scales = []
        for l in range(len(acts_per_layer)):
            flat_acts = torch.cat(acts_per_layer[l], dim=0).numpy()
            val = float(np.percentile(flat_acts, percentile))
            scales.append(val if val > 0.0 else 1.0)
        return scales

    def _emit_spikes(self, v: torch.Tensor, v_th: torch.Tensor) -> torch.Tensor:
        if self.spike_mode == "multi_spike":
            return torch.where(v >= v_th, torch.floor(v / v_th), torch.zeros_like(v))
        return (v >= v_th).float()

    @torch.no_grad()
    def simulate_batch_readout(
        self,
        xb: torch.Tensor,
        T: int,
        input_gen: Optional[torch.Generator] = None,
        return_diagnostics: bool = False,
    ):
        """
        Runs SNN for T steps on batch `xb` and returns time-averaged readout current
        S_out(T) / T, plus optional saturation/overflow diagnostics.
        """
        self.eval()
        device = xb.device
        bsz = xb.size(0)
        x_flat = xb.view(bsz, -1)

        v_mem = [
            torch.zeros(bsz, self.layer_sizes[l + 1], device=device)
            for l in range(self.num_hidden)
        ]
        out_acc = torch.zeros(bsz, self.layer_sizes[-1], device=device)

        if self.input_encoding == "analog_current":
            static_cur0 = self.linears[0](x_flat)
        else:
            static_cur0 = None
            x_prob = torch.clamp(x_flat, 0.0, 1.0)

        multispike_steps = 0
        post_reset_overflow_steps = 0
        total_neuron_steps = 0

        for _ in range(1, T + 1):
            if self.input_encoding == "analog_current":
                cur = static_cur0
            else:
                in_spikes = torch.empty_like(x_prob).bernoulli_(x_prob, generator=input_gen)
                cur = self.linears[0](in_spikes)

            for l in range(self.num_hidden):
                v_mem[l] = v_mem[l] + cur
                v_th = self.thresholds[l]
                multispike_steps += int((v_mem[l] >= 2.0 * v_th).sum().item())
                spikes = self._emit_spikes(v_mem[l], v_th)
                v_mem[l] = v_mem[l] - spikes * v_th
                post_reset_overflow_steps += int((v_mem[l] >= v_th).sum().item())
                total_neuron_steps += v_mem[l].numel()
                cur = self.linears[l + 1](spikes)

            out_acc = out_acc + cur

        readout = out_acc / float(T)
        if not return_diagnostics:
            return readout
        denom = max(1, total_neuron_steps)
        return readout, {
            "multispike_or_saturation_step_fraction": multispike_steps / denom,
            "post_reset_overflow_fraction": post_reset_overflow_steps / denom,
        }

    @torch.no_grad()
    def evaluate_budgets(
        self,
        test_loader: DataLoader,
        timesteps_budget: List[int],
        ann_test_acc: float,
        stability_window_k: int = 5,
        input_gen: Optional[torch.Generator] = None,
    ) -> Dict[int, Dict[str, object]]:
        """
        Simulates SNN across `timesteps_budget`, accumulating all integer operation
        counts in exact 64-bit integer arithmetic (`torch.int64` and Python `int`),
        including per-example dynamic early-stopped event-count SynOps.
        """
        self.eval()
        device = next(self.parameters()).device
        max_T = max(timesteps_budget)
        budget_set = set(timesteps_budget)

        accum = {
            t: {
                "correct": 0,
                "windowed_overall_policy_correct": 0,
                "windowed_converged_correct": 0,
                "windowed_unconverged_correct": 0,
                "samples": 0,
                "hidden_event_count_synops_proxy_sum": 0,
                "windowed_early_stop_event_count_synops_sum": 0,
                "hidden_neurobench_eff_acs_sum": 0,
                "hidden_neurobench_eff_macs_sum": 0,
                "hidden_binary_active_step_acs_sum": 0,
                "hidden_active_full_fanout_ops_sum": 0,
                "hidden_dense_synops_sum": 0,
                "input_static_eff_macs_sum": 0,
                "input_static_dense_macs_sum": 0,
                "input_repeated_eff_macs_sum": 0,
                "input_repeated_dense_ops_sum": 0,
                "input_spike_eff_acs_sum": 0,
                "input_spike_active_full_fanout_ops_sum": 0,
                "analog_input_zero_frac_sum": 0.0,
                "layer_spikes_sum": np.zeros(self.num_hidden, dtype=np.int64),
                "silent_frac_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "firing_rate_median_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "firing_rate_p99_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "firing_rate_max_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "firing_rate_max_mean_ratio_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "total_hidden_spikes": 0,
                "total_hidden_active_neuron_steps": 0,
                "total_hidden_neuron_steps": 0,
                "total_input_spikes": 0,
                "total_input_active_neuron_steps": 0,
                "total_input_neuron_steps": 0,
                "multispike_step_count": 0,
                "post_reset_overflow_count": 0,
                "oracle_first_correct_sum": 0.0,
                "retrospective_convergence_sum": 0.0,
                "unconverged_retrospective_count": 0,
                "windowed_stable_censored_sum": 0.0,
                "windowed_stable_conditional_sum": 0.0,
                "converged_windowed_count": 0,
                "unconverged_windowed_count": 0,
                "sim_wall_clock_sec": 0.0,
            }
            for t in timesteps_budget
        }

        in_fanout_vec_i64 = self.input_effective_fanout.to(device=device, dtype=torch.int64)
        fanout_vecs_i64 = [
            f.to(device=device, dtype=torch.int64) for f in self.effective_fanouts
        ]

        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            bsz = xb.size(0)
            x_flat = xb.view(bsz, -1)

            v_mem = [
                torch.zeros(bsz, self.layer_sizes[l + 1], device=device)
                for l in range(self.num_hidden)
            ]
            out_acc = torch.zeros(bsz, self.layer_sizes[-1], device=device)

            cum_spikes_i64 = [
                torch.zeros(bsz, self.layer_sizes[l + 1], dtype=torch.int64, device=device)
                for l in range(self.num_hidden)
            ]
            cum_binary_eq1_steps_i64 = [
                torch.zeros(bsz, self.layer_sizes[l + 1], dtype=torch.int64, device=device)
                for l in range(self.num_hidden)
            ]
            cum_multispike_gt1_steps_i64 = [
                torch.zeros(bsz, self.layer_sizes[l + 1], dtype=torch.int64, device=device)
                for l in range(self.num_hidden)
            ]
            cum_active_gt0_steps_i64 = [
                torch.zeros(bsz, self.layer_sizes[l + 1], dtype=torch.int64, device=device)
                for l in range(self.num_hidden)
            ]
            cum_pre_ge2_steps = torch.zeros((), dtype=torch.int64, device=device)
            cum_post_overflow_steps = torch.zeros((), dtype=torch.int64, device=device)

            cum_input_spikes_i64 = torch.zeros(
                bsz, self.layer_sizes[0], dtype=torch.int64, device=device
            )
            cum_active_input_steps_i64 = torch.zeros((), dtype=torch.int64, device=device)

            preds_history = torch.full((bsz, max_T), -1, dtype=torch.long, device=device)
            valid_out_history = torch.zeros((bsz, max_T), dtype=torch.bool, device=device)
            # Track per-sample cumulative hidden event-count SynOps at each step t in [1..max_T]
            cum_sample_event_ops_history_i64 = torch.zeros(
                (bsz, max_T), dtype=torch.int64, device=device
            )
            running_sample_event_ops_i64 = torch.zeros(bsz, dtype=torch.int64, device=device)

            oracle_first_correct = torch.full((bsz,), float(max_T), device=device)
            has_identified_oracle = torch.zeros(bsz, dtype=torch.bool, device=device)

            cum_sim_wall_clock_sec = 0.0
            if self.input_encoding == "analog_current":
                _sync_device(str(device))
                t0 = time.perf_counter()
                static_cur0 = self.linears[0](x_flat)
                _sync_device(str(device))
                cum_sim_wall_clock_sec += time.perf_counter() - t0

                nonzero_in_mask_i64 = (x_flat.abs() > 1e-12).to(torch.int64)
                batch_input_eff_macs = int(
                    (nonzero_in_mask_i64 * in_fanout_vec_i64.unsqueeze(0)).sum().item()
                )
                batch_input_zero_frac = float(
                    (1.0 - nonzero_in_mask_i64.to(torch.float64).mean(dim=1)).sum().item()
                )
            else:
                static_cur0 = None
                x_prob = torch.clamp(x_flat, 0.0, 1.0)
                batch_input_eff_macs = 0
                batch_input_zero_frac = 0.0

            step_spikes_buf = [None] * self.num_hidden

            for t in range(1, max_T + 1):
                # --- Synchronized forward-step software simulation timer ---
                _sync_device(str(device))
                t_step_start = time.perf_counter()

                if self.input_encoding == "analog_current":
                    cur = static_cur0
                    in_spikes = None
                else:
                    in_spikes = torch.empty_like(x_prob).bernoulli_(x_prob, generator=input_gen)
                    cur = self.linears[0](in_spikes)

                for l in range(self.num_hidden):
                    v_mem[l] = v_mem[l] + cur
                    v_th = self.thresholds[l]
                    spikes = self._emit_spikes(v_mem[l], v_th)
                    v_mem[l] = v_mem[l] - spikes * v_th
                    step_spikes_buf[l] = spikes
                    cur = self.linears[l + 1](spikes)

                out_acc = out_acc + cur
                preds = out_acc.argmax(dim=1)

                _sync_device(str(device))
                cum_sim_wall_clock_sec += time.perf_counter() - t_step_start
                # --- End synchronized forward-step software simulation timer ---

                if in_spikes is not None:
                    in_sp_i64 = in_spikes.to(torch.int64)
                    cum_input_spikes_i64 += in_sp_i64
                    cum_active_input_steps_i64 += (in_sp_i64 > 0).to(torch.int64).sum()

                for l in range(self.num_hidden):
                    sp = step_spikes_buf[l]
                    sp_i64 = sp.to(torch.int64)
                    v_th = self.thresholds[l]
                    cum_spikes_i64[l] += sp_i64
                    cum_binary_eq1_steps_i64[l] += (sp_i64 == 1).to(torch.int64)
                    cum_multispike_gt1_steps_i64[l] += (sp_i64 > 1).to(torch.int64)
                    cum_active_gt0_steps_i64[l] += (sp_i64 > 0).to(torch.int64)
                    cum_pre_ge2_steps += ((v_mem[l] + sp * v_th) >= (2.0 * v_th)).to(torch.int64).sum()
                    cum_post_overflow_steps += (v_mem[l] >= v_th).to(torch.int64).sum()
                    running_sample_event_ops_i64 += (
                        sp_i64 * fanout_vecs_i64[l].unsqueeze(0)
                    ).sum(dim=1)

                cum_sample_event_ops_history_i64[:, t - 1] = running_sample_event_ops_i64
                nonzero_out = out_acc.abs().sum(dim=1) > 0
                preds_history[:, t - 1] = preds
                valid_out_history[:, t - 1] = nonzero_out

                correct_now = (preds == yb) & nonzero_out
                newly_correct = correct_now & (~has_identified_oracle)
                oracle_first_correct[newly_correct] = float(t)
                has_identified_oracle = has_identified_oracle | newly_correct

                if t in budget_set:
                    rec = accum[t]
                    rec["sim_wall_clock_sec"] += cum_sim_wall_clock_sec
                    rec["correct"] += int((preds == yb).sum().item())
                    rec["samples"] += bsz
                    rec["oracle_first_correct_sum"] += float(
                        torch.clamp(oracle_first_correct, max=float(t)).sum().item()
                    )
                    rec["multispike_step_count"] += int(cum_pre_ge2_steps.item())
                    rec["post_reset_overflow_count"] += int(cum_post_overflow_steps.item())

                    # Retrospective convergence timestep in [1..t]
                    final_pred_t = preds_history[:, t - 1].unsqueeze(1)
                    matches_final = (preds_history[:, :t] == final_pred_t) & valid_out_history[:, :t]
                    suffix_stable = torch.flip(
                        torch.cumprod(torch.flip(matches_final.int(), dims=[1]), dim=1),
                        dims=[1],
                    ).bool()
                    any_stable = suffix_stable.any(dim=1)
                    first_stable_idx = suffix_stable.int().argmax(dim=1) + 1
                    retro_t_vec = torch.where(
                        any_stable,
                        first_stable_idx.float(),
                        torch.full_like(first_stable_idx, float(t), dtype=torch.float32),
                    )
                    rec["retrospective_convergence_sum"] += float(retro_t_vec.sum().item())
                    unconv_retro = (~any_stable) | ((first_stable_idx == t) & (t > 1))
                    rec["unconverged_retrospective_count"] += int(unconv_retro.sum().item())

                    # Online windowed stopping rule separating converged vs. unconverged
                    k_win = min(stability_window_k, t)
                    win_found = torch.zeros(bsz, dtype=torch.bool, device=device)
                    win_step = torch.full((bsz,), float(t), device=device)
                    win_pred = preds.clone()

                    for start_idx in range(0, t - k_win + 1):
                        window_slice = preds_history[:, start_idx : start_idx + k_win]
                        valid_slice = valid_out_history[:, start_idx : start_idx + k_win]
                        ref_p = window_slice[:, 0:1]
                        all_same = (window_slice == ref_p).all(dim=1) & valid_slice.all(dim=1)
                        newly_win = all_same & (~win_found)
                        win_step[newly_win] = float(start_idx + k_win)
                        win_pred[newly_win] = window_slice[newly_win, -1]
                        win_found = win_found | newly_win

                    rec["windowed_stable_censored_sum"] += float(win_step.sum().item())
                    # Exact event-count SynOps accumulated up to each sample's online stopping step `win_step`
                    stop_idx_i64 = (win_step.to(torch.int64) - 1).clamp(min=0, max=t - 1).unsqueeze(1)
                    stopped_ops_per_sample_i64 = torch.gather(
                        cum_sample_event_ops_history_i64, dim=1, index=stop_idx_i64
                    ).squeeze(1)
                    rec["windowed_early_stop_event_count_synops_sum"] += int(
                        stopped_ops_per_sample_i64.sum().item()
                    )

                    if win_found.any():
                        rec["windowed_stable_conditional_sum"] += float(
                            win_step[win_found].sum().item()
                        )
                        rec["windowed_converged_correct"] += int(
                            (win_pred[win_found] == yb[win_found]).sum().item()
                        )
                    if (~win_found).any():
                        rec["windowed_unconverged_correct"] += int(
                            (preds[~win_found] == yb[~win_found]).sum().item()
                        )
                    rec["windowed_overall_policy_correct"] += int((win_pred == yb).sum().item())
                    rec["converged_windowed_count"] += int(win_found.sum().item())
                    rec["unconverged_windowed_count"] += int((~win_found).sum().item())

                    # Input-layer operation accounting (exact Python int accumulation)
                    if self.input_encoding == "analog_current":
                        rec["input_static_eff_macs_sum"] += int(batch_input_eff_macs)
                        rec["input_static_dense_macs_sum"] += int(
                            bsz * self.input_dense_macs_single_pass
                        )
                        rec["input_repeated_eff_macs_sum"] += int(batch_input_eff_macs * t)
                        rec["input_repeated_dense_ops_sum"] += int(
                            bsz * self.input_dense_macs_single_pass * t
                        )
                        rec["analog_input_zero_frac_sum"] += batch_input_zero_frac
                    else:
                        rec["input_static_eff_macs_sum"] += 0
                        rec["input_static_dense_macs_sum"] += 0
                        rec["input_repeated_eff_macs_sum"] += 0
                        rec["input_spike_eff_acs_sum"] += int(
                            (cum_input_spikes_i64 * in_fanout_vec_i64.unsqueeze(0)).sum().item()
                        )
                        rec["input_spike_active_full_fanout_ops_sum"] += int(
                            (cum_input_spikes_i64.sum(dim=1) * self.input_dense_fanout).sum().item()
                        )
                        rec["input_repeated_dense_ops_sum"] += int(
                            bsz * self.input_dense_macs_single_pass * t
                        )
                        rec["total_input_spikes"] += int(cum_input_spikes_i64.sum().item())
                        rec["total_input_active_neuron_steps"] += int(
                            cum_active_input_steps_i64.item()
                        )
                        rec["total_input_neuron_steps"] += int(bsz * self.layer_sizes[0] * t)

                    # Hidden-layer operation accounting (exact int64 products & sums)
                    rec["hidden_dense_synops_sum"] += int(
                        bsz * self.hidden_dense_ops_single_step * t
                    )
                    for l in range(self.num_hidden):
                        s_i64 = cum_spikes_i64[l]
                        eq1_i64 = cum_binary_eq1_steps_i64[l]
                        gt1_i64 = cum_multispike_gt1_steps_i64[l]
                        gt0_i64 = cum_active_gt0_steps_i64[l]
                        f_vec_i64 = fanout_vecs_i64[l].unsqueeze(0)
                        dense_fanout = int(self.linears[l + 1].out_features)

                        rec["hidden_event_count_synops_proxy_sum"] += int(
                            (s_i64 * f_vec_i64).sum().item()
                        )
                        rec["hidden_neurobench_eff_acs_sum"] += int(
                            (eq1_i64 * f_vec_i64).sum().item()
                        )
                        rec["hidden_neurobench_eff_macs_sum"] += int(
                            (gt1_i64 * f_vec_i64).sum().item()
                        )
                        rec["hidden_binary_active_step_acs_sum"] += int(
                            (gt0_i64 * f_vec_i64).sum().item()
                        )
                        rec["hidden_active_full_fanout_ops_sum"] += int(
                            (s_i64.sum(dim=1) * dense_fanout).sum().item()
                        )
                        rec["total_hidden_active_neuron_steps"] += int(gt0_i64.sum().item())
                        rec["layer_spikes_sum"][l] += int(s_i64.sum().item())
                        rec["silent_frac_sum"][l] += float(
                            (s_i64 == 0).to(torch.float64).mean(dim=1).sum().item()
                        )

                        rates_l = (s_i64.to(torch.float64) / float(t)).detach().cpu().numpy()
                        med_per_sample = np.median(rates_l, axis=1)
                        p99_per_sample = np.percentile(rates_l, 99.0, axis=1)
                        max_per_sample = np.max(rates_l, axis=1)
                        mean_per_sample = np.maximum(np.mean(rates_l, axis=1), 1e-8)
                        rec["firing_rate_median_sum"][l] += float(np.sum(med_per_sample))
                        rec["firing_rate_p99_sum"][l] += float(np.sum(p99_per_sample))
                        rec["firing_rate_max_sum"][l] += float(np.sum(max_per_sample))
                        rec["firing_rate_max_mean_ratio_sum"][l] += float(
                            np.sum(max_per_sample / mean_per_sample)
                        )

                        rec["total_hidden_spikes"] += int(s_i64.sum().item())
                        rec["total_hidden_neuron_steps"] += int(
                            bsz * self.layer_sizes[l + 1] * t
                        )

        results = {}
        for t in timesteps_budget:
            rec = accum[t]
            n = max(1, rec["samples"])
            snn_acc = rec["correct"] / n
            hidden_event_proxy = rec["hidden_event_count_synops_proxy_sum"] / n
            stopped_event_proxy = rec["windowed_early_stop_event_count_synops_sum"] / n
            hidden_nb_acs = rec["hidden_neurobench_eff_acs_sum"] / n
            hidden_nb_macs = rec["hidden_neurobench_eff_macs_sum"] / n
            hidden_bin_step_acs = rec["hidden_binary_active_step_acs_sum"] / n
            hidden_active_full = rec["hidden_active_full_fanout_ops_sum"] / n
            hidden_dense = rec["hidden_dense_synops_sum"] / n

            in_static_eff = rec["input_static_eff_macs_sum"] / n
            in_static_dense = rec["input_static_dense_macs_sum"] / n
            in_rep_eff = rec["input_repeated_eff_macs_sum"] / n
            in_rep_dense = rec["input_repeated_dense_ops_sum"] / n
            in_spike_eff = rec["input_spike_eff_acs_sum"] / n
            in_spike_full = rec["input_spike_active_full_fanout_ops_sum"] / n

            if self.input_encoding == "analog_current":
                total_ops_static_in = in_static_dense + hidden_event_proxy
                total_eff_ops_static_in = in_static_eff + hidden_event_proxy
                total_ops_repeated_in = in_rep_dense + hidden_event_proxy
                spiking_pop_sparsity = 1.0 - (
                    rec["total_hidden_active_neuron_steps"]
                    / max(1, rec["total_hidden_neuron_steps"])
                )
            else:
                total_ops_static_in = in_spike_eff + hidden_event_proxy
                total_eff_ops_static_in = in_spike_eff + hidden_event_proxy
                total_ops_repeated_in = in_spike_eff + hidden_event_proxy
                spiking_pop_sparsity = 1.0 - (
                    (
                        rec["total_hidden_active_neuron_steps"]
                        + rec["total_input_active_neuron_steps"]
                    )
                    / max(
                        1,
                        rec["total_hidden_neuron_steps"] + rec["total_input_neuron_steps"],
                    )
                )

            hidden_sparsity = 1.0 - (
                rec["total_hidden_active_neuron_steps"]
                / max(1, rec["total_hidden_neuron_steps"])
            )

            n_conv = rec["converged_windowed_count"]
            n_unconv = rec["unconverged_windowed_count"]
            cond_acc = (
                rec["windowed_converged_correct"] / n_conv if n_conv > 0 else float("nan")
            )
            unconv_acc = (
                rec["windowed_unconverged_correct"] / n_unconv if n_unconv > 0 else float("nan")
            )
            cond_steps = (
                rec["windowed_stable_conditional_sum"] / n_conv if n_conv > 0 else float("nan")
            )

            results[t] = {
                "test_accuracy": float(snn_acc),
                "ann_to_snn_acc_gap": float(ann_test_acc - snn_acc),
                "windowed_early_stop_overall_policy_accuracy": float(
                    rec["windowed_overall_policy_correct"] / n
                ),
                "windowed_early_stop_conditional_accuracy": float(cond_acc),
                "windowed_unconverged_final_accuracy": float(unconv_acc),
                "hidden_event_count_synops_proxy_per_sample": float(hidden_event_proxy),
                "windowed_early_stop_event_count_synops_per_sample": float(stopped_event_proxy),
                "hidden_neurobench_eff_ACs_per_sample": float(hidden_nb_acs),
                "hidden_neurobench_eff_MACs_per_sample": float(hidden_nb_macs),
                "hidden_binary_active_step_ACs_per_sample": float(hidden_bin_step_acs),
                "hidden_active_spike_full_fanout_ops_per_sample": float(hidden_active_full),
                "hidden_dense_SynOps_per_sample": float(hidden_dense),
                "input_eff_MACs_per_sample": float(in_static_eff),
                "input_static_MACs_per_sample": float(in_static_dense),
                "input_repeated_eff_MACs_per_sample": float(in_rep_eff),
                "input_repeated_dense_ops_per_sample": float(in_rep_dense),
                "input_spike_eff_ACs_per_sample": float(in_spike_eff),
                "input_spike_active_full_fanout_ops_per_sample": float(in_spike_full),
                "total_ops_static_input_per_sample": float(total_ops_static_in),
                "total_eff_ops_static_input_per_sample": float(total_eff_ops_static_in),
                "total_ops_repeated_input_per_sample": float(total_ops_repeated_in),
                "spike_count_by_layer": [
                    float(round(float(val) / n, 4)) for val in rec["layer_spikes_sum"]
                ],
                "silent_fraction_by_layer": [
                    float(round(float(val) / n, 4)) for val in rec["silent_frac_sum"]
                ],
                "firing_rate_median_by_layer": [
                    float(round(float(val) / n, 4)) for val in rec["firing_rate_median_sum"]
                ],
                "firing_rate_p99_by_layer": [
                    float(round(float(val) / n, 4)) for val in rec["firing_rate_p99_sum"]
                ],
                "firing_rate_max_by_layer": [
                    float(round(float(val) / n, 4)) for val in rec["firing_rate_max_sum"]
                ],
                "firing_rate_max_to_mean_ratio_by_layer": [
                    float(round(float(val) / n, 4))
                    for val in rec["firing_rate_max_mean_ratio_sum"]
                ],
                "multispike_or_saturation_step_fraction": float(
                    rec["multispike_step_count"] / max(1, rec["total_hidden_neuron_steps"])
                ),
                "post_reset_overflow_fraction": float(
                    rec["post_reset_overflow_count"] / max(1, rec["total_hidden_neuron_steps"])
                ),
                "hidden_activation_sparsity": float(hidden_sparsity),
                "spiking_population_sparsity": float(spiking_pop_sparsity),
                "analog_input_zero_fraction": float(
                    rec["analog_input_zero_frac_sum"] / n
                    if self.input_encoding == "analog_current"
                    else 0.0
                ),
                "retrospective_convergence_steps": float(rec["retrospective_convergence_sum"] / n),
                "unconverged_fraction_retrospective": float(
                    rec["unconverged_retrospective_count"] / n
                ),
                "windowed_stable_steps_censored": float(
                    rec["windowed_stable_censored_sum"] / n
                ),
                "windowed_stable_steps_conditional": float(cond_steps),
                "unconverged_fraction_windowed": float(n_unconv / n),
                "oracle_first_correct_steps": float(rec["oracle_first_correct_sum"] / n),
                "amortized_batch_sim_ms_per_sample": float(
                    (rec["sim_wall_clock_sec"] * 1000.0) / n
                ),
            }
        return results


# =====================================================================
# 5. Comprehensive Verification Unit Tests (Tests 1-7)
# =====================================================================

def run_all_verification_unit_tests(
    cfg: ExperimentConfigV2,
    train_dataset: torch.utils.data.Dataset,
    val_loader: DataLoader,
    calib_loader: DataLoader,
    atol: float = 1e-5,
) -> None:
    """
    Executes all 7 verification unit tests, including:
      - Positive-path AND three negative-path unit tests for `confirmatory_interaction_efficiency_success`
      - Primary vs. Exploratory Holm family split & `ann_test_accuracy` deduplication check
      - Zero-common-seed placeholder row preservation check
      - Exact >2^24 int64 accumulation and 1e-12 float64 input zero-fraction checks.
    """
    print("=====================================================================")
    print("Running Minimal Experiment v2.2.8 Verification Suite (Tests 1-7)")
    print("=====================================================================")

    rng = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device)
    model = FullyConnectedReLU(
        cfg.layer_sizes, dropout_rate=cfg.dropout_rate, init_generator=rng.gen_init
    ).to(cfg.device)
    model.eval()

    with torch.no_grad():
        x_rand = torch.randn(64, cfg.layer_sizes[0], device=cfg.device)
        x_val, _ = next(iter(val_loader))
        x_val = x_val.to(cfg.device)

        out_rand_before, _ = model(x_rand)
        out_val_before, _ = model(x_val)
        m_before = compute_balance_metrics(
            model, p=cfg.balance_norm, eps=cfg.balance_eps, include_bias_in_cost=cfg.include_bias_in_cost
        )

        iters, converged_flag, cum_logs = full_balance_network(
            model,
            p=cfg.balance_norm,
            tol=1e-5,
            max_iters=cfg.full_balance_max_iters,
            eps=cfg.balance_eps,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )

        out_rand_after, _ = model(x_rand)
        out_val_after, _ = model(x_val)
        m_after = compute_balance_metrics(
            model, p=cfg.balance_norm, eps=cfg.balance_eps, include_bias_in_cost=cfg.include_bias_in_cost
        )

        diff_rand = float(torch.max(torch.abs(out_rand_before - out_rand_after)).item())
        diff_val = float(torch.max(torch.abs(out_val_before - out_val_after)).item())
        assert diff_rand < atol and diff_val < atol, "Untrained ANN functional invariance failed!"
        assert m_after["raw_deficit_eligible"] < m_before["raw_deficit_eligible"] * 1e-4

        print(
            f"[Test 1 PASS] Untrained ANN Invariance | Sweeps: {iters} (Converged={converged_flag}) | "
            f"Eligible: {m_after['eligible_neuron_fraction']*100:.1f}% | "
            f"Max diff: {diff_rand:.2e} (rand), {diff_val:.2e} (val)"
        )

        # Test 2: Dead-neuron, bias-only neuron, and bias-edge log-ratio norm diagnostics
        dead_model = copy.deepcopy(model)
        dead_model.linears[1].weight.data[:, 0] = 0.0
        dead_model.linears[0].weight.data[1, :] = 0.0
        partial_balance_step(
            dead_model,
            p=cfg.balance_norm,
            eps=cfg.balance_eps,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        m_dead = compute_balance_metrics(
            dead_model,
            p=cfg.balance_norm,
            eps=cfg.balance_eps,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        _, edge_norm_no_bias = compute_edge_log_ratio_norm(
            cum_logs, cfg.layer_sizes, include_bias_edges=False
        )
        n_l2, edge_norm_with_bias = compute_edge_log_ratio_norm(
            cum_logs, cfg.layer_sizes, include_bias_edges=True
        )
        assert m_dead["n_zero_cost"] >= 1 and m_dead["n_bias_only_neurons"] >= 1
        assert abs(edge_norm_with_bias ** 2 - (edge_norm_no_bias ** 2 + n_l2 ** 2)) < 1e-4
        print(
            f"[Test 2 PASS] Dead/Bias-Only & Bias-Edge Norm Diagnostics | "
            f"Zero-Cost: {m_dead['n_zero_cost']} | Bias-Only: {m_dead['n_bias_only_neurons']} | "
            f"EdgeNorm(with bias): {edge_norm_with_bias:.4f}"
        )

    # Test 3: Paired-seed full parameter, input+label minibatch, and DIRECT dropout mask identity
    rng_a = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device)
    rng_b = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device)
    rng_b2 = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device)
    loader_a = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=rng_a.gen_loader
    )
    loader_b = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=rng_b.gen_loader
    )
    mod_a = FullyConnectedReLU(cfg.layer_sizes, dropout_rate=0.0, init_generator=rng_a.gen_init).to(cfg.device)
    mod_b = FullyConnectedReLU(cfg.layer_sizes, dropout_rate=0.5, init_generator=rng_b.gen_init).to(cfg.device)
    mod_b2 = FullyConnectedReLU(cfg.layer_sizes, dropout_rate=0.5, init_generator=rng_b2.gen_init).to(cfg.device)

    for lin_a, lin_b, lin_b2 in zip(mod_a.linears, mod_b.linears, mod_b2.linears):
        assert torch.equal(lin_a.weight, lin_b.weight) and torch.equal(lin_b.weight, lin_b2.weight)
        assert torch.equal(lin_a.bias, lin_b.bias) and torch.equal(lin_b.bias, lin_b2.bias)

    for d_layer in mod_b.dropouts:
        d_layer.record_last_mask = True
    for d_layer in mod_b2.dropouts:
        d_layer.record_last_mask = True

    mod_a.train()
    mod_b.train()
    mod_b2.train()
    for epoch_check in range(2):
        for (xa, ya), (xb_batch, yb_batch) in zip(loader_a, loader_b):
            assert torch.equal(xa, xb_batch) and torch.equal(ya, yb_batch)
            with torch.no_grad():
                x_sub = xa[:8].to(cfg.device)
                _ = mod_a(x_sub, dropout_generator=rng_a.gen_dropout)
                out_b1, _ = mod_b(x_sub, dropout_generator=rng_b.gen_dropout)
                out_b2, _ = mod_b2(x_sub, dropout_generator=rng_b2.gen_dropout)
                assert torch.equal(out_b1, out_b2)
                for d1, d2 in zip(mod_b.dropouts, mod_b2.dropouts):
                    assert d1.last_mask is not None and d2.last_mask is not None
                    assert torch.equal(d1.last_mask, d2.last_mask), "Direct dropout masks diverged!"
    for d_layer in mod_b.dropouts:
        d_layer.record_last_mask = False
    print(
        "[Test 3 PASS] Paired-Seed Isolation | Verified all weights/biases, "
        "minibatch inputs+labels, and direct Bernoulli dropout mask tensors."
    )

    # Test 4: Full `partial_balance_step` integration test with Adam
    rng_adam = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device)
    mod_adam = FullyConnectedReLU(
        cfg.layer_sizes, dropout_rate=0.0, init_generator=rng_adam.gen_init
    ).to(cfg.device)
    opt_adam = torch.optim.Adam(mod_adam.parameters(), lr=1e-3)
    opt_adam.zero_grad()
    loss_adam = F.cross_entropy(
        mod_adam(x_val)[0], torch.zeros(x_val.size(0), dtype=torch.long, device=cfg.device)
    )
    loss_adam.backward()
    opt_adam.step()

    w0_p = mod_adam.linears[0].weight
    b0_p = mod_adam.linears[0].bias
    w1_p = mod_adam.linears[1].weight
    w_last_p = mod_adam.linears[-1].weight

    m_w0_pre = opt_adam.state[w0_p]["exp_avg"].clone()
    v_w0_pre = opt_adam.state[w0_p]["exp_avg_sq"].clone()
    m_b0_pre = opt_adam.state[b0_p]["exp_avg"].clone()
    v_b0_pre = opt_adam.state[b0_p]["exp_avg_sq"].clone()
    m_w1_pre = opt_adam.state[w1_p]["exp_avg"].clone()
    v_w1_pre = opt_adam.state[w1_p]["exp_avg_sq"].clone()
    m_wlast_pre = opt_adam.state[w_last_p]["exp_avg"].clone()
    v_wlast_pre = opt_adam.state[w_last_p]["exp_avg_sq"].clone()

    step_log_lams = partial_balance_step(
        mod_adam,
        p=cfg.balance_norm,
        eps=cfg.balance_eps,
        include_bias_in_cost=cfg.include_bias_in_cost,
        optimizer=opt_adam,
        optimizer_state_mode="covector",
    )
    lam0 = torch.exp(step_log_lams[0]).to(cfg.device)
    lam1 = torch.exp(step_log_lams[1]).to(cfg.device)
    lam_last = torch.exp(step_log_lams[-1]).to(cfg.device)

    s0_in = lam0.unsqueeze(1)
    s1_combined = lam1.unsqueeze(1) * (1.0 / lam0).unsqueeze(0)
    s_last_out = (1.0 / lam_last).unsqueeze(0)

    assert torch.allclose(opt_adam.state[w0_p]["exp_avg"], m_w0_pre / s0_in, atol=1e-6)
    assert torch.allclose(opt_adam.state[w0_p]["exp_avg_sq"], v_w0_pre / (s0_in ** 2), atol=1e-6)
    assert torch.allclose(opt_adam.state[b0_p]["exp_avg"], m_b0_pre / lam0, atol=1e-6)
    assert torch.allclose(opt_adam.state[b0_p]["exp_avg_sq"], v_b0_pre / (lam0 ** 2), atol=1e-6)
    assert torch.allclose(opt_adam.state[w1_p]["exp_avg"], m_w1_pre / s1_combined, atol=1e-6)
    assert torch.allclose(opt_adam.state[w1_p]["exp_avg_sq"], v_w1_pre / (s1_combined ** 2), atol=1e-6)
    assert torch.allclose(opt_adam.state[w_last_p]["exp_avg"], m_wlast_pre / s_last_out, atol=1e-6)
    assert torch.allclose(opt_adam.state[w_last_p]["exp_avg_sq"], v_wlast_pre / (s_last_out ** 2), atol=1e-6)
    print(
        "[Test 4 PASS] Optimizer-State Integration | Verified `partial_balance_step` "
        "covectorial moment scaling on incoming weights, biases, intermediate, and outgoing weights."
    )

    # Test 5: Trained ANN functional equivalence for BOTH non-dropout and dropout models
    opt_a = torch.optim.SGD(mod_a.parameters(), lr=0.01)
    opt_b = torch.optim.SGD(mod_b.parameters(), lr=0.01)
    mod_a.train()
    mod_b.train()
    for step_idx, (xb_tr, yb_tr) in enumerate(loader_a):
        if step_idx >= 20:
            break
        xb_tr, yb_tr = xb_tr.to(cfg.device), yb_tr.to(cfg.device)
        opt_a.zero_grad()
        loss_a = F.cross_entropy(mod_a(xb_tr)[0], yb_tr)
        loss_a.backward()
        opt_a.step()

        opt_b.zero_grad()
        loss_b = F.cross_entropy(mod_b(xb_tr, dropout_generator=rng_b.gen_dropout)[0], yb_tr)
        loss_b.backward()
        opt_b.step()

    with torch.no_grad():
        mod_a.eval()
        mod_b.eval()
        for label_str, tr_mod in [("No-Dropout Trained", mod_a), ("Dropout Trained", mod_b)]:
            mod_copy = copy.deepcopy(tr_mod)
            logits_pre, _ = mod_copy(x_val)
            sweeps, conv_flag, _ = full_balance_network(
                mod_copy,
                p=cfg.balance_norm,
                tol=1e-5,
                max_iters=cfg.full_balance_max_iters,
                eps=cfg.balance_eps,
                include_bias_in_cost=cfg.include_bias_in_cost,
            )
            logits_post, _ = mod_copy(x_val)
            max_d = float(torch.max(torch.abs(logits_pre - logits_post)).item())
            assert max_d < atol, f"{label_str} ANN functional invariance failed: {max_d}"
            print(
                f"[Test 5 PASS] Trained ANN Invariance ({label_str}) | "
                f"Sweeps: {sweeps} (Converged={conv_flag}) | Max logit diff: {max_d:.2e}"
            )

        # Test 6: Multi-horizon SNN readout convergence
        ann_logits, _ = mod_b(x_val)
        ann_preds = ann_logits.argmax(dim=1)
        horizons_test = [10, 25, 50, 100, 200, 500]

        for t_mode in ("fixed", "recalibrated"):
            snn_test = ConvertedSNN(
                ann=mod_b,
                threshold_mode=t_mode,
                input_encoding="analog_current",
                spike_mode=cfg.spike_mode,
                calib_loader=calib_loader,
                percentile=cfg.calib_percentile,
            ).to(cfg.device)

            metrics_by_t = {}
            for T_h in horizons_test:
                snn_rate_out, diag_h = snn_test.simulate_batch_readout(
                    x_val, T=T_h, return_diagnostics=True
                )
                per_ex_cos = F.cosine_similarity(ann_logits, snn_rate_out, dim=1)
                mae = float(torch.mean(torch.abs(ann_logits - snn_rate_out)).item())
                rel_l2 = float(
                    (
                        torch.norm(ann_logits - snn_rate_out, p=2, dim=1)
                        / (torch.norm(ann_logits, p=2, dim=1) + 1e-8)
                    ).mean().item()
                )
                argmax_agree = float((snn_rate_out.argmax(dim=1) == ann_preds).float().mean().item())
                metrics_by_t[T_h] = (
                    float(per_ex_cos.mean().item()),
                    float(torch.quantile(per_ex_cos, 0.10).item()),
                    mae,
                    rel_l2,
                    argmax_agree,
                    diag_h["post_reset_overflow_fraction"],
                )

            c_10, _, _, rl2_10, ag_10, _ = metrics_by_t[10]
            c_500, p10_500, mae_500, rl2_500, ag_500, ovf_500 = metrics_by_t[500]
            role_tag = "Confirmatory Gate" if t_mode == "recalibrated" else "Diagnostic"
            if t_mode == "recalibrated":
                assert c_500 > 0.98 and p10_500 > 0.95 and rl2_500 < 0.15 and ag_500 >= 0.90, (
                    f"Recalibrated SNN readout failed convergence tolerances: "
                    f"cos={c_500:.4f}, p10={p10_500:.4f}, relL2={rl2_500:.4f}, agree={ag_500:.3f}"
                )
            print(
                f"[Test 6a PASS] SNN Readout ({t_mode} [{role_tag}], {cfg.spike_mode}) | "
                f"T=10->500: MeanCos={c_10:.3f}->{c_500:.4f} (10thPct={p10_500:.3f}), "
                f"RelL2={rl2_10:.3f}->{rl2_500:.4f}, Agree={ag_10*100:.1f}%->{ag_500*100:.1f}%, "
                f"PostResetOverflow@500={ovf_500:.4f}"
            )

        snn_rate_1 = ConvertedSNN(
            ann=mod_b,
            threshold_mode="recalibrated",
            input_encoding="rate_spikes",
            spike_mode=cfg.spike_mode,
            calib_loader=calib_loader,
            percentile=cfg.calib_percentile,
        ).to(cfg.device)
        g_rate_1 = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device).gen_snn_input
        g_rate_2 = IsolatedRNGStreams(cfg.seeds[0], device=cfg.device).gen_snn_input
        r_out_1 = snn_rate_1.simulate_batch_readout(x_val, T=200, input_gen=g_rate_1)
        r_out_2 = snn_rate_1.simulate_batch_readout(x_val, T=200, input_gen=g_rate_2)
        assert torch.equal(r_out_1, r_out_2), "Rate-coded SNN RNG reset failed to reproduce identical output!"
        r_cos = float(F.cosine_similarity(ann_logits, r_out_1, dim=1).mean().item())
        assert r_cos > 0.90, f"Rate-coded SNN readout failed diagnostic alignment check: cos={r_cos:.4f}"
        print(
            f"[Test 6b PASS] Rate-Coded SNN Readout (Diagnostic) & RNG Reproducibility | "
            f"Exact RNG match verified | MeanCos@200={r_cos:.4f}"
        )

        # Test 7: Hand-calculated tiny-network test + >2^24 (20,000,001 events) int64 overflow test,
        # 1e-12 float64 zero-fraction check, dynamic early-stop SynOps check, dataset guard,
        # reversible 1/5/8 & fresh-seed toggles, 1-seed and zero-variance NaN guards,
        # primary vs. exploratory Holm family split (with duplicate ANN exclusion), and
        # positive + THREE negative-path unit tests for `confirmatory_interaction_efficiency_success`
        tiny_ann = FullyConnectedReLU((3, 2, 2), dropout_rate=0.0).to(cfg.device)
        tiny_ann.linears[0].weight.data = torch.tensor([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]], device=cfg.device)
        tiny_ann.linears[0].bias.data.zero_()
        tiny_ann.linears[1].weight.data = torch.tensor([[1.0, 0.0], [-0.5, 0.0]], device=cfg.device)
        tiny_ann.linears[1].bias.data.zero_()

        x_tiny = torch.tensor([[1.0, 0.0, 0.0]], device=cfg.device)
        y_tiny = torch.tensor([0], device=cfg.device)
        tiny_loader = DataLoader(TensorDataset(x_tiny, y_tiny), batch_size=1, shuffle=False)

        tiny_snn = ConvertedSNN(
            ann=tiny_ann,
            threshold_mode="fixed",
            input_encoding="analog_current",
            spike_mode="multi_spike",
            fixed_threshold=1.0,
        ).to(cfg.device)
        tiny_res = tiny_snn.evaluate_budgets(
            test_loader=tiny_loader,
            timesteps_budget=[2],
            ann_test_acc=1.0,
            stability_window_k=1,
        )[2]
        assert tiny_res["input_static_MACs_per_sample"] == 6.0
        assert tiny_res["input_eff_MACs_per_sample"] == 1.0
        assert np.isclose(
            tiny_res["analog_input_zero_fraction"], 2.0 / 3.0, rtol=0.0, atol=1e-12
        )
        assert tiny_res["hidden_event_count_synops_proxy_per_sample"] == 8.0
        # With k=1, sample stabilizes at t=1 (where 2 spikes x 2 fanout = 4.0 SynOps were emitted)
        assert tiny_res["windowed_early_stop_event_count_synops_per_sample"] == 4.0
        assert tiny_res["hidden_neurobench_eff_ACs_per_sample"] == 0.0
        assert tiny_res["hidden_neurobench_eff_MACs_per_sample"] == 4.0
        assert tiny_res["hidden_binary_active_step_ACs_per_sample"] == 4.0
        assert tiny_res["hidden_dense_SynOps_per_sample"] == 8.0
        assert np.isclose(
            tiny_res["multispike_or_saturation_step_fraction"], 0.5, rtol=0.0, atol=1e-12
        )

        # Permanent > 2^24 (16,777,216) exact int64 accumulation test (20,000,001 events across 1,000 samples)
        high_ann = FullyConnectedReLU((1, 1, 1), dropout_rate=0.0).to(cfg.device)
        high_ann.linears[0].weight.data.fill_(1.0)
        high_ann.linears[0].bias.data.zero_()
        high_ann.linears[1].weight.data.fill_(1.0)
        high_ann.linears[1].bias.data.zero_()
        x_high = torch.full((1000, 1), 20000.0, device=cfg.device)
        x_high[0, 0] = 20001.0
        y_high = torch.zeros(1000, dtype=torch.long, device=cfg.device)
        high_loader = DataLoader(TensorDataset(x_high, y_high), batch_size=1000, shuffle=False)
        high_snn = ConvertedSNN(
            ann=high_ann,
            threshold_mode="fixed",
            input_encoding="analog_current",
            spike_mode="multi_spike",
            fixed_threshold=1.0,
        ).to(cfg.device)
        high_res = high_snn.evaluate_budgets(
            test_loader=high_loader,
            timesteps_budget=[1],
            ann_test_acc=1.0,
            stability_window_k=1,
        )[1]
        assert high_res["hidden_event_count_synops_proxy_per_sample"] == 20000.001

        # Verify reversible 1-seed / fresh 5-seed / fresh 8-seed switching
        seed_cfg = ExperimentConfigV2(
            run_single_seed_smoke_first=True,
            use_fresh_confirmatory_seeds=True,
            use_main_8_seeds=True,
        )
        assert seed_cfg.get_active_seeds() == [42]
        seed_cfg.run_single_seed_smoke_first = False
        assert seed_cfg.get_active_seeds() == [43, 44, 45, 46, 47, 48, 49, 50]
        seed_cfg.use_main_8_seeds = False
        assert seed_cfg.get_active_seeds() == [43, 44, 45, 46, 47]
        seed_cfg.use_fresh_confirmatory_seeds = False
        assert seed_cfg.get_active_seeds() == [42, 43, 44, 45, 46]

        # Verify 1-seed and zero-variance NaN guards
        m1, sd1, se1, ci1, t1, p1 = _paired_summary_stats(np.array([42.0]))
        assert m1 == 42.0 and np.isnan(sd1) and np.isnan(ci1) and np.isnan(p1)
        mz, sdz, _, ciz, tz, pz = _paired_summary_stats(np.array([-5.0, -5.0, -5.0]))
        lcb_z = _paired_one_sided_lcb95(np.array([-0.002, -0.002, -0.002]))
        assert mz == -5.0 and sdz == 0.0 and np.isnan(ciz) and np.isnan(tz) and np.isnan(pz)
        assert np.isnan(lcb_z)

        # Verify full-family-size Holm-Bonferroni adjustment with NaNs (M = 3 -> [0.03, nan, 0.08])
        holm_test = _holm_bonferroni_adjust([0.01, np.nan, 0.04])
        assert (
            np.isclose(holm_test[0], 0.03)
            and np.isnan(holm_test[1])
            and np.isclose(holm_test[2], 0.08)
        )

        # Helper to build synthetic 5-seed factorial DataFrames for positive & negative-path gate tests
        gA_name = "Group_A_Base (Drop:Off, Bal:Off)"
        gB_name = "Group_B_DropoutOnly (Drop:On, Bal:Off)"
        gC_name = "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)"
        gD_name = "Group_D_Interaction (Drop:On, Bal:Interleaved)"

        def _build_synth_df(ops_fn, acc_fn) -> pd.DataFrame:
            rows_s = []
            for idx_s, s_val in enumerate([43, 44, 45, 46, 47]):
                for m_name in (gA_name, gB_name, gC_name, gD_name):
                    ops_v = ops_fn(m_name, idx_s)
                    acc_v = acc_fn(m_name, idx_s)
                    rows_s.append({
                        "method": m_name,
                        "seed": s_val,
                        "threshold_mode": "recalibrated",
                        "inference_steps": 50,
                        "ann_test_accuracy": acc_v,
                        "test_accuracy": acc_v,
                        "ann_to_snn_acc_gap": 0.0,
                        "windowed_early_stop_overall_policy_accuracy": acc_v,
                        "hidden_event_count_synops_proxy_per_sample": ops_v,
                        "windowed_early_stop_event_count_synops_per_sample": ops_v * 0.5,
                        "hidden_neurobench_eff_ACs_per_sample": ops_v,
                        "hidden_neurobench_eff_MACs_per_sample": 0.0,
                        "hidden_binary_active_step_ACs_per_sample": ops_v,
                        "hidden_active_spike_full_fanout_ops_per_sample": ops_v,
                        "total_ops_static_input_per_sample": ops_v + 100.0,
                        "total_eff_ops_static_input_per_sample": ops_v + 50.0,
                        "retrospective_convergence_steps": 10.0,
                        "windowed_stable_steps_censored": 10.0,
                    })
            return pd.DataFrame(rows_s)

        sub_deltas = [0.0011, 0.0014, 0.0009, 0.0013, 0.0012]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_f:
            tmp_csv_path = tmp_f.name
        try:
            test_cfg_int = copy.deepcopy(cfg)
            test_cfg_int.interaction_csv = tmp_csv_path

            # (1) Positive-Path Gate Test: interaction < 0 (p < 0.05), D - B < 0 (p < 0.05), acc non-inferior
            df_pos = _build_synth_df(
                ops_fn=lambda m, i: (
                    100.0 if m == gA_name else
                    90.0 if m == gB_name else
                    99.0 if m == gC_name else
                    89.0 - sub_deltas[i]
                ),
                acc_fn=lambda m, i: 0.9500 + (0.0002 * sub_deltas[i] if m == gD_name else 0.0),
            )
            rep_pos = compute_factorial_interaction_report(df_pos, test_cfg_int, verbose=False)
            # Full planned grid is 2 threshold_modes x 5 budgets x 14 metrics = 140 rows:
            # - 1 primary confirmatory endpoint (`primary_confirmatory_endpoint`)
            # - 9 duplicate ANN accuracy display rows (`duplicate_ann_display_only_excluded_from_holm`)
            # - 130 unique exploratory comparisons (`exploratory_holm_family`, M_exploratory = 130)
            assert len(rep_pos) == 140, f"Expected 140 planned rows, got {len(rep_pos)}"
            assert int((rep_pos["hypothesis_family_role"] == "primary_confirmatory_endpoint").sum()) == 1
            assert int((rep_pos["hypothesis_family_role"] == "exploratory_holm_family").sum()) == 130
            assert int(
                (rep_pos["hypothesis_family_role"] == "duplicate_ann_display_only_excluded_from_holm").sum()
            ) == 9
            # Verify missing-horizon placeholder rows (`fixed`, T=10 was not in `df_pos`) have n_seeds == 0
            missing_row = rep_pos[
                (rep_pos["threshold_mode"] == "fixed") & (rep_pos["inference_steps"] == 10)
            ].iloc[0]
            assert int(missing_row["n_seeds"]) == 0 and np.isnan(missing_row["Group_A_mean"])

            prim_pos = rep_pos[rep_pos["is_primary_confirmatory_outcome"] == True].iloc[0]
            assert prim_pos["interaction_DiD_sd"] > 0.0
            assert bool(prim_pos["confirmatory_interaction_efficiency_success"]) is True

            # (2) Negative-Path Test A: D - B is negative (-1.0, favorable direction!) but
            # statistically NON-SIGNIFICANT (p > 0.05) due to seed noise, while the DiD interaction
            # is strongly negative (-10.0, p < 0.0001) and accuracy non-inferiority passes -> MUST BE FALSE
            noisy_db = [-4.0, +2.0, -3.0, +1.5, -1.5]  # mean = -1.0, SD = 2.669, t = -0.838, p = 0.449 > 0.05
            did_tight = [-10.01, -9.99, -10.02, -9.98, -10.00]  # mean = -10.0, p < 1e-10
            df_neg_a = _build_synth_df(
                ops_fn=lambda m, i: (
                    100.0 if m == gA_name else
                    90.0 if m == gB_name else
                    100.0 + (noisy_db[i] - did_tight[i]) if m == gC_name else
                    90.0 + noisy_db[i]
                ),
                acc_fn=lambda m, i: 0.9500 + (0.001 * (i + 1) if m == gD_name else 0.0),
            )
            rep_neg_a = compute_factorial_interaction_report(df_neg_a, test_cfg_int, verbose=False)
            prim_neg_a = rep_neg_a[rep_neg_a["is_primary_confirmatory_outcome"] == True].iloc[0]
            assert prim_neg_a["interaction_DiD_mean"] < 0.0 and prim_neg_a["interaction_DiD_p_unadjusted"] < 0.001
            assert prim_neg_a["simple_balance_given_dropout_D_minus_B_mean"] < 0.0
            assert prim_neg_a["simple_balance_given_dropout_D_minus_B_p_unadjusted_exploratory"] > 0.05
            assert bool(prim_neg_a["practical_D_vs_B_efficiency_success"]) is False
            assert bool(prim_neg_a["confirmatory_interaction_efficiency_success"]) is False, (
                "Negative-Path Test A failed: confirmatory flag must be False when D vs B p > 0.05!"
            )

            # (3) Negative-Path Test B: Both interaction < 0 (p < 0.001) and D - B < 0 (p < 0.001) pass,
            # and D passes D-vs-B accuracy non-inferiority (+0.5%), but D FAILS D-vs-A accuracy
            # non-inferiority (-1.5% < -1.0% margin) -> practical_D_vs_B is True, confirmatory is FALSE
            df_neg_b = _build_synth_df(
                ops_fn=lambda m, i: (
                    100.0 if m == gA_name else
                    90.0 if m == gB_name else
                    99.0 if m == gC_name else
                    80.0 - sub_deltas[i]
                ),
                acc_fn=lambda m, i: (
                    0.9600 if m == gA_name else
                    0.9400 if m == gB_name else
                    0.9600 if m == gC_name else
                    0.9450 - 0.0005 * sub_deltas[i]  # D - B = +0.005 (passes), D - A = -0.015 (fails!)
                ),
            )
            rep_neg_b = compute_factorial_interaction_report(df_neg_b, test_cfg_int, verbose=False)
            prim_neg_b = rep_neg_b[rep_neg_b["is_primary_confirmatory_outcome"] == True].iloc[0]
            assert bool(prim_neg_b["passes_stat_acc_noninferiority_D_vs_B"]) is True
            assert bool(prim_neg_b["passes_stat_acc_noninferiority_D_vs_A"]) is False
            assert bool(prim_neg_b["practical_D_vs_B_efficiency_success"]) is True
            assert bool(prim_neg_b["confirmatory_interaction_efficiency_success"]) is False, (
                "Negative-Path Test B failed: confirmatory flag must be False when D-vs-A accuracy fails!"
            )

            # (4) Negative-Path Test C: Efficiency passes, but D fails D-vs-B accuracy non-inferiority -> FALSE
            df_neg_c = _build_synth_df(
                ops_fn=lambda m, i: (
                    100.0 if m == gA_name else
                    90.0 if m == gB_name else
                    99.0 if m == gC_name else
                    80.0 - sub_deltas[i]
                ),
                acc_fn=lambda m, i: (
                    0.9250 - 0.001 * sub_deltas[i] if m == gD_name else 0.9500
                ),
            )
            rep_neg_c = compute_factorial_interaction_report(df_neg_c, test_cfg_int, verbose=False)
            prim_neg_c = rep_neg_c[rep_neg_c["is_primary_confirmatory_outcome"] == True].iloc[0]
            assert bool(prim_neg_c["passes_stat_acc_noninferiority_D_vs_B"]) is False
            assert bool(prim_neg_c["confirmatory_interaction_efficiency_success"]) is False
        finally:
            if os.path.exists(tmp_csv_path):
                os.remove(tmp_csv_path)

        bad_cfg = copy.deepcopy(cfg)
        bad_cfg.dataset_name = "MNIST-DVS"
        bad_cfg.allow_synthetic_fallback = False
        rejected_properly = False
        try:
            build_datasets_and_eval_loaders(bad_cfg)
        except NotImplementedError:
            rejected_properly = True
        assert rejected_properly, "Failed to reject unsupported dataset when allow_synthetic_fallback=False!"
        print(
            "[Test 7 PASS] >2^24 Int64 Test, Dynamic Stopped SynOps, 130-Comparison Exploratory Holm Family, "
            "Positive & 3 Negative-Path Confirmatory Gate Tests | All verified."
        )
    print("=====================================================================\n")


# =====================================================================
# 6. Phase 3 Training Pipeline & Sustained Rolling-Window Plateau Check
# =====================================================================

@torch.no_grad()
def evaluate_ann(model: FullyConnectedReLU, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)
    return float(correct / max(1, total))


def compute_fanout_weighted_synop_penalty(
    hidden_acts: List[torch.Tensor], fanouts: List[int]
) -> torch.Tensor:
    """
    Computes batch-averaged, hidden-summed fan-out-weighted L1 activation penalty
    (a Sorbaro-inspired activation-sparsity proxy, not a reproduction of QReLU/Sinabs).
    """
    synop_pen = torch.tensor(0.0, device=hidden_acts[0].device)
    for act, fanout in zip(hidden_acts, fanouts):
        layer_activity = act.abs().sum(dim=1).mean(dim=0)
        synop_pen = synop_pen + float(fanout) * layer_activity
    return synop_pen


def train_and_prepare_condition(
    condition_name: str,
    dropout_on: bool,
    balance_schedule: str,
    seed: int,
    cfg: ExperimentConfigV2,
    train_dataset: torch.utils.data.Dataset,
    val_loader: Optional[DataLoader] = None,
    synoploss_alpha_override: Optional[float] = None,
    reference_cum_log_lambdas: Optional[List[torch.Tensor]] = None,
    reference_epoch_log_lambdas: Optional[List[List[torch.Tensor]]] = None,
) -> Tuple[
    FullyConnectedReLU,
    float,
    float,
    float,
    int,
    int,
    Optional[bool],
    List[torch.Tensor],
    List[List[torch.Tensor]],
    Dict[str, object],
]:
    """
    Trains a single condition with strictly isolated RNG streams and evaluates
    a sustained rolling-window validation plateau criterion over the final
    `cfg.val_plateau_window` epochs (with `+ 1e-12` numerical tolerance).
    """
    set_global_seed(seed)
    rng = IsolatedRNGStreams(seed, device=cfg.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=rng.gen_loader,
    )

    p_drop = cfg.dropout_rate if dropout_on else 0.0
    model = FullyConnectedReLU(
        cfg.layer_sizes, dropout_rate=p_drop, init_generator=rng.gen_init
    ).to(cfg.device)

    if cfg.optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.learning_rate, momentum=cfg.sgd_momentum
        )
    elif cfg.optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer_name: {cfg.optimizer_name}")

    train_epoch_loop_wall_time_sec = 0.0
    balance_overhead_sec = 0.0
    val_eval_sec = 0.0
    balance_sweeps_used = 0
    random_rescale_sweeps_used = 0
    balance_converged: Optional[bool] = None

    cum_log_lambdas = [
        torch.zeros(cfg.layer_sizes[l + 1], dtype=torch.float32)
        for l in range(len(cfg.layer_sizes) - 2)
    ]
    epoch_log_lambdas_history: List[List[torch.Tensor]] = []

    if balance_schedule == "pre_training":
        _sync_device(cfg.device)
        t_bal = time.perf_counter()
        sweeps, conv_flag, pre_log_lam = full_balance_network(
            model,
            p=cfg.balance_norm,
            tol=cfg.full_balance_tol,
            max_iters=cfg.full_balance_max_iters,
            eps=cfg.balance_eps,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        balance_sweeps_used += sweeps
        balance_converged = conv_flag
        for l in range(len(cum_log_lambdas)):
            cum_log_lambdas[l] += pre_log_lam[l]
        _sync_device(cfg.device)
        balance_overhead_sec += time.perf_counter() - t_bal

    alpha_pen = (
        synoploss_alpha_override
        if synoploss_alpha_override is not None
        else cfg.synoploss_alpha
    )

    epoch_train_losses: List[float] = []
    epoch_val_accs: List[float] = []
    per_epoch_records: List[Dict[str, object]] = []

    for epoch_idx in range(cfg.epochs):
        model.train()
        ep_loss_sum_tensor = torch.zeros((), dtype=torch.float64, device=cfg.device)
        ep_samples = 0

        _sync_device(cfg.device)
        t_ep_loop = time.perf_counter()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            need_acts = balance_schedule == "fanout_l1_synop_proxy"
            logits, hidden_acts = model(
                xb,
                return_activations=need_acts,
                dropout_generator=rng.gen_dropout,
            )
            loss = F.cross_entropy(logits, yb)

            if need_acts and hidden_acts is not None and alpha_pen > 0.0:
                synop_pen = compute_fanout_weighted_synop_penalty(
                    hidden_acts, model.fanouts
                )
                loss = loss + alpha_pen * synop_pen

            loss.backward()
            optimizer.step()
            ep_loss_sum_tensor += loss.detach().double() * xb.size(0)
            ep_samples += xb.size(0)
        _sync_device(cfg.device)
        ep_loop_sec = time.perf_counter() - t_ep_loop
        train_epoch_loop_wall_time_sec += ep_loop_sec

        ep_train_loss = float(ep_loss_sum_tensor.item()) / max(1, ep_samples)
        epoch_train_losses.append(ep_train_loss)

        ep_bal_sec = 0.0
        if balance_schedule == "interleaved":
            _sync_device(cfg.device)
            t_bal = time.perf_counter()
            step_log_lam = partial_balance_step(
                model,
                p=cfg.balance_norm,
                eps=cfg.balance_eps,
                include_bias_in_cost=cfg.include_bias_in_cost,
                optimizer=optimizer,
                optimizer_state_mode=cfg.optimizer_state_mode,
            )
            balance_sweeps_used += 1
            epoch_log_lambdas_history.append([t.clone() for t in step_log_lam])
            for l in range(len(cum_log_lambdas)):
                cum_log_lambdas[l] += step_log_lam[l]
            _sync_device(cfg.device)
            ep_bal_sec = time.perf_counter() - t_bal
            balance_overhead_sec += ep_bal_sec

        elif balance_schedule == "interleaved_random_rescaling":
            _sync_device(cfg.device)
            t_bal = time.perf_counter()
            ref_ep = (
                reference_epoch_log_lambdas[epoch_idx]
                if (
                    reference_epoch_log_lambdas is not None
                    and epoch_idx < len(reference_epoch_log_lambdas)
                )
                else None
            )
            step_rand_log_lam = matched_random_function_preserving_rescale(
                model,
                gen=rng.gen_rescale,
                reference_log_lambdas=ref_ep,
                optimizer=optimizer,
                optimizer_state_mode=cfg.optimizer_state_mode,
            )
            random_rescale_sweeps_used += 1
            epoch_log_lambdas_history.append([t.clone() for t in step_rand_log_lam])
            for l in range(len(cum_log_lambdas)):
                cum_log_lambdas[l] += step_rand_log_lam[l]
            _sync_device(cfg.device)
            ep_bal_sec = time.perf_counter() - t_bal
            balance_overhead_sec += ep_bal_sec

        ep_val_acc = float("nan")
        if val_loader is not None:
            _sync_device(cfg.device)
            t_val = time.perf_counter()
            ep_val_acc = evaluate_ann(model, val_loader, cfg.device)
            _sync_device(cfg.device)
            val_eval_sec += time.perf_counter() - t_val
            epoch_val_accs.append(ep_val_acc)

        per_epoch_records.append({
            "method": condition_name,
            "seed": seed,
            "epoch": epoch_idx + 1,
            "train_loss": float(ep_train_loss),
            "val_accuracy": _float_or_nan(ep_val_acc),
            "train_epoch_loop_wall_time_sec": float(ep_loop_sec),
            "epoch_balance_or_rescale_sec": float(ep_bal_sec),
        })

    if balance_schedule == "post_training":
        _sync_device(cfg.device)
        t_bal = time.perf_counter()
        sweeps, conv_flag, post_log_lam = full_balance_network(
            model,
            p=cfg.balance_norm,
            tol=cfg.full_balance_tol,
            max_iters=cfg.full_balance_max_iters,
            eps=cfg.balance_eps,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        balance_sweeps_used += sweeps
        balance_converged = conv_flag
        for l in range(len(cum_log_lambdas)):
            cum_log_lambdas[l] += post_log_lam[l]
        _sync_device(cfg.device)
        balance_overhead_sec += time.perf_counter() - t_bal

    elif balance_schedule == "post_random_rescaling":
        _sync_device(cfg.device)
        t_bal = time.perf_counter()
        rand_log_lam = matched_random_function_preserving_rescale(
            model,
            gen=rng.gen_rescale,
            reference_log_lambdas=reference_cum_log_lambdas,
        )
        random_rescale_sweeps_used += 1
        for l in range(len(cum_log_lambdas)):
            cum_log_lambdas[l] += rand_log_lam[l]
        _sync_device(cfg.device)
        balance_overhead_sec += time.perf_counter() - t_bal

    if balance_schedule == "interleaved":
        final_bal_check = compute_balance_metrics(
            model,
            p=cfg.balance_norm,
            eps=cfg.balance_eps,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        balance_converged = bool(
            final_bal_check["max_ratio_deviation_eligible"] <= cfg.full_balance_tol
        )

    has_multi_epoch_val = len(epoch_val_accs) >= 2
    last_delta = (
        epoch_val_accs[-1] - epoch_val_accs[-2]
        if has_multi_epoch_val
        else float("nan")
    )

    # Sustained rolling-window plateau check with `+ 1e-12` float tolerance
    w_len = max(2, cfg.val_plateau_window)
    if len(epoch_val_accs) >= w_len:
        win_vals = epoch_val_accs[-w_len:]
        win_span = float(max(win_vals) - min(win_vals))
        win_net_gain = float(win_vals[-1] - win_vals[0])
        plateau_reached = bool(win_span <= (cfg.val_plateau_delta_tol + 1e-12))
    else:
        win_span = float("nan")
        win_net_gain = float("nan")
        plateau_reached = False

    lc_stats: Dict[str, object] = {
        "epoch_1_train_loss": epoch_train_losses[0] if epoch_train_losses else float("nan"),
        "epoch_final_train_loss": epoch_train_losses[-1] if epoch_train_losses else float("nan"),
        "epoch_1_val_acc": epoch_val_accs[0] if epoch_val_accs else float("nan"),
        "epoch_mid_val_acc": (
            epoch_val_accs[len(epoch_val_accs) // 2] if epoch_val_accs else float("nan")
        ),
        "epoch_final_val_acc": epoch_val_accs[-1] if epoch_val_accs else float("nan"),
        "val_acc_last_2_epoch_delta": last_delta,
        "val_acc_window_max_min_span": win_span,
        "val_acc_window_net_gain": win_net_gain,
        "val_acc_plateau_reached": plateau_reached,
        "train_loss_history": [float(round(x, 6)) for x in epoch_train_losses],
        "val_acc_history": [float(round(x, 4)) for x in epoch_val_accs],
        "per_epoch_records": per_epoch_records,
    }

    return (
        model,
        train_epoch_loop_wall_time_sec,
        balance_overhead_sec,
        val_eval_sec,
        balance_sweeps_used,
        random_rescale_sweeps_used,
        balance_converged,
        cum_log_lambdas,
        epoch_log_lambdas_history,
        lc_stats,
    )


@torch.no_grad()
def _compute_validation_synop_activity(
    model: FullyConnectedReLU, val_loader: DataLoader, device: str
) -> float:
    """Computes sample-weighted mean validation fan-out L1 activity."""
    model.eval()
    weighted_pen_sum = 0.0
    total_samples = 0
    for xb, _ in val_loader:
        xb = xb.to(device)
        bsz = xb.size(0)
        _, h_acts = model(xb, return_activations=True)
        batch_pen = compute_fanout_weighted_synop_penalty(h_acts, model.fanouts).item()
        weighted_pen_sum += batch_pen * bsz
        total_samples += bsz
    return float(weighted_pen_sum / max(1, total_samples))


def tune_synoploss_alpha_on_validation(
    cfg: ExperimentConfigV2,
    train_dataset: torch.utils.data.Dataset,
    val_loader: DataLoader,
) -> Tuple[float, str]:
    """
    Selects `synoploss_alpha` strictly on the validation split.
    If no positive candidate satisfies both the accuracy tolerance and activity reduction,
    returns `(0.0, "no_qualifying_candidate_fallback_0")`.
    """
    if not cfg.tune_synoploss_on_val:
        return cfg.synoploss_alpha, "tuning_disabled"

    pilot_seed = cfg.exploratory_pilot_seed
    ref_model, _, _, _, _, _, _, _, _, _ = train_and_prepare_condition(
        condition_name="ValTune_Ref_Dropout",
        dropout_on=True,
        balance_schedule="none",
        seed=pilot_seed,
        cfg=cfg,
        train_dataset=train_dataset,
    )
    ref_val_acc = evaluate_ann(ref_model, val_loader, cfg.device)
    ref_mean_pen = _compute_validation_synop_activity(ref_model, val_loader, cfg.device)

    best_alpha: Optional[float] = None
    lowest_qualifying_pen = ref_mean_pen

    for candidate_alpha in cfg.synoploss_alpha_grid:
        if candidate_alpha <= 0.0:
            continue
        cand_model, _, _, _, _, _, _, _, _, _ = train_and_prepare_condition(
            condition_name=f"ValTune_Alpha_{candidate_alpha}",
            dropout_on=True,
            balance_schedule="fanout_l1_synop_proxy",
            seed=pilot_seed,
            cfg=cfg,
            train_dataset=train_dataset,
            synoploss_alpha_override=candidate_alpha,
        )
        val_acc = evaluate_ann(cand_model, val_loader, cfg.device)
        mean_pen = _compute_validation_synop_activity(cand_model, val_loader, cfg.device)

        if (
            val_acc >= ref_val_acc - cfg.synoploss_val_acc_tolerance
            and mean_pen < lowest_qualifying_pen
        ):
            lowest_qualifying_pen = mean_pen
            best_alpha = candidate_alpha

    if best_alpha is None:
        print(
            f"[Validation Tuning Warning] No candidate alpha reduced activity below "
            f"reference ({ref_mean_pen:.2f}) within {cfg.synoploss_val_acc_tolerance*100:.1f}% "
            f"of Ref Val Acc ({ref_val_acc:.4f}). Returning 0.0 (unregularized fallback)."
        )
        return 0.0, "no_qualifying_candidate_fallback_0"

    print(
        f"[Validation Tuning] Selected synoploss_alpha = {best_alpha:.1e} "
        f"(Val Activity: {ref_mean_pen:.2f} -> {lowest_qualifying_pen:.2f}, "
        f"Ref Val Acc = {ref_val_acc:.4f})"
    )
    return best_alpha, "tuned_qualified"


# =====================================================================
# 7. Data Loading with Strict Provenance Enforcement
# =====================================================================

def build_datasets_and_eval_loaders(
    cfg: ExperimentConfigV2,
) -> Tuple[torch.utils.data.Dataset, DataLoader, DataLoader, DataLoader, str, str]:
    """
    Enforces strict dataset provenance:
      - If `cfg.dataset_name` is `"MNIST"`, loads MNIST or raises `RuntimeError` unless
        `cfg.allow_synthetic_fallback=True`.
      - If `cfg.dataset_name` is `"SYNTHETIC"`, requires `cfg.allow_synthetic_fallback=True`.
      - Any other dataset name immediately raises `NotImplementedError`.
    """
    set_global_seed(cfg.exploratory_pilot_seed)
    split_gen = torch.Generator(device="cpu").manual_seed(cfg.exploratory_pilot_seed + 999)
    ds_upper = cfg.dataset_name.upper()

    if ds_upper not in ("MNIST", "SYNTHETIC"):
        raise NotImplementedError(
            f"Dataset '{cfg.dataset_name}' is not implemented in this harness. "
            f"Supported dataset is 'MNIST' (or 'SYNTHETIC' with allow_synthetic_fallback=True)."
        )

    if ds_upper == "SYNTHETIC" and not cfg.allow_synthetic_fallback:
        raise RuntimeError(
            "dataset_name='SYNTHETIC' requires explicit opt-in via allow_synthetic_fallback=True."
        )

    if ds_upper == "MNIST":
        if TORCHVISION_AVAILABLE:
            transform = transforms.Compose([transforms.ToTensor()])
            try:
                full_train = datasets.MNIST(
                    root=cfg.data_dir, train=True, download=True, transform=transform
                )
                test_set = datasets.MNIST(
                    root=cfg.data_dir, train=False, download=True, transform=transform
                )
                train_set, val_set, calib_set = random_split(
                    full_train, [50000, 5000, 5000], generator=split_gen
                )
                return (
                    train_set,
                    DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False),
                    DataLoader(calib_set, batch_size=cfg.batch_size, shuffle=False),
                    DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False),
                    "MNIST",
                    "train:50000|val:5000|calib:5000|test:10000",
                )
            except Exception as e:
                if not cfg.allow_synthetic_fallback:
                    raise RuntimeError(
                        f"MNIST download/load failed ({e}) and `allow_synthetic_fallback=False`. "
                        f"Set `allow_synthetic_fallback=True` only for offline smoke tests."
                    ) from e
                print(
                    f"[Provenance Notice] MNIST unavailable ({e}); "
                    f"using explicit opt-in Synthetic_ReLU_Teacher_Fallback."
                )
        else:
            if not cfg.allow_synthetic_fallback:
                raise RuntimeError(
                    "torchvision is unavailable for MNIST and `allow_synthetic_fallback=False`."
                )

    N_total = 3200
    X = torch.rand(N_total, cfg.layer_sizes[0], generator=split_gen)
    W1_true = torch.randn(cfg.layer_sizes[0], 64, generator=split_gen) / math.sqrt(cfg.layer_sizes[0])
    W2_true = torch.randn(64, cfg.layer_sizes[-1], generator=split_gen) / math.sqrt(64)
    logits_true = F.relu(X @ W1_true) @ W2_true
    y = logits_true.argmax(dim=1)
    dataset = TensorDataset(X, y)
    train_set, val_set, calib_set, test_set = random_split(
        dataset, [2000, 400, 400, 400], generator=split_gen
    )
    return (
        train_set,
        DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(calib_set, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False),
        "Synthetic_ReLU_Teacher_Fallback",
        "train:2000|val:400|calib:400|test:400",
    )


# =====================================================================
# 8. Statistical Analysis: Primary vs. Exploratory Holm Family & Deduplication
# =====================================================================

def _paired_summary_stats(vec: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Returns (mean, sd, se, ci95_halfwidth, t_stat, p_val_two_sided) computed in full
    float64 precision.
    Safeguards:
      - When `n < 2` (1-seed smoke test), returns `NaN` for all inferential quantities.
      - When `n >= 2` and `se <= 1e-15` (degenerate zero sample variance across seeds),
        returns `sd = 0.0, se = 0.0`, and `NaN` for `(ci95, t_stat, p_val)` so zero-variance
        discretization artifacts are never treated as infinite t-statistics with p = 0.0.
    """
    n = len(vec)
    if n == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    mean = float(np.mean(vec))
    if n < 2:
        return (
            mean,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    sd = float(np.std(vec, ddof=1))
    se = sd / math.sqrt(n)
    if se <= 1e-15:
        return (
            mean,
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
        )
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci95 = t_crit * se
    t_stat = mean / se
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return mean, sd, se, ci95, t_stat, p_val


def _paired_one_sided_lcb95(diff_vec: np.ndarray) -> float:
    """
    Computes the one-sided 95% lower confidence bound LCB_95%(d_bar) = d_bar - t_{0.95, n-1} * SE
    for a paired accuracy non-inferiority test (H0: mu_1 - mu_0 <= -margin vs H1: mu_1 - mu_0 > -margin).
    Returns `NaN` when `n < 2` or when sample standard error `se <= 1e-15` (degenerate zero variance).
    """
    n = len(diff_vec)
    if n < 2:
        return float("nan")
    mean = float(np.mean(diff_vec))
    sd = float(np.std(diff_vec, ddof=1))
    se = sd / math.sqrt(n)
    if se <= 1e-15:
        return float("nan")
    t_crit_one_sided = float(stats.t.ppf(0.95, df=n - 1))
    return float(mean - t_crit_one_sided * se)


def _holm_bonferroni_adjust(
    p_values: List[float], planned_family_size: Optional[int] = None
) -> List[float]:
    """
    Applies Holm's step-down procedure across the FULL pre-specified family size
    `M = max(len(p_values), planned_family_size or len(p_values))`, treating unestimable
    (`NaN`) or omitted comparisons as non-rejections placed at the end of the step-down
    order while preserving `np.nan` in their output positions.
    """
    n_list = len(p_values)
    if n_list == 0:
        return []
    m_total = max(n_list, int(planned_family_size)) if planned_family_size is not None else n_list
    finite_items = [
        (idx, float(p)) for idx, p in enumerate(p_values) if (p is not None and np.isfinite(p))
    ]
    if len(finite_items) == 0:
        return [np.nan] * n_list

    indexed_sorted = sorted(finite_items, key=lambda x: x[1])
    adjusted = [np.nan] * n_list
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_sorted):
        adj_p = min(1.0, p_val * (m_total - rank))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = float(running_max)
    return adjusted


def compute_factorial_interaction_report(
    df: pd.DataFrame, cfg: ExperimentConfigV2, verbose: bool = True
) -> pd.DataFrame:
    """
    Analyzes the 2x2 factorial design (Groups A, B, C, D) across the complete planned
    grid (`2 threshold_modes x len(cfg.timesteps_budget) x 14 metrics`), explicitly
    separating:
      1. `primary_confirmatory_endpoint` (1 pre-specified primary outcome, tested at alpha=0.05)
      2. `exploratory_holm_family` (the unique secondary exploratory comparisons: all SNN
         comparisons excluding the primary cell, plus 1 canonical `ann_test_accuracy` comparison)
      3. `duplicate_ann_display_only_excluded_from_holm` (the remaining 9 repeated
         `ann_test_accuracy` rows across other horizons/thresholds, retained for table display
         but excluded from inflating the Holm family size).
    """
    metrics_to_analyze = [
        "ann_test_accuracy",
        "test_accuracy",
        "ann_to_snn_acc_gap",
        "windowed_early_stop_overall_policy_accuracy",
        "hidden_event_count_synops_proxy_per_sample",
        "windowed_early_stop_event_count_synops_per_sample",
        "hidden_neurobench_eff_ACs_per_sample",
        "hidden_neurobench_eff_MACs_per_sample",
        "hidden_binary_active_step_ACs_per_sample",
        "hidden_active_spike_full_fanout_ops_per_sample",
        "total_ops_static_input_per_sample",
        "total_eff_ops_static_input_per_sample",
        "retrospective_convergence_steps",
        "windowed_stable_steps_censored",
    ]
    lower_is_better_ops_metrics = {
        "hidden_event_count_synops_proxy_per_sample",
        "windowed_early_stop_event_count_synops_per_sample",
        "hidden_neurobench_eff_ACs_per_sample",
        "hidden_neurobench_eff_MACs_per_sample",
        "hidden_binary_active_step_ACs_per_sample",
        "hidden_active_spike_full_fanout_ops_per_sample",
        "total_ops_static_input_per_sample",
        "total_eff_ops_static_input_per_sample",
    }

    gA = "Group_A_Base (Drop:Off, Bal:Off)"
    gB = "Group_B_DropoutOnly (Drop:On, Bal:Off)"
    gC = "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)"
    gD = "Group_D_Interaction (Drop:On, Bal:Interleaved)"

    planned_threshold_modes = ["fixed", "recalibrated"]
    planned_timesteps = list(cfg.timesteps_budget)
    n_horizon_cells = len(planned_threshold_modes) * len(planned_timesteps)
    # 13 SNN metrics across `n_horizon_cells`, minus 1 primary endpoint, plus 1 global ANN accuracy test
    n_snn_metrics = len(metrics_to_analyze) - 1
    exploratory_holm_family_size = (n_horizon_cells * n_snn_metrics) - 1 + 1

    rows = []
    for thresh_mode in planned_threshold_modes:
        for T in planned_timesteps:
            grp_df = df[
                (df["threshold_mode"] == thresh_mode) & (df["inference_steps"] == int(T))
            ]
            has_all_groups = False
            if len(grp_df) > 0:
                piv = grp_df.pivot(index="seed", columns="method")
                if all(m in piv.columns.levels[1] for m in [gA, gB, gC, gD]):
                    has_all_groups = True

            for metric in metrics_to_analyze:
                is_primary = (
                    metric == cfg.primary_confirmatory_metric
                    and thresh_mode == cfg.primary_confirmatory_threshold_mode
                    and int(T) == cfg.primary_confirmatory_timestep
                )
                is_canonical_ann_row = (
                    metric == "ann_test_accuracy"
                    and thresh_mode == cfg.primary_confirmatory_threshold_mode
                    and int(T) == cfg.primary_confirmatory_timestep
                )

                if is_primary:
                    family_role = "primary_confirmatory_endpoint"
                    row_family_size = 1
                elif metric == "ann_test_accuracy" and not is_canonical_ann_row:
                    family_role = "duplicate_ann_display_only_excluded_from_holm"
                    row_family_size = 0
                else:
                    family_role = "exploratory_holm_family"
                    row_family_size = exploratory_holm_family_size

                if (
                    not has_all_groups
                    or "test_accuracy" not in piv.columns.levels[0]
                    or metric not in piv.columns.levels[0]
                ):
                    rows.append({
                        "threshold_mode": thresh_mode,
                        "inference_steps": int(T),
                        "metric": metric,
                        "is_primary_confirmatory_outcome": is_primary,
                        "hypothesis_family_role": family_role,
                        "planned_family_size": row_family_size,
                        "acc_D_minus_A_mean": np.nan,
                        "acc_D_minus_A_lcb95_one_sided": np.nan,
                        "passes_point_acc_noninferiority_D_vs_A": False,
                        "passes_stat_acc_noninferiority_D_vs_A": False,
                        "acc_D_minus_B_mean": np.nan,
                        "acc_D_minus_B_lcb95_one_sided": np.nan,
                        "passes_point_acc_noninferiority_D_vs_B": False,
                        "passes_stat_acc_noninferiority_D_vs_B": False,
                        "degenerate_zero_var_interaction": False,
                        "degenerate_zero_var_D_minus_B": False,
                        "practical_D_vs_A_efficiency_success": False,
                        "practical_D_vs_B_efficiency_success": False,
                        "confirmatory_interaction_efficiency_success": False,
                        "n_seeds": 0,
                        "common_seeds_used": "[]",
                        "Group_A_mean": np.nan,
                        "Group_A_ci95_halfwidth": np.nan,
                        "Group_B_mean": np.nan,
                        "Group_B_ci95_halfwidth": np.nan,
                        "Group_C_mean": np.nan,
                        "Group_C_ci95_halfwidth": np.nan,
                        "Group_D_mean": np.nan,
                        "Group_D_ci95_halfwidth": np.nan,
                        "factorial_main_dropout_mean": np.nan,
                        "factorial_main_dropout_ci95_halfwidth": np.nan,
                        "factorial_main_dropout_p_unadjusted_exploratory": np.nan,
                        "factorial_main_balance_mean": np.nan,
                        "factorial_main_balance_ci95_halfwidth": np.nan,
                        "factorial_main_balance_p_unadjusted_exploratory": np.nan,
                        "simple_dropout_unbalanced_B_minus_A_mean": np.nan,
                        "simple_dropout_unbalanced_B_minus_A_ci95_halfwidth": np.nan,
                        "simple_dropout_unbalanced_B_minus_A_p_unadjusted_exploratory": np.nan,
                        "simple_dropout_given_balance_D_minus_C_mean": np.nan,
                        "simple_dropout_given_balance_D_minus_C_ci95_halfwidth": np.nan,
                        "simple_dropout_given_balance_D_minus_C_p_unadjusted_exploratory": np.nan,
                        "simple_balance_nodropout_C_minus_A_mean": np.nan,
                        "simple_balance_nodropout_C_minus_A_ci95_halfwidth": np.nan,
                        "simple_balance_nodropout_C_minus_A_p_unadjusted_exploratory": np.nan,
                        "simple_balance_given_dropout_D_minus_B_mean": np.nan,
                        "simple_balance_given_dropout_D_minus_B_sd": np.nan,
                        "simple_balance_given_dropout_D_minus_B_ci95_halfwidth": np.nan,
                        "simple_balance_given_dropout_D_minus_B_ci95_lower": np.nan,
                        "simple_balance_given_dropout_D_minus_B_ci95_upper": np.nan,
                        "simple_balance_given_dropout_D_minus_B_p_unadjusted_exploratory": np.nan,
                        "combined_D_minus_A_mean": np.nan,
                        "combined_D_minus_A_ci95_halfwidth": np.nan,
                        "combined_D_minus_A_p_unadjusted_exploratory": np.nan,
                        "interaction_DiD_mean": np.nan,
                        "interaction_DiD_sd": np.nan,
                        "interaction_DiD_se": np.nan,
                        "interaction_DiD_ci95_halfwidth": np.nan,
                        "interaction_DiD_ci95_lower": np.nan,
                        "interaction_DiD_ci95_upper": np.nan,
                        "interaction_DiD_t_stat": np.nan,
                        "interaction_DiD_p_unadjusted": np.nan,
                        "per_seed_D_minus_B_diffs": "[]",
                        "per_seed_D_minus_A_diffs": "[]",
                        "per_seed_interaction_DiD_diffs": "[]",
                    })
                    continue

                acc_complete_df = piv["test_accuracy"].dropna(subset=[gA, gB, gC, gD])
                metric_complete_df = piv[metric].dropna(subset=[gA, gB, gC, gD])
                common_seeds = acc_complete_df.index.intersection(metric_complete_df.index)

                if len(common_seeds) == 0:
                    rows.append({
                        "threshold_mode": thresh_mode,
                        "inference_steps": int(T),
                        "metric": metric,
                        "is_primary_confirmatory_outcome": is_primary,
                        "hypothesis_family_role": family_role,
                        "planned_family_size": row_family_size,
                        "acc_D_minus_A_mean": np.nan,
                        "acc_D_minus_A_lcb95_one_sided": np.nan,
                        "passes_point_acc_noninferiority_D_vs_A": False,
                        "passes_stat_acc_noninferiority_D_vs_A": False,
                        "acc_D_minus_B_mean": np.nan,
                        "acc_D_minus_B_lcb95_one_sided": np.nan,
                        "passes_point_acc_noninferiority_D_vs_B": False,
                        "passes_stat_acc_noninferiority_D_vs_B": False,
                        "degenerate_zero_var_interaction": False,
                        "degenerate_zero_var_D_minus_B": False,
                        "practical_D_vs_A_efficiency_success": False,
                        "practical_D_vs_B_efficiency_success": False,
                        "confirmatory_interaction_efficiency_success": False,
                        "n_seeds": 0,
                        "common_seeds_used": "[]",
                        "Group_A_mean": np.nan,
                        "Group_A_ci95_halfwidth": np.nan,
                        "Group_B_mean": np.nan,
                        "Group_B_ci95_halfwidth": np.nan,
                        "Group_C_mean": np.nan,
                        "Group_C_ci95_halfwidth": np.nan,
                        "Group_D_mean": np.nan,
                        "Group_D_ci95_halfwidth": np.nan,
                        "factorial_main_dropout_mean": np.nan,
                        "factorial_main_dropout_ci95_halfwidth": np.nan,
                        "factorial_main_dropout_p_unadjusted_exploratory": np.nan,
                        "factorial_main_balance_mean": np.nan,
                        "factorial_main_balance_ci95_halfwidth": np.nan,
                        "factorial_main_balance_p_unadjusted_exploratory": np.nan,
                        "simple_dropout_unbalanced_B_minus_A_mean": np.nan,
                        "simple_dropout_unbalanced_B_minus_A_ci95_halfwidth": np.nan,
                        "simple_dropout_unbalanced_B_minus_A_p_unadjusted_exploratory": np.nan,
                        "simple_dropout_given_balance_D_minus_C_mean": np.nan,
                        "simple_dropout_given_balance_D_minus_C_ci95_halfwidth": np.nan,
                        "simple_dropout_given_balance_D_minus_C_p_unadjusted_exploratory": np.nan,
                        "simple_balance_nodropout_C_minus_A_mean": np.nan,
                        "simple_balance_nodropout_C_minus_A_ci95_halfwidth": np.nan,
                        "simple_balance_nodropout_C_minus_A_p_unadjusted_exploratory": np.nan,
                        "simple_balance_given_dropout_D_minus_B_mean": np.nan,
                        "simple_balance_given_dropout_D_minus_B_sd": np.nan,
                        "simple_balance_given_dropout_D_minus_B_ci95_halfwidth": np.nan,
                        "simple_balance_given_dropout_D_minus_B_ci95_lower": np.nan,
                        "simple_balance_given_dropout_D_minus_B_ci95_upper": np.nan,
                        "simple_balance_given_dropout_D_minus_B_p_unadjusted_exploratory": np.nan,
                        "combined_D_minus_A_mean": np.nan,
                        "combined_D_minus_A_ci95_halfwidth": np.nan,
                        "combined_D_minus_A_p_unadjusted_exploratory": np.nan,
                        "interaction_DiD_mean": np.nan,
                        "interaction_DiD_sd": np.nan,
                        "interaction_DiD_se": np.nan,
                        "interaction_DiD_ci95_halfwidth": np.nan,
                        "interaction_DiD_ci95_lower": np.nan,
                        "interaction_DiD_ci95_upper": np.nan,
                        "interaction_DiD_t_stat": np.nan,
                        "interaction_DiD_p_unadjusted": np.nan,
                        "per_seed_D_minus_B_diffs": "[]",
                        "per_seed_D_minus_A_diffs": "[]",
                        "per_seed_interaction_DiD_diffs": "[]",
                    })
                    continue

                sub = metric_complete_df.loc[common_seeds]
                acc_sub = acc_complete_df.loc[common_seeds]
                n_common = len(common_seeds)

                acc_diff_DA = (acc_sub[gD] - acc_sub[gA]).to_numpy(dtype=np.float64)
                acc_DA_mean = float(np.mean(acc_diff_DA))
                acc_DA_lcb95 = _paired_one_sided_lcb95(acc_diff_DA)
                passes_point_DA = bool(acc_DA_mean >= -cfg.acc_noninferiority_margin)
                passes_stat_DA = bool(
                    np.isfinite(acc_DA_lcb95) and (acc_DA_lcb95 > -cfg.acc_noninferiority_margin)
                )

                acc_diff_DB = (acc_sub[gD] - acc_sub[gB]).to_numpy(dtype=np.float64)
                acc_DB_mean = float(np.mean(acc_diff_DB))
                acc_DB_lcb95 = _paired_one_sided_lcb95(acc_diff_DB)
                passes_point_DB = bool(acc_DB_mean >= -cfg.acc_noninferiority_margin)
                passes_stat_DB = bool(
                    np.isfinite(acc_DB_lcb95) and (acc_DB_lcb95 > -cfg.acc_noninferiority_margin)
                )

                yA = sub[gA].to_numpy(dtype=np.float64)
                yB = sub[gB].to_numpy(dtype=np.float64)
                yC = sub[gC].to_numpy(dtype=np.float64)
                yD = sub[gD].to_numpy(dtype=np.float64)

                fact_main_drop = 0.5 * ((yB - yA) + (yD - yC))
                fact_main_bal = 0.5 * ((yC - yA) + (yD - yB))
                simple_drop_unbal = yB - yA
                simple_drop_given_bal = yD - yC
                simple_bal_nodrop = yC - yA
                simple_bal_given_drop = yD - yB
                combined_vs_base = yD - yA
                interaction_did = (yD - yC) - (yB - yA)

                mA, _, _, ciA, _, _ = _paired_summary_stats(yA)
                mB, _, _, ciB, _, _ = _paired_summary_stats(yB)
                mC, _, _, ciC, _, _ = _paired_summary_stats(yC)
                mD, _, _, ciD, _, _ = _paired_summary_stats(yD)

                m_fdrop, _, _, ci_fdrop, _, p_fdrop = _paired_summary_stats(fact_main_drop)
                m_fbal, _, _, ci_fbal, _, p_fbal = _paired_summary_stats(fact_main_bal)
                m_sdrop0, _, _, ci_sdrop0, _, p_sdrop0 = _paired_summary_stats(simple_drop_unbal)
                m_sdrop1, _, _, ci_sdrop1, _, p_sdrop1 = _paired_summary_stats(
                    simple_drop_given_bal
                )
                m_sbal0, _, _, ci_sbal0, _, p_sbal0 = _paired_summary_stats(simple_bal_nodrop)
                m_sbal1, sd_sbal1, se_sbal1, ci_sbal1, _, p_sbal1 = _paired_summary_stats(
                    simple_bal_given_drop
                )
                m_comb, _, _, ci_comb, _, p_comb = _paired_summary_stats(combined_vs_base)
                m_int, sd_int, se_int, ci_int, t_int, p_int = _paired_summary_stats(
                    interaction_did
                )

                degen_zero_var_int = bool(n_common >= 2 and se_int == 0.0)
                degen_zero_var_DB = bool(n_common >= 2 and se_sbal1 == 0.0)

                is_ops_metric = metric in lower_is_better_ops_metrics
                practical_DA_success = bool(
                    is_ops_metric
                    and np.isfinite(p_comb)
                    and (m_comb < 0.0)
                    and (p_comb < 0.05)
                    and passes_stat_DA
                )
                practical_DB_success = bool(
                    is_ops_metric
                    and np.isfinite(p_sbal1)
                    and (m_sbal1 < 0.0)
                    and (p_sbal1 < 0.05)
                    and passes_stat_DB
                )
                confirmatory_int_success = bool(
                    is_primary
                    and np.isfinite(p_int)
                    and (m_int < 0.0)
                    and (p_int < 0.05)
                    and practical_DB_success
                    and passes_stat_DA
                )

                db_ci_lower = (
                    float(m_sbal1 - ci_sbal1)
                    if (np.isfinite(m_sbal1) and np.isfinite(ci_sbal1))
                    else np.nan
                )
                db_ci_upper = (
                    float(m_sbal1 + ci_sbal1)
                    if (np.isfinite(m_sbal1) and np.isfinite(ci_sbal1))
                    else np.nan
                )
                int_ci_lower = (
                    float(m_int - ci_int)
                    if (np.isfinite(m_int) and np.isfinite(ci_int))
                    else np.nan
                )
                int_ci_upper = (
                    float(m_int + ci_int)
                    if (np.isfinite(m_int) and np.isfinite(ci_int))
                    else np.nan
                )

                rows.append({
                    "threshold_mode": thresh_mode,
                    "inference_steps": int(T),
                    "metric": metric,
                    "is_primary_confirmatory_outcome": is_primary,
                    "hypothesis_family_role": family_role,
                    "planned_family_size": row_family_size,
                    "acc_D_minus_A_mean": _float_or_nan(acc_DA_mean),
                    "acc_D_minus_A_lcb95_one_sided": _float_or_nan(acc_DA_lcb95),
                    "passes_point_acc_noninferiority_D_vs_A": passes_point_DA,
                    "passes_stat_acc_noninferiority_D_vs_A": passes_stat_DA,
                    "acc_D_minus_B_mean": _float_or_nan(acc_DB_mean),
                    "acc_D_minus_B_lcb95_one_sided": _float_or_nan(acc_DB_lcb95),
                    "passes_point_acc_noninferiority_D_vs_B": passes_point_DB,
                    "passes_stat_acc_noninferiority_D_vs_B": passes_stat_DB,
                    "degenerate_zero_var_interaction": degen_zero_var_int,
                    "degenerate_zero_var_D_minus_B": degen_zero_var_DB,
                    "practical_D_vs_A_efficiency_success": practical_DA_success,
                    "practical_D_vs_B_efficiency_success": practical_DB_success,
                    "confirmatory_interaction_efficiency_success": confirmatory_int_success,
                    "n_seeds": n_common,
                    "common_seeds_used": str([int(s) for s in common_seeds]),
                    "Group_A_mean": _float_or_nan(mA),
                    "Group_A_ci95_halfwidth": _float_or_nan(ciA),
                    "Group_B_mean": _float_or_nan(mB),
                    "Group_B_ci95_halfwidth": _float_or_nan(ciB),
                    "Group_C_mean": _float_or_nan(mC),
                    "Group_C_ci95_halfwidth": _float_or_nan(ciC),
                    "Group_D_mean": _float_or_nan(mD),
                    "Group_D_ci95_halfwidth": _float_or_nan(ciD),
                    "factorial_main_dropout_mean": _float_or_nan(m_fdrop),
                    "factorial_main_dropout_ci95_halfwidth": _float_or_nan(ci_fdrop),
                    "factorial_main_dropout_p_unadjusted_exploratory": _float_or_nan(p_fdrop),
                    "factorial_main_balance_mean": _float_or_nan(m_fbal),
                    "factorial_main_balance_ci95_halfwidth": _float_or_nan(ci_fbal),
                    "factorial_main_balance_p_unadjusted_exploratory": _float_or_nan(p_fbal),
                    "simple_dropout_unbalanced_B_minus_A_mean": _float_or_nan(m_sdrop0),
                    "simple_dropout_unbalanced_B_minus_A_ci95_halfwidth": _float_or_nan(ci_sdrop0),
                    "simple_dropout_unbalanced_B_minus_A_p_unadjusted_exploratory": _float_or_nan(
                        p_sdrop0
                    ),
                    "simple_dropout_given_balance_D_minus_C_mean": _float_or_nan(m_sdrop1),
                    "simple_dropout_given_balance_D_minus_C_ci95_halfwidth": _float_or_nan(
                        ci_sdrop1
                    ),
                    "simple_dropout_given_balance_D_minus_C_p_unadjusted_exploratory": _float_or_nan(
                        p_sdrop1
                    ),
                    "simple_balance_nodropout_C_minus_A_mean": _float_or_nan(m_sbal0),
                    "simple_balance_nodropout_C_minus_A_ci95_halfwidth": _float_or_nan(ci_sbal0),
                    "simple_balance_nodropout_C_minus_A_p_unadjusted_exploratory": _float_or_nan(
                        p_sbal0
                    ),
                    "simple_balance_given_dropout_D_minus_B_mean": _float_or_nan(m_sbal1),
                    "simple_balance_given_dropout_D_minus_B_sd": _float_or_nan(sd_sbal1),
                    "simple_balance_given_dropout_D_minus_B_ci95_halfwidth": _float_or_nan(
                        ci_sbal1
                    ),
                    "simple_balance_given_dropout_D_minus_B_ci95_lower": db_ci_lower,
                    "simple_balance_given_dropout_D_minus_B_ci95_upper": db_ci_upper,
                    "simple_balance_given_dropout_D_minus_B_p_unadjusted_exploratory": _float_or_nan(
                        p_sbal1
                    ),
                    "combined_D_minus_A_mean": _float_or_nan(m_comb),
                    "combined_D_minus_A_ci95_halfwidth": _float_or_nan(ci_comb),
                    "combined_D_minus_A_p_unadjusted_exploratory": _float_or_nan(p_comb),
                    "interaction_DiD_mean": _float_or_nan(m_int),
                    "interaction_DiD_sd": _float_or_nan(sd_int),
                    "interaction_DiD_se": _float_or_nan(se_int),
                    "interaction_DiD_ci95_halfwidth": _float_or_nan(ci_int),
                    "interaction_DiD_ci95_lower": int_ci_lower,
                    "interaction_DiD_ci95_upper": int_ci_upper,
                    "interaction_DiD_t_stat": _float_or_nan(t_int),
                    "interaction_DiD_p_unadjusted": _float_or_nan(p_int),
                    "per_seed_D_minus_B_diffs": str([float(x) for x in simple_bal_given_drop]),
                    "per_seed_D_minus_A_diffs": str([float(x) for x in combined_vs_base]),
                    "per_seed_interaction_DiD_diffs": str([float(x) for x in interaction_did]),
                })

    int_df = pd.DataFrame(rows)
    if len(int_df) > 0:
        # Apply Holm adjustment strictly across the 130 unique exploratory comparisons
        exploratory_mask = int_df["hypothesis_family_role"] == "exploratory_holm_family"
        exploratory_p_list = int_df.loc[exploratory_mask, "interaction_DiD_p_unadjusted"].tolist()
        adjusted_exploratory_p = _holm_bonferroni_adjust(
            exploratory_p_list, planned_family_size=exploratory_holm_family_size
        )
        holm_col = np.full(len(int_df), np.nan, dtype=np.float64)
        holm_col[exploratory_mask.to_numpy()] = adjusted_exploratory_p
        int_df["interaction_DiD_p_holm"] = holm_col

    int_df.to_csv(cfg.interaction_csv, index=False)
    if verbose:
        print(
            f"Saved 2x2 factorial interaction analysis ({len(int_df)} rows; "
            f"1 primary confirmatory test + {exploratory_holm_family_size} unique "
            f"exploratory Holm tests) to {cfg.interaction_csv}"
        )
    return int_df


# =====================================================================
# 9. Main Execution & Visualization Pipeline
# =====================================================================

def run_minimal_experiment_v2(cfg: ExperimentConfigV2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    (
        train_dataset,
        val_loader,
        calib_loader,
        test_loader,
        actual_dataset_name,
        split_sizes_str,
    ) = build_datasets_and_eval_loaders(cfg)

    # Step 1: Run comprehensive verification unit tests (Tests 1-7)
    run_all_verification_unit_tests(cfg, train_dataset, val_loader, calib_loader)

    # Step 2: Define Tier 1 (Core 2x2 Factorial Grid) and optional Tier 2 (Controls)
    experimental_conditions = [
        ("Group_A_Base (Drop:Off, Bal:Off)", False, "none"),
        ("Group_B_DropoutOnly (Drop:On, Bal:Off)", True, "none"),
        ("Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)", False, "interleaved"),
        ("Group_D_Interaction (Drop:On, Bal:Interleaved)", True, "interleaved"),
    ]

    if cfg.run_tier2_controls:
        tuned_alpha, tuning_status = tune_synoploss_alpha_on_validation(
            cfg, train_dataset, val_loader
        )
        experimental_conditions.extend([
            ("Control_PreTrainBalance (Drop:Off, Bal:Pre)", False, "pre_training"),
            ("Control_PreTrainBalance_Dropout (Drop:On, Bal:Pre)", True, "pre_training"),
            ("Control_PostTrainBalance (Drop:Off, Bal:Post)", False, "post_training"),
            (
                "Control_PostRandomRescale_NeuronMagMatched (Drop:Off, PostRandomScale)",
                False,
                "post_random_rescaling",
            ),
            (
                "Control_InterleavedRandomRescale_NeuronMagMatched (Drop:Off, InterleavedRandomScale)",
                False,
                "interleaved_random_rescaling",
            ),
            (
                "Control_InterleavedRandomRescale_Dropout_NeuronMagMatched (Drop:On, InterleavedRandomScale)",
                True,
                "interleaved_random_rescaling",
            ),
            ("Baseline_Fanout_L1_SynOpProxy (Drop:On, SynOpLoss)", True, "fanout_l1_synop_proxy"),
        ])
    else:
        tuned_alpha, tuning_status = 0.0, "tier2_disabled"

    records = []
    all_learning_curve_rows: List[Dict[str, object]] = []
    arch_str = "-".join(map(str, cfg.layer_sizes))
    seeds_to_run = cfg.get_active_seeds()

    for seed in seeds_to_run:
        seed_role_str = (
            "exploratory_pilot_seed"
            if int(seed) == int(cfg.exploratory_pilot_seed)
            else "fresh_confirmatory_seed"
        )
        print(
            f"--- Running Seed {seed} [{seed_role_str}] ({len(seeds_to_run)}-seed mode, "
            f"{cfg.epochs} epochs) on {actual_dataset_name} ({split_sizes_str}) ---"
        )
        seed_ref_cum_log_lambdas: Optional[List[torch.Tensor]] = None
        seed_ref_epoch_log_lambdas: Optional[List[List[torch.Tensor]]] = None
        seed_ref_drop_epoch_log_lambdas: Optional[List[List[torch.Tensor]]] = None

        for method_name, dropout_on, bal_sched in experimental_conditions:
            ref_ep_logs = (
                seed_ref_drop_epoch_log_lambdas
                if (dropout_on and seed_ref_drop_epoch_log_lambdas is not None)
                else seed_ref_epoch_log_lambdas
            )
            (
                ann,
                train_loop_sec,
                bal_sec,
                val_sec,
                bal_sweeps,
                rand_sweeps,
                bal_conv,
                cum_log_lam,
                epoch_log_lam,
                lc_stats,
            ) = train_and_prepare_condition(
                condition_name=method_name,
                dropout_on=dropout_on,
                balance_schedule=bal_sched,
                seed=seed,
                cfg=cfg,
                train_dataset=train_dataset,
                val_loader=val_loader,
                synoploss_alpha_override=tuned_alpha,
                reference_cum_log_lambdas=seed_ref_cum_log_lambdas,
                reference_epoch_log_lambdas=ref_ep_logs,
            )

            all_learning_curve_rows.extend(lc_stats["per_epoch_records"])

            if cfg.epochs >= cfg.val_plateau_window and not lc_stats["val_acc_plateau_reached"]:
                print(
                    f"  [Rolling Plateau Notice] {method_name} (seed={seed}): "
                    f"Last-{cfg.val_plateau_window}-epoch Val Acc span = "
                    f"{lc_stats['val_acc_window_max_min_span']:.4f} "
                    f"(net gain = {lc_stats['val_acc_window_net_gain']:+.4f}, "
                    f"tol = {cfg.val_plateau_delta_tol})."
                )

            if method_name == "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)":
                seed_ref_cum_log_lambdas = [t.clone() for t in cum_log_lam]
                seed_ref_epoch_log_lambdas = [
                    [t.clone() for t in ep_list] for ep_list in epoch_log_lam
                ]
            elif method_name == "Group_D_Interaction (Drop:On, Bal:Interleaved)":
                seed_ref_drop_epoch_log_lambdas = [
                    [t.clone() for t in ep_list] for ep_list in epoch_log_lam
                ]

            ann_val_acc = evaluate_ann(ann, val_loader, cfg.device)
            ann_test_acc = evaluate_ann(ann, test_loader, cfg.device)
            bal_metrics = compute_balance_metrics(
                ann,
                p=cfg.balance_norm,
                eps=cfg.balance_eps,
                include_bias_in_cost=cfg.include_bias_in_cost,
            )
            param_count = sum(p.numel() for p in ann.parameters())
            all_log_lam = torch.cat(cum_log_lam).numpy()
            mean_abs_log_lam = float(np.mean(np.abs(all_log_lam)))
            std_log_lam = float(np.std(all_log_lam))
            neuron_log_l2, edge_log_ratio_l2 = compute_edge_log_ratio_norm(
                cum_log_lam,
                cfg.layer_sizes,
                include_bias_edges=cfg.include_bias_in_cost,
            )

            for threshold_mode in ["fixed", "recalibrated"]:
                rng_snn = IsolatedRNGStreams(seed, device=cfg.device)
                snn = ConvertedSNN(
                    ann=ann,
                    threshold_mode=threshold_mode,
                    input_encoding=cfg.input_encoding,
                    spike_mode=cfg.spike_mode,
                    calib_loader=calib_loader,
                    percentile=cfg.calib_percentile,
                    fixed_threshold=1.0,
                ).to(cfg.device)

                snn_eval = snn.evaluate_budgets(
                    test_loader=test_loader,
                    timesteps_budget=cfg.timesteps_budget,
                    ann_test_acc=ann_test_acc,
                    stability_window_k=cfg.stability_window_k,
                    input_gen=rng_snn.gen_snn_input,
                )

                for T, m in snn_eval.items():
                    records.append({
                        "method": method_name,
                        "seed": seed,
                        "seed_role": seed_role_str,
                        "configured_dataset": cfg.dataset_name,
                        "actual_dataset_name": actual_dataset_name,
                        "split_sizes": split_sizes_str,
                        "is_synthetic_fallback": actual_dataset_name != "MNIST",
                        "architecture": arch_str,
                        "epochs": cfg.epochs,
                        "optimizer": cfg.optimizer_name,
                        "learning_rate": cfg.learning_rate,
                        "optimizer_state_mode": cfg.optimizer_state_mode,
                        "input_encoding": cfg.input_encoding,
                        "spike_mode": cfg.spike_mode,
                        "dropout_type": cfg.dropout_type if dropout_on else "none",
                        "dropout_rate": cfg.dropout_rate if dropout_on else 0.0,
                        "balance_schedule": bal_sched,
                        "balance_norm": f"L{cfg.balance_norm}",
                        "include_bias_in_cost": cfg.include_bias_in_cost,
                        "balance_sweeps_used": bal_sweeps,
                        "random_rescale_sweeps_used": rand_sweeps,
                        "balance_converged": bal_conv,
                        "synoploss_tuned_alpha": (
                            float(tuned_alpha) if bal_sched == "fanout_l1_synop_proxy" else 0.0
                        ),
                        "synoploss_tuning_status": (
                            tuning_status if bal_sched == "fanout_l1_synop_proxy" else "not_applicable"
                        ),
                        "threshold_mode": threshold_mode,
                        "calibration_policy_detail": (
                            f"tied_layerwise_p{cfg.calib_percentile}"
                            if threshold_mode == "recalibrated"
                            else "fixed_vth_1.0"
                        ),
                        "quantization": "none_fp32",
                        "inference_steps": int(T),
                        # Learning curve stabilization diagnostics & rolling-window plateau
                        "epoch_1_train_loss": _float_or_nan(lc_stats["epoch_1_train_loss"]),
                        "epoch_final_train_loss": _float_or_nan(lc_stats["epoch_final_train_loss"]),
                        "epoch_1_val_acc": _float_or_nan(lc_stats["epoch_1_val_acc"]),
                        "epoch_mid_val_acc": _float_or_nan(lc_stats["epoch_mid_val_acc"]),
                        "ann_val_accuracy": float(ann_val_acc),
                        "val_acc_last_2_epoch_delta": _float_or_nan(
                            lc_stats["val_acc_last_2_epoch_delta"]
                        ),
                        "val_acc_window_max_min_span": _float_or_nan(
                            lc_stats["val_acc_window_max_min_span"]
                        ),
                        "val_acc_window_net_gain": _float_or_nan(
                            lc_stats["val_acc_window_net_gain"]
                        ),
                        "val_acc_plateau_reached": lc_stats["val_acc_plateau_reached"],
                        "train_loss_history_by_epoch": str(lc_stats["train_loss_history"]),
                        "val_acc_history_by_epoch": str(lc_stats["val_acc_history"]),
                        "ann_test_accuracy": float(ann_test_acc),
                        "test_accuracy": float(m["test_accuracy"]),
                        "ann_to_snn_acc_gap": float(m["ann_to_snn_acc_gap"]),
                        # Separated early-stopping accuracies
                        "windowed_early_stop_overall_policy_accuracy": float(
                            m["windowed_early_stop_overall_policy_accuracy"]
                        ),
                        "windowed_early_stop_conditional_accuracy": _float_or_nan(
                            m["windowed_early_stop_conditional_accuracy"]
                        ),
                        "windowed_unconverged_final_accuracy": _float_or_nan(
                            m["windowed_unconverged_final_accuracy"]
                        ),
                        # Weight-space balance eligibility & manifold diagnostics
                        "n_total_neurons": bal_metrics["n_total_neurons"],
                        "n_active_eligible": bal_metrics["n_active_eligible"],
                        "eligible_neuron_fraction": float(bal_metrics["eligible_neuron_fraction"]),
                        "n_zero_cost_excluded": bal_metrics["n_zero_cost"],
                        "n_near_zero_excluded": bal_metrics["n_near_zero_cost"],
                        "n_bias_only_neurons": bal_metrics["n_bias_only_neurons"],
                        "raw_deficit_eligible": float(bal_metrics["raw_deficit_eligible"]),
                        "raw_deficit_all": float(bal_metrics["raw_deficit_all"]),
                        "norm_deficit_eligible": float(bal_metrics["mean_norm_deficit_eligible"]),
                        "norm_deficit_all": float(bal_metrics["mean_norm_deficit_all"]),
                        "max_norm_deficit_eligible": float(
                            bal_metrics["max_norm_deficit_eligible"]
                        ),
                        "max_norm_deficit_all": float(bal_metrics["max_norm_deficit_all"]),
                        "mean_abs_log_lambda": float(mean_abs_log_lam),
                        "std_log_lambda": float(std_log_lam),
                        "neuron_log_lambda_l2_norm": float(neuron_log_l2),
                        "edge_log_ratio_l2_norm_with_bias": float(edge_log_ratio_l2),
                        # Complete operation accounting (exact int64 accumulation -> float64)
                        "hidden_event_count_synops_proxy_per_sample": float(
                            m["hidden_event_count_synops_proxy_per_sample"]
                        ),
                        "windowed_early_stop_event_count_synops_per_sample": float(
                            m["windowed_early_stop_event_count_synops_per_sample"]
                        ),
                        "hidden_neurobench_eff_ACs_per_sample": float(
                            m["hidden_neurobench_eff_ACs_per_sample"]
                        ),
                        "hidden_neurobench_eff_MACs_per_sample": float(
                            m["hidden_neurobench_eff_MACs_per_sample"]
                        ),
                        "hidden_binary_active_step_ACs_per_sample": float(
                            m["hidden_binary_active_step_ACs_per_sample"]
                        ),
                        "hidden_active_spike_full_fanout_ops_per_sample": float(
                            m["hidden_active_spike_full_fanout_ops_per_sample"]
                        ),
                        "hidden_dense_SynOps_per_sample": float(
                            m["hidden_dense_SynOps_per_sample"]
                        ),
                        "input_eff_MACs_per_sample": float(m["input_eff_MACs_per_sample"]),
                        "input_static_MACs_per_sample": float(m["input_static_MACs_per_sample"]),
                        "input_repeated_eff_MACs_per_sample": float(
                            m["input_repeated_eff_MACs_per_sample"]
                        ),
                        "input_repeated_dense_ops_per_sample": float(
                            m["input_repeated_dense_ops_per_sample"]
                        ),
                        "input_spike_eff_ACs_per_sample": float(
                            m["input_spike_eff_ACs_per_sample"]
                        ),
                        "input_spike_active_full_fanout_ops_per_sample": float(
                            m["input_spike_active_full_fanout_ops_per_sample"]
                        ),
                        "total_ops_static_input_per_sample": float(
                            m["total_ops_static_input_per_sample"]
                        ),
                        "total_eff_ops_static_input_per_sample": float(
                            m["total_eff_ops_static_input_per_sample"]
                        ),
                        "total_ops_repeated_input_per_sample": float(
                            m["total_ops_repeated_input_per_sample"]
                        ),
                        # Per-example firing-rate quantiles, saturation & unambiguous sparsity
                        "spike_count_by_layer": str(m["spike_count_by_layer"]),
                        "silent_fraction_by_layer": str(m["silent_fraction_by_layer"]),
                        "firing_rate_median_by_layer": str(m["firing_rate_median_by_layer"]),
                        "firing_rate_p99_by_layer": str(m["firing_rate_p99_by_layer"]),
                        "firing_rate_max_by_layer": str(m["firing_rate_max_by_layer"]),
                        "firing_rate_max_to_mean_ratio_by_layer": str(
                            m["firing_rate_max_to_mean_ratio_by_layer"]
                        ),
                        "multispike_or_saturation_step_fraction": float(
                            m["multispike_or_saturation_step_fraction"]
                        ),
                        "post_reset_overflow_fraction": float(
                            m["post_reset_overflow_fraction"]
                        ),
                        "hidden_activation_sparsity": float(m["hidden_activation_sparsity"]),
                        "spiking_population_sparsity": float(m["spiking_population_sparsity"]),
                        "analog_input_zero_fraction": float(m["analog_input_zero_fraction"]),
                        # Latency & synchronized forward-step software simulation time
                        "retrospective_convergence_steps": float(
                            m["retrospective_convergence_steps"]
                        ),
                        "unconverged_fraction_retrospective": float(
                            m["unconverged_fraction_retrospective"]
                        ),
                        "windowed_stable_steps_censored": float(
                            m["windowed_stable_steps_censored"]
                        ),
                        "windowed_stable_steps_conditional": _float_or_nan(
                            m["windowed_stable_steps_conditional"]
                        ),
                        "unconverged_fraction_windowed": float(
                            m["unconverged_fraction_windowed"]
                        ),
                        "oracle_first_correct_steps": float(m["oracle_first_correct_steps"]),
                        "amortized_batch_sim_ms_per_sample": float(
                            m["amortized_batch_sim_ms_per_sample"]
                        ),
                        "parameter_count": param_count,
                        # Mutually exclusive training-phase wall-clock breakdown
                        "train_epoch_loop_wall_time_sec": float(train_loop_sec),
                        "balance_overhead_sec": float(bal_sec),
                        "val_eval_time_sec": float(val_sec),
                        "measured_training_phase_wall_time_sec": float(
                            train_loop_sec + bal_sec + val_sec
                        ),
                        "energy_value": None,
                        "energy_method": "hybrid_software_ops_proxy_only_not_joules",
                    })

    lc_df = pd.DataFrame(all_learning_curve_rows)
    lc_df.to_csv(cfg.learning_curves_csv, index=False)
    print(f"\nSaved per-epoch learning curves ({len(lc_df)} rows) to {cfg.learning_curves_csv}")

    df = pd.DataFrame(records)
    df.to_csv(cfg.output_csv, index=False)
    print(f"Saved full-precision experimental log ({len(df)} rows) to {cfg.output_csv}")

    int_df = compute_factorial_interaction_report(df, cfg)
    return df, int_df


def _compute_pareto_frontier_mask(ops: np.ndarray, accs: np.ndarray) -> np.ndarray:
    """Identifies non-dominated configurations (lower ops AND higher/equal accuracy)."""
    n = len(ops)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (ops[j] <= ops[i] and accs[j] >= accs[i]) and (
                ops[j] < ops[i] or accs[j] > accs[i]
            ):
                is_efficient[i] = False
                break
    return is_efficient


def plot_v2_diagnostics(df: pd.DataFrame, pareto_path: str, gap_path: str) -> None:
    """
    Generates:
      1. SNN Accuracy vs. Hidden Event-Count SynOps Proxy Tradeoff Curves with error bars
         (only plotted when n_seeds >= 2) and non-dominated Pareto Frontier points overlaid.
      2. ANN-to-SNN Accuracy Gap G(T) = Acc_ANN - Acc_SNN(T) vs. simulation duration T.
    """
    core_methods = [
        "Group_A_Base (Drop:Off, Bal:Off)",
        "Group_B_DropoutOnly (Drop:On, Bal:Off)",
        "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)",
        "Group_D_Interaction (Drop:On, Bal:Interleaved)",
        "Baseline_Fanout_L1_SynOpProxy (Drop:On, SynOpLoss)",
    ]
    sub = df[df["method"].isin(core_methods)]
    grouped = (
        sub.groupby(["threshold_mode", "method", "inference_steps"])
        .agg(
            n_seeds=("seed", "nunique"),
            acc_mean=("test_accuracy", "mean"),
            acc_std=("test_accuracy", "std"),
            gap_mean=("ann_to_snn_acc_gap", "mean"),
            gap_std=("ann_to_snn_acc_gap", "std"),
            hidden_ops_mean=("hidden_event_count_synops_proxy_per_sample", "mean"),
            hidden_ops_std=("hidden_event_count_synops_proxy_per_sample", "std"),
        )
        .reset_index()
    )

    has_multi_seed = bool((grouped["n_seeds"] >= 2).any())
    err_note = " (mean ± SD across seeds)" if has_multi_seed else " (1-seed point estimate; SD not estimable)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, mode in zip(axes, ["fixed", "recalibrated"]):
        mode_df = grouped[grouped["threshold_mode"] == mode].copy()
        for method in core_methods:
            m_df = mode_df[mode_df["method"] == method].sort_values("inference_steps")
            if len(m_df) == 0:
                continue
            if has_multi_seed:
                ax.errorbar(
                    m_df["hidden_ops_mean"],
                    m_df["acc_mean"],
                    xerr=m_df["hidden_ops_std"].to_numpy(),
                    yerr=m_df["acc_std"].to_numpy(),
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    label=method,
                )
            else:
                ax.plot(
                    m_df["hidden_ops_mean"],
                    m_df["acc_mean"],
                    marker="o",
                    linewidth=1.8,
                    label=method,
                )
        if len(mode_df) > 0:
            pareto_mask = _compute_pareto_frontier_mask(
                mode_df["hidden_ops_mean"].to_numpy(),
                mode_df["acc_mean"].to_numpy(),
            )
            p_df = mode_df[pareto_mask].sort_values("hidden_ops_mean")
            ax.plot(
                p_df["hidden_ops_mean"],
                p_df["acc_mean"],
                color="black",
                linestyle="--",
                marker="*",
                markersize=10,
                linewidth=1.2,
                label="Non-Dominated Pareto Frontier",
            )
        ax.set_title(f"Threshold Policy: {mode.capitalize()}")
        ax.set_xlabel(f"Hidden Event-Count SynOps Proxy per Sample{err_note}")
        ax.set_ylabel("SNN Test Accuracy")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    plt.suptitle("Minimal Experiment v2.2.8: SNN Accuracy vs. Event-Count SynOps Proxy & Pareto Frontier")
    plt.tight_layout()
    plt.savefig(pareto_path, dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, mode in zip(axes, ["fixed", "recalibrated"]):
        mode_df = grouped[grouped["threshold_mode"] == mode]
        for method in core_methods:
            m_df = mode_df[mode_df["method"] == method].sort_values("inference_steps")
            if len(m_df) == 0:
                continue
            if has_multi_seed:
                ax.errorbar(
                    m_df["inference_steps"],
                    m_df["gap_mean"],
                    yerr=m_df["gap_std"].to_numpy(),
                    marker="s",
                    linewidth=2,
                    capsize=3,
                    label=method,
                )
            else:
                ax.plot(
                    m_df["inference_steps"],
                    m_df["gap_mean"],
                    marker="s",
                    linewidth=2,
                    label=method,
                )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"Threshold Policy: {mode.capitalize()}")
        ax.set_xlabel("Simulation Timesteps (T)")
        ax.set_ylabel(f"ANN-to-SNN Accuracy Gap G(T) = Acc_ANN - Acc_SNN(T){err_note}")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    plt.suptitle("ANN-to-SNN Conversion Validation: Accuracy Gap Convergence Across T")
    plt.tight_layout()
    plt.savefig(gap_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    config = ExperimentConfigV2()
    results_df, interaction_df = run_minimal_experiment_v2(config)
    plot_v2_diagnostics(results_df, config.output_plot, config.gap_plot)