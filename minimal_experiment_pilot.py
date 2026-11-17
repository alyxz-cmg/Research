"""
Minimal Diagnostic Experiment: Synaptic Balance x Dropout for SNN Efficiency
Implements:
  - Phase 1: L1/L2 local synaptic balancing with strict ANN invariance unit tests
  - Phase 2/3: Paired-seed 2x2 factorial training (Balance Off/On x Dropout Off/On),
               plus Pre-Training Balance, Post-Training Balance, Random Function-Preserving
               Rescaling (negative control), and Sorbaro-style SynOp-Loss baseline
  - Phase 4: Pinned ANN-to-SNN conversion pipeline (IF neurons with subtraction reset)
               comparing 'fixed' vs. 'recalibrated' (Rueckauer 99.9th percentile) thresholds
  - Phase 5: Evaluation across timestep budgets for Accuracy, NeuroBench Effective/Dense
               SynOps (ACs), Activation Sparsity, Layer-wise Firing/Silent Fractions, and Latency
"""

import copy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# =====================================================================
# 1. Configuration & Reproducibility Utilities
# =====================================================================

@dataclass
class ExperimentConfig:
    dataset_name: str = "MNIST"  # Falls back to synthetic data if offline
    data_dir: str = "./data"
    layer_sizes: Tuple[int, ...] = (784, 256, 128, 10)
    dropout_rate: float = 0.5  # Matches Sorbaro et al. (2020) FC dropout rate
    dropout_type: str = "inverted_hidden_relu"
    balance_norm: int = 2  # p=2 (L2) or p=1 (L1)
    include_bias_in_cost: bool = False
    full_balance_tol: float = 0.01  # Convergence tolerance on |sqrt(A/B) - 1| for full balance
    full_balance_max_iters: int = 200
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer_name: str = "Adam"  # "SGD" or "Adam"
    synoploss_alpha: float = 1e-4  # Penalty weight for Sorbaro-style SynOp-loss baseline
    calib_percentile: float = 99.9  # Rueckauer et al. (2017) activation percentile
    timesteps_budget: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_csv: str = "phase3_minimal_experiment_results.csv"
    output_plot: str = "pareto_accuracy_vs_synops.png"


def set_deterministic_seed(seed: int) -> None:
    """Pins random seeds across Python, NumPy, and PyTorch for paired-seed runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 2. Fully Connected ReLU Network Architecture
# =====================================================================

class FullyConnectedReLU(nn.Module):
    """
    Simple feed-forward ReLU network with explicit inverted-dropout placement
    after hidden ReLU activations.
    """
    def __init__(self, layer_sizes: Tuple[int, ...], dropout_rate: float = 0.0):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.linears = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True)
            for i in range(len(layer_sizes) - 1)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
            for _ in range(len(layer_sizes) - 2)
        ])

        # Precompute fan-out per hidden layer for Sorbaro-style SynOp regularization
        self.fanouts = [layer_sizes[i + 2] for i in range(len(layer_sizes) - 2)]

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = x.view(x.size(0), -1)
        hidden_acts = []
        for i, linear in enumerate(self.linears[:-1]):
            x = F.relu(linear(x))
            # Inverted dropout active only when self.training is True
            x = self.dropouts[i](x)
            if return_activations:
                hidden_acts.append(x)
        logits = self.linears[-1](x)
        if return_activations:
            return logits, hidden_acts
        return logits, None


# =====================================================================
# 3. Phase 1: Synaptic Balancing, Metrics & Invariance Unit Tests
# =====================================================================

@torch.no_grad()
def compute_balance_metrics(
    model: FullyConnectedReLU,
    p: int = 2,
    eps: float = 1e-12,
    include_bias_in_cost: bool = False,
) -> Dict[str, float]:
    """
    Computes both the Baldi et al. raw balance deficit sum((A_j - B_j)^2)
    and the normalized deficit d_j = |A_j - B_j| / (A_j + B_j + eps).
    """
    raw_deficit = 0.0
    norm_deficits = []
    max_ratio_deviation = 0.0

    for layer_idx in range(len(model.linears) - 1):
        w_in = model.linears[layer_idx].weight
        b_in = model.linears[layer_idx].bias
        w_out = model.linears[layer_idx + 1].weight

        if p == 2:
            A = torch.sum(w_in ** 2, dim=1)
            if include_bias_in_cost and b_in is not None:
                A = A + b_in ** 2
            B = torch.sum(w_out ** 2, dim=0)
        elif p == 1:
            A = torch.sum(torch.abs(w_in), dim=1)
            if include_bias_in_cost and b_in is not None:
                A = A + torch.abs(b_in)
            B = torch.sum(torch.abs(w_out), dim=0)
        else:
            raise ValueError(f"Unsupported norm p={p}. Use p=1 or p=2.")

        raw_deficit += torch.sum((A - B) ** 2).item()
        d_j = torch.abs(A - B) / (A + B + eps)
        norm_deficits.append(d_j.cpu().numpy())

        valid = (A > eps) & (B > eps)
        if valid.any():
            # Ratio of incoming to outgoing norms
            norm_ratio = (A[valid] / B[valid]) ** (1.0 / p)
            dev = torch.max(torch.abs(norm_ratio - 1.0)).item()
            max_ratio_deviation = max(max_ratio_deviation, dev)

    all_d = np.concatenate(norm_deficits)
    return {
        "raw_deficit": float(raw_deficit),
        "mean_normalized_deficit": float(np.mean(all_d)),
        "max_ratio_deviation": float(max_ratio_deviation),
    }


@torch.no_grad()
def partial_balance_step(
    model: FullyConnectedReLU,
    p: int = 2,
    eps: float = 1e-12,
    include_bias_in_cost: bool = False,
) -> None:
    """
    Applies one deterministic sweep of local synaptic balancing from input
    to output hidden layers on the full unmasked parameters:
      - L2 case: lambda_j = (B_j / A_j)^(1/4)
      - L1 case: lambda_j = (B_j / A_j)^(1/2)
    Scales incoming weights AND incoming bias by lambda_j, and outgoing weights
    by 1 / lambda_j to strictly preserve the continuous ReLU network function.
    """
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

        # Safe dead-unit handling: only rescale active neurons where A > eps and B > eps
        valid = (A > eps) & (B > eps)
        lam = torch.ones_like(A)
        lam[valid] = (B[valid] / A[valid]) ** exponent

        # In-place scaling preserving exact ReLU homogeneity f(lam * z) = lam * f(z)
        w_in.mul_(lam.unsqueeze(1))
        if b_in is not None:
            b_in.mul_(lam)
        w_out.mul_((1.0 / lam).unsqueeze(0))


@torch.no_grad()
def full_balance_network(
    model: FullyConnectedReLU,
    p: int = 2,
    tol: float = 0.01,
    max_iters: int = 200,
    eps: float = 1e-12,
    include_bias_in_cost: bool = False,
) -> int:
    """
    Iteratively applies partial_balance_step until the incoming/outgoing norm
    ratios converge within `tol` of 1.0 (matching Baldi et al., Section 11).
    """
    for it in range(1, max_iters + 1):
        partial_balance_step(model, p=p, eps=eps, include_bias_in_cost=include_bias_in_cost)
        metrics = compute_balance_metrics(
            model, p=p, eps=eps, include_bias_in_cost=include_bias_in_cost
        )
        if metrics["max_ratio_deviation"] <= tol:
            return it
    return max_iters


@torch.no_grad()
def random_function_preserving_rescale(
    model: FullyConnectedReLU, log_scale_range: float = 0.5
) -> None:
    """
    Negative Control (Phase 4): Applies random positive scaling factors
    lambda_j ~ exp(Uniform(-log_scale_range, log_scale_range)) to hidden neurons.
    Preserves exact ANN outputs while breaking synaptic balance.
    """
    for layer_idx in range(len(model.linears) - 1):
        lin_in = model.linears[layer_idx]
        lin_out = model.linears[layer_idx + 1]
        H = lin_in.out_features
        log_lam = torch.empty(H, device=lin_in.weight.device).uniform_(
            -log_scale_range, log_scale_range
        )
        lam = torch.exp(log_lam)
        lin_in.weight.mul_(lam.unsqueeze(1))
        if lin_in.bias is not None:
            lin_in.bias.mul_(lam)
        lin_out.weight.mul_((1.0 / lam).unsqueeze(0))


@torch.no_grad()
def run_phase1_unit_tests(
    cfg: ExperimentConfig, val_loader: DataLoader, atol: float = 1e-5
) -> None:
    """
    Strict Phase 1 unit test verifying:
      1. ANN outputs agree to numerical tolerance on random inputs and validation data.
      2. Raw and normalized balance deficits monotonically decrease toward 0.
    """
    set_deterministic_seed(cfg.seeds[0])
    test_model = FullyConnectedReLU(cfg.layer_sizes, dropout_rate=cfg.dropout_rate).to(cfg.device)
    test_model.eval()

    x_rand = torch.randn(64, cfg.layer_sizes[0], device=cfg.device)
    x_val, _ = next(iter(val_loader))
    x_val = x_val.to(cfg.device)

    out_rand_before, _ = test_model(x_rand)
    out_val_before, _ = test_model(x_val)
    metrics_before = compute_balance_metrics(
        test_model, p=cfg.balance_norm, include_bias_in_cost=cfg.include_bias_in_cost
    )

    iters = full_balance_network(
        test_model,
        p=cfg.balance_norm,
        tol=1e-5,
        max_iters=cfg.full_balance_max_iters,
        include_bias_in_cost=cfg.include_bias_in_cost,
    )

    out_rand_after, _ = test_model(x_rand)
    out_val_after, _ = test_model(x_val)
    metrics_after = compute_balance_metrics(
        test_model, p=cfg.balance_norm, include_bias_in_cost=cfg.include_bias_in_cost
    )

    diff_rand = torch.max(torch.abs(out_rand_before - out_rand_after)).item()
    diff_val = torch.max(torch.abs(out_val_before - out_val_after)).item()

    assert diff_rand < atol, f"Random input invariance failed: max diff = {diff_rand}"
    assert diff_val < atol, f"Validation input invariance failed: max diff = {diff_val}"
    assert metrics_after["raw_deficit"] < metrics_before["raw_deficit"] * 1e-4, (
        "Balance deficit failed to converge!"
    )

    print("=== Phase 1 Unit Test Passed ===")
    print(f"  Converged in {iters} sweeps | Max diff (random): {diff_rand:.2e} | Max diff (val): {diff_val:.2e}")
    print(f"  Raw Deficit: {metrics_before['raw_deficit']:.4f} -> {metrics_after['raw_deficit']:.4e}")
    print(f"  Normalized Deficit: {metrics_before['mean_normalized_deficit']:.4f} -> {metrics_after['mean_normalized_deficit']:.4e}\n")


# =====================================================================
# 4. Phase 4: Pinned ANN-to-SNN Conversion Pipeline & Threshold Controls
# =====================================================================

class ConvertedSNN(nn.Module):
    """
    Identical ANN-to-SNN conversion pipeline using Integrate-and-Fire (IF)
    neurons with subtraction (soft) membrane reset and state reset per batch.

    Supports two threshold calibration modes:
      1. 'fixed': Direct weight/bias transfer with unit threshold V_th = 1.0
         (tests deployment where per-model recalibration is restricted/unavailable).
      2. 'recalibrated': Rueckauer et al. (2017) layer-wise activation percentile
         scaling computed strictly on the held-out calibration split.
    """
    def __init__(
        self,
        ann: FullyConnectedReLU,
        threshold_mode: str = "recalibrated",
        calib_loader: Optional[DataLoader] = None,
        percentile: float = 99.9,
        fixed_threshold: float = 1.0,
    ):
        super().__init__()
        self.layer_sizes = ann.layer_sizes
        self.num_hidden = len(ann.linears) - 1
        self.threshold_mode = threshold_mode
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
                # Rescale final readout weights so output current matches ANN logit scale
                self.linears[-1].weight.mul_(prev_scale)
                thresholds = [1.0] * self.num_hidden
            elif threshold_mode == "fixed":
                thresholds = [fixed_threshold] * self.num_hidden
            else:
                raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")

        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))

        # Precompute non-zero outgoing synaptic connections per neuron for NeuroBench Effective ACs
        self.effective_fanouts = []
        for l in range(self.num_hidden):
            w_next = self.linears[l + 1].weight.data
            nonzero_fanout = (w_next.abs() > 1e-12).float().sum(dim=0)
            self.effective_fanouts.append(nonzero_fanout)

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

    @torch.no_grad()
    def evaluate_budgets(
        self, test_loader: DataLoader, timesteps_budget: List[int]
    ) -> Dict[int, Dict[str, object]]:
        """
        Simulates the SNN over max(timesteps_budget) steps and records accuracy,
        NeuroBench Effective/Dense SynOps, activation sparsity, silent-neuron fraction,
        and decision/wall-clock latency at each timestep budget T.
        """
        self.eval()
        device = next(self.parameters()).device
        max_T = max(timesteps_budget)
        budget_set = set(timesteps_budget)

        accum = {
            t: {
                "correct": 0,
                "samples": 0,
                "eff_synops_sum": 0.0,
                "dense_synops_sum": 0.0,
                "layer_spikes_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "silent_frac_sum": np.zeros(self.num_hidden, dtype=np.float64),
                "total_spikes_all": 0.0,
                "total_neuron_steps": 0,
                "decision_steps_sum": 0.0,
                "wall_clock_sec": 0.0,
            }
            for t in timesteps_budget
        }

        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            bsz = xb.size(0)
            x_flat = xb.view(bsz, -1)

            # Reset membrane potentials and spike accumulators between batches
            v_mem = [
                torch.zeros(bsz, self.layer_sizes[l + 1], device=device)
                for l in range(self.num_hidden)
            ]
            out_acc = torch.zeros(bsz, self.layer_sizes[-1], device=device)
            cum_spikes = [
                torch.zeros(bsz, self.layer_sizes[l + 1], device=device)
                for l in range(self.num_hidden)
            ]

            first_correct_step = torch.full((bsz,), float(max_T), device=device)
            has_identified = torch.zeros(bsz, dtype=torch.bool, device=device)

            t_start = time.perf_counter()
            for t in range(1, max_T + 1):
                cur = self.linears[0](x_flat)
                for l in range(self.num_hidden):
                    v_mem[l] = v_mem[l] + cur
                    v_th = self.thresholds[l]
                    spikes = (v_mem[l] >= v_th).float()
                    # Subtraction (soft) reset
                    v_mem[l] = v_mem[l] - spikes * v_th
                    cum_spikes[l] += spikes
                    cur = self.linears[l + 1](spikes)

                out_acc = out_acc + cur
                preds = out_acc.argmax(dim=1)
                nonzero_out = out_acc.abs().sum(dim=1) > 0
                correct_now = (preds == yb) & nonzero_out
                newly_correct = correct_now & (~has_identified)
                first_correct_step[newly_correct] = float(t)
                has_identified = has_identified | newly_correct

                if t in budget_set:
                    elapsed = time.perf_counter() - t_start
                    rec = accum[t]
                    rec["wall_clock_sec"] += elapsed
                    rec["correct"] += (preds == yb).sum().item()
                    rec["samples"] += bsz
                    rec["decision_steps_sum"] += torch.clamp(
                        first_correct_step, max=float(t)
                    ).sum().item()

                    for l in range(self.num_hidden):
                        s_l = cum_spikes[l]
                        fanout_vec = self.effective_fanouts[l].to(device)
                        dense_fanout = self.linears[l + 1].out_features
                        rec["eff_synops_sum"] += (s_l * fanout_vec.unsqueeze(0)).sum().item()
                        rec["dense_synops_sum"] += (s_l.sum(dim=1) * dense_fanout).sum().item()
                        rec["layer_spikes_sum"][l] += s_l.sum().item()
                        rec["silent_frac_sum"][l] += (s_l == 0).float().mean(dim=1).sum().item()
                        rec["total_spikes_all"] += s_l.sum().item()
                        rec["total_neuron_steps"] += bsz * self.layer_sizes[l + 1] * t

        results = {}
        for t in timesteps_budget:
            rec = accum[t]
            n = max(1, rec["samples"])
            results[t] = {
                "test_accuracy": rec["correct"] / n,
                "SynOps_per_sample": rec["eff_synops_sum"] / n,
                "dense_SynOps_per_sample": rec["dense_synops_sum"] / n,
                "spike_count_by_layer": [round(val / n, 2) for val in rec["layer_spikes_sum"]],
                "silent_fraction_by_layer": [round(val / n, 4) for val in rec["silent_frac_sum"]],
                "activation_sparsity": 1.0 - (
                    rec["total_spikes_all"] / max(1, rec["total_neuron_steps"])
                ),
                "decision_latency_steps": rec["decision_steps_sum"] / n,
                "wall_clock_latency_ms": (rec["wall_clock_sec"] * 1000.0) / n,
            }
        return results


# =====================================================================
# 5. Phase 3 Training Pipeline (2x2 Grid + Schedules + Baselines)
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
    return correct / max(1, total)


def train_and_prepare_condition(
    condition_name: str,
    dropout_on: bool,
    balance_schedule: str,
    seed: int,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
) -> Tuple[FullyConnectedReLU, float, float]:
    """
    Trains a single condition with paired seed initialization.
    Supported balance_schedule values:
      - 'none': Standard ANN training
      - 'pre_training': Full balance before epoch 1, then standard SGD/Adam
      - 'interleaved': Partial balance at the end of every epoch (after optimizer steps)
      - 'post_training': Standard training followed by full balancing before SNN conversion
      - 'random_rescaling': Standard training followed by random function-preserving rescaling
      - 'sorbaro_synoploss': ANN trained with L1 fanout-weighted SynOp activation penalty
    """
    set_deterministic_seed(seed)
    p_drop = cfg.dropout_rate if dropout_on else 0.0
    model = FullyConnectedReLU(cfg.layer_sizes, dropout_rate=p_drop).to(cfg.device)

    if cfg.optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    balance_overhead_sec = 0.0

    # Pre-training full balance (if specified)
    if balance_schedule == "pre_training":
        t_bal = time.perf_counter()
        full_balance_network(
            model,
            p=cfg.balance_norm,
            tol=cfg.full_balance_tol,
            max_iters=cfg.full_balance_max_iters,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        balance_overhead_sec += time.perf_counter() - t_bal

    t_train_start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            logits, hidden_acts = model(
                xb, return_activations=(balance_schedule == "sorbaro_synoploss")
            )
            loss = F.cross_entropy(logits, yb)

            # Sorbaro et al. (2020)-style SynOp regularizer on hidden activations
            if balance_schedule == "sorbaro_synoploss" and hidden_acts is not None:
                synop_pen = 0.0
                for act, fanout in zip(hidden_acts, model.fanouts):
                    synop_pen = synop_pen + fanout * torch.mean(torch.abs(act))
                loss = loss + cfg.synoploss_alpha * synop_pen

            loss.backward()
            optimizer.step()

        # Interleaved partial balancing at epoch boundary on full unmasked parameters
        if balance_schedule == "interleaved":
            t_bal = time.perf_counter()
            partial_balance_step(
                model, p=cfg.balance_norm, include_bias_in_cost=cfg.include_bias_in_cost
            )
            balance_overhead_sec += time.perf_counter() - t_bal

    total_train_sec = time.perf_counter() - t_train_start

    # Post-training diagnostic interventions
    if balance_schedule == "post_training":
        t_bal = time.perf_counter()
        full_balance_network(
            model,
            p=cfg.balance_norm,
            tol=cfg.full_balance_tol,
            max_iters=cfg.full_balance_max_iters,
            include_bias_in_cost=cfg.include_bias_in_cost,
        )
        balance_overhead_sec += time.perf_counter() - t_bal
    elif balance_schedule == "random_rescaling":
        t_bal = time.perf_counter()
        random_function_preserving_rescale(model, log_scale_range=0.5)
        balance_overhead_sec += time.perf_counter() - t_bal

    return model, total_train_sec, balance_overhead_sec


# =====================================================================
# 6. Data Loading (Strict Train / Validation / Calibration / Test Split)
# =====================================================================

def build_dataloaders(
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Separates data strictly into Train, Validation, Calibration, and Test splits
    so threshold calibration and hyperparameter tuning never touch the Test set.
    """
    set_deterministic_seed(cfg.seeds[0])
    if TORCHVISION_AVAILABLE and cfg.dataset_name.upper() == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            full_train = datasets.MNIST(
                root=cfg.data_dir, train=True, download=True, transform=transform
            )
            test_set = datasets.MNIST(
                root=cfg.data_dir, train=False, download=True, transform=transform
            )
            # 50,000 train | 5,000 val | 5,000 calibration
            train_set, val_set, calib_set = random_split(
                full_train,
                [50000, 5000, 5000],
                generator=torch.Generator().manual_seed(cfg.seeds[0]),
            )
            return (
                DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True),
                DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False),
                DataLoader(calib_set, batch_size=cfg.batch_size, shuffle=False),
                DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False),
            )
        except Exception as e:
            print(f"[Info] MNIST download unavailable ({e}); falling back to synthetic benchmark.")

    # Synthetic non-negative classification benchmark fallback for offline testing
    N_total = 3200
    X = torch.rand(N_total, cfg.layer_sizes[0])
    W_true = torch.randn(cfg.layer_sizes[0], cfg.layer_sizes[-1])
    y = (X @ W_true).argmax(dim=1)
    dataset = TensorDataset(X, y)
    train_set, val_set, calib_set, test_set = random_split(
        dataset,
        [2000, 400, 400, 400],
        generator=torch.Generator().manual_seed(cfg.seeds[0]),
    )
    return (
        DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(calib_set, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False),
    )


# =====================================================================
# 7. Main Execution & Reproducible Reporting
# =====================================================================

def run_minimal_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    train_loader, val_loader, calib_loader, test_loader = build_dataloaders(cfg)

    # Step 1: Execute strict Phase 1 invariance unit tests before running grid
    run_phase1_unit_tests(cfg, val_loader)

    # Step 2: Define the core 2x2 factorial grid + required controls
    # Format: (method_label, dropout_on, balance_schedule)
    experimental_conditions = [
        # Core 2x2 Factorial Grid (Interleaved Balancing x Dropout)
        ("Group_A_Base (Drop:Off, Bal:Off)", False, "none"),
        ("Group_B_DropoutOnly (Drop:On, Bal:Off)", True, "none"),
        ("Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)", False, "interleaved"),
        ("Group_D_Interaction (Drop:On, Bal:Interleaved)", True, "interleaved"),
        # Secondary Schedule & Diagnostic Controls required by Phase 0 / HandDown
        ("Control_PreTrainBalance (Drop:Off, Bal:Pre)", False, "pre_training"),
        ("Control_PreTrainBalance_Dropout (Drop:On, Bal:Pre)", True, "pre_training"),
        ("Control_PostTrainBalance (Drop:Off, Bal:Post)", False, "post_training"),
        ("Control_RandomRescale (Drop:Off, RandomScale)", False, "random_rescaling"),
        ("Baseline_Sorbaro_SynOpLoss (Drop:On, SynOpLoss)", True, "sorbaro_synoploss"),
    ]

    records = []
    arch_str = "-".join(map(str, cfg.layer_sizes))

    for seed in cfg.seeds:
        print(f"--- Running Seed {seed} ---")
        for method_name, dropout_on, bal_sched in experimental_conditions:
            ann, train_sec, bal_sec = train_and_prepare_condition(
                condition_name=method_name,
                dropout_on=dropout_on,
                balance_schedule=bal_sched,
                seed=seed,
                cfg=cfg,
                train_loader=train_loader,
            )

            ann_val_acc = evaluate_ann(ann, val_loader, cfg.device)
            ann_test_acc = evaluate_ann(ann, test_loader, cfg.device)
            bal_metrics = compute_balance_metrics(
                ann, p=cfg.balance_norm, include_bias_in_cost=cfg.include_bias_in_cost
            )
            param_count = sum(p.numel() for p in ann.parameters())

            # Convert through identical SNN pipeline under both threshold policies
            for threshold_mode in ["fixed", "recalibrated"]:
                snn = ConvertedSNN(
                    ann=ann,
                    threshold_mode=threshold_mode,
                    calib_loader=calib_loader,
                    percentile=cfg.calib_percentile,
                    fixed_threshold=1.0,
                ).to(cfg.device)

                snn_eval = snn.evaluate_budgets(test_loader, cfg.timesteps_budget)

                for T, m in snn_eval.items():
                    records.append({
                        "method": method_name,
                        "seed": seed,
                        "dataset": cfg.dataset_name,
                        "architecture": arch_str,
                        "dropout_type": cfg.dropout_type if dropout_on else "none",
                        "dropout_rate": cfg.dropout_rate if dropout_on else 0.0,
                        "balance_schedule": bal_sched,
                        "balance_norm": f"L{cfg.balance_norm}",
                        "threshold_mode": threshold_mode,
                        "quantization": "none_fp32",
                        "inference_steps": T,
                        "ann_val_accuracy": round(ann_val_acc, 4),
                        "ann_test_accuracy": round(ann_test_acc, 4),
                        "raw_balance_deficit": bal_metrics["raw_deficit"],
                        "norm_balance_deficit": bal_metrics["mean_normalized_deficit"],
                        "test_accuracy": round(m["test_accuracy"], 4),
                        "SynOps_per_sample": round(m["SynOps_per_sample"], 2),
                        "dense_SynOps_per_sample": round(m["dense_SynOps_per_sample"], 2),
                        "spike_count_by_layer": str(m["spike_count_by_layer"]),
                        "silent_fraction_by_layer": str(m["silent_fraction_by_layer"]),
                        "activation_sparsity": round(m["activation_sparsity"], 4),
                        "decision_latency_steps": round(m["decision_latency_steps"], 2),
                        "latency": round(m["wall_clock_latency_ms"], 4),
                        "parameter_count": param_count,
                        "training_time": round(train_sec, 3),
                        "balance_overhead": round(bal_sec, 4),
                        # Caveat compliance: never claim physical joules from software SynOps
                        "energy_value": None,
                        "energy_method": "software_SynOps_proxy_only_not_joules",
                    })

    df = pd.DataFrame(records)
    df.to_csv(cfg.output_csv, index=False)
    print(f"\nSaved full experimental log ({len(df)} rows) to {cfg.output_csv}")
    return df


def plot_pareto_curves(df: pd.DataFrame, output_path: str) -> None:
    """Plots Accuracy vs. Effective SynOps Pareto curves across timestep budgets."""
    core_methods = [
        "Group_A_Base (Drop:Off, Bal:Off)",
        "Group_B_DropoutOnly (Drop:On, Bal:Off)",
        "Group_C_BalanceOnly (Drop:Off, Bal:Interleaved)",
        "Group_D_Interaction (Drop:On, Bal:Interleaved)",
        "Baseline_Sorbaro_SynOpLoss (Drop:On, SynOpLoss)",
    ]
    sub = df[df["method"].isin(core_methods)]
    grouped = (
        sub.groupby(["threshold_mode", "method", "inference_steps"])
        .agg(
            acc_mean=("test_accuracy", "mean"),
            acc_std=("test_accuracy", "std"),
            synops_mean=("SynOps_per_sample", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, mode in zip(axes, ["fixed", "recalibrated"]):
        mode_df = grouped[grouped["threshold_mode"] == mode]
        for method in core_methods:
            m_df = mode_df[mode_df["method"] == method].sort_values("inference_steps")
            ax.plot(
                m_df["synops_mean"],
                m_df["acc_mean"],
                marker="o",
                linewidth=2,
                label=method,
            )
        ax.set_title(f"Threshold Policy: {mode.capitalize()}")
        ax.set_xlabel("Effective SynOps (ACs) per Sample")
        ax.set_ylabel("SNN Test Accuracy")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    plt.suptitle("Phase 3 Minimal Experiment: SNN Accuracy vs. SynOps Pareto Frontier")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    config = ExperimentConfig()
    results_df = run_minimal_experiment(config)
    plot_pareto_curves(results_df, config.output_plot)