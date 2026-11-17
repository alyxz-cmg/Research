# Phase 0 Gap Statement: Synaptic Balance × Dropout for SNN Efficiency

**Prepared:** 28 October 2026 · **Status:** Provisional; this is not a guarantee of originality.

## Literature position

Baldi et al.'s published *A Theory of Synaptic Neural Balance: From Local to Global Order* defines balance as equality between the additive cost of a neuron's incoming and outgoing weights. Alexos, Domingo, and Baldi (AAAI 2025) test balancing before training and interleaved with SGD, reporting learning-speed/performance effects in ANN settings. Neither establishes that Baldi-style balance reduces energy after ANN-to-SNN conversion.

Sorbaro et al. (2020) are the central SNN-efficiency baseline: they combine a SynOp-oriented loss and activation quantization with ANN-to-SNN conversion and Sinabs simulation. Their CNN configurations already include dropout. Therefore, dropout in an SNN-related pipeline or SynOp reduction alone is not novel. Rueckauer et al. (2017) and Diehl et al. (2015) show that conversion, weight/threshold scaling and calibration affect SNN outcomes. NeuroBench (2025) provides standardized SynOps, sparsity, latency and system measurements. Related work also covers dropout/DropConnect under neuromorphic communication constraints (Sakai et al., 2019), weight scaling and bias adjustment for modeled power reduction (Merkel, 2023), homeostatic scaling in SNNs (Xu et al., 2025), and a distinct recurrent-network “synaptic balancing” rule (Stock et al., 2022).

## Candidate gap and question

This targeted review has **not verified a study that factorially tests Baldi-style synaptic balance × dropout during ANN training, converts the trained conditions through a controlled ANN-to-SNN pipeline, and compares accuracy–SynOps against conventional calibration and SynOp-aware training**. This is a candidate gap, not proof none exists. Before claiming novelty, perform citation chasing and full-text/supplement/code audits using synonyms such as weight rescaling, synaptic scaling, neuron-wise balancing and function-preserving reparameterization.

**Research question:** *Does interleaving Baldi-style balancing with dropout during ANN training change the accuracy–SynOps Pareto frontier after controlled ANN-to-SNN conversion, versus dropout alone, balance alone, conventional weight/threshold calibration, and a SynOp-regularized baseline? Under what conditions does any effect persist?*

## Minimal experiment

Start with a fully connected ReLU network and verify that balancing preserves ANN outputs within numerical tolerance. Run a matched 2×2 design (balance off/on × dropout off/on); separately compare pre-training versus interleaved balancing. Control the architecture, data split, optimizer budget, dropout placement, seeds and evaluation examples. Convert every model with one pinned ANN-to-SNN pipeline. Measure accuracy at fixed timestep budgets, NeuroBench SynOps/effective ACs, activation sparsity, per-layer activity, latency and model size. Include conventional weight/threshold calibration, random function-preserving rescaling where feasible, and a Sorbaro-style SynOp-loss baseline. Add low-bit weights or restricted threshold calibration only after the core effect is clear.

## Essential caveats

- **SynOps are not joules.** Without hardware measurements or a validated target-specific energy model including memory, communication, neuron updates and static power, claim operation-count changes, not physical energy savings.
- **Synaptic balance is not E/I balance.** E/I needs explicit excitatory/inhibitory populations (ideally obeying Dale's law) and current/input-based metrics; treat it as a separate study.
- **Conversion can confound results.** Quantization, thresholds, finite timesteps and normalization may alter equivalence. Document the conversion method and compare recalibrated versus restricted-threshold settings separately.
- **Define dropout exactly.** State rate, placement, mask semantics and timing relative to balancing/optimizer steps. Dropout, DropConnect and dropped spike-communication events differ.
- **Prioritize reproducibility.** Begin fully connected; handle biases, dead/zero-cost units, skip connections and weight sharing explicitly. Pin versions, verify SynOps on toy networks, use multiple seeds/confidence intervals, avoid test-set tuning and report null results.

## Key links

- [Baldi et al. (2025), published theory](https://www.sciencedirect.com/science/article/pii/S0004370225000797) · [theory code](https://github.com/ARahmansetayesh/a-theory-of-neural-synaptic-balance)
- [Alexos et al. (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/33683) · [code](https://github.com/antonyalexos/Neural-Balance)
- [Sorbaro et al. (2020)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full) · [SynOpLoss](https://gitlab.com/synsense/synoploss) · [Sinabs](https://github.com/synsense/sinabs)
- [NeuroBench (2025)](https://www.nature.com/articles/s41467-025-56739-4) · [code](https://github.com/NeuroBench/NeuroBench) · [SynOps metric](https://neurobench.readthedocs.io/en/latest/metrics/workload_metrics/synaptic_operations.html)
- Required adjacent comparisons: [Rueckauer et al. (2017)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00682/full), [Diehl et al. (2015)](https://doi.org/10.1109/IJCNN.2015.7280696), [Sakai et al. (2019)](https://doi.org/10.1109/JETCAS.2019.2952642), [Merkel (2023)](https://arxiv.org/abs/2306.07416), [Stock et al. (2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010418), [Barral & Reyes (2016)](https://www.nature.com/articles/nn.4415), [Xu et al. (2025)](https://ieeexplore.ieee.org/document/10479209/).

**Recommendation:** Treat balance × dropout as the first diagnostic experiment, not the finished paper. Determine whether the interaction appears during ANN training and survives conversion before expanding into E/I or physical-energy studies.