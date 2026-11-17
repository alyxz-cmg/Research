# Research Handoff: Synaptic Neural Balance for Neuromorphic Computing

**Purpose:** Give a future agent or research collaborator enough context to continue this project without reconstructing the conversation.

**Last updated:** 2026-10-26  
**Project status:** Early research-planning stage. No code, experiments, or results have been reported in this conversation.  
**Primary advisor:** Professor Pierre Baldi, UC Irvine. The student reports that Professor Baldi is their research advisor.

---

## 1. Executive summary

The student wants to pursue an original research project based on Section 12.4, **“Neuromorphic Hardware,”** of Pierre Baldi et al., *A Theory of Synaptic Neural Balance: From Local to Global Order*.

The key premise in that section is that synaptic balancing is a local operation that may have useful global consequences in physical neural systems. A neuron may not have access to the network-wide training loss, but it could in principle monitor a local balance quantity. The paper also argues that constraints of neuromorphic hardware may make some balanced representations preferable, and that balancing *may* help minimize energy. The paper does **not** establish that synaptic balancing reduces energy on neuromorphic hardware; that remains a research question.

The student emailed Professor Baldi to express interest. His response was:

> Yes this is an interesting direction. I would add to your list three things: (1) trying to understand if balance leads to more energy efficient neuromorphic systems (it may depend on the hardware details); (2) trying to understand if synaptic balance leads to, or at least favors, excitation/inhibition balance; (3) what happens when you interleave dropout and balancing. These are real research questions so they require quite a bit of work and do not expect your advisor to do the work for you. Real progress on any of these questions would be awesome. If you make some progress, I will put you in touch with one of my grad students.

The most practical initial plan is to **build a small, reproducible simulation pipeline, validate the balance transformation, and test the interaction between synaptic balancing and dropout while measuring downstream spiking-neural-network (SNN) efficiency**. This uses a manageable training-method experiment to establish technical credibility and connects it to the advisor's first and third questions. The excitation/inhibition (E/I) question should initially be treated as a separate branch because it requires an explicit model of excitatory and inhibitory populations and careful operational definitions.

Do not assume in advance that balance reduces energy, that balance implies E/I balance, or that dropout and balance interact beneficially. These are hypotheses to test. A null result is possible and must be reported honestly.

### Provisional working title

**Interleaving Synaptic Balance and Dropout for Efficient Spiking Neural Network Inference**

This is only a starting title. It must be revised after the literature review and pilot experiments identify a specific novel contribution.

### Primary candidate research question

Does applying synaptic neural balancing before or during training, alone and in combination with dropout, improve the accuracy-versus-synaptic-operation trade-off after conversion to an SNN, relative to established scaling and energy-aware training baselines?

### Important scope warning

The advisor suggested **three possible research directions**, not necessarily one paper that must solve all three. Avoid forcing energy efficiency, E/I balance, and dropout into a single over-scoped project. A staged plan can start with balance plus dropout, carry out an SNN efficiency evaluation, and keep E/I balance as a separate extension if there is time and a scientifically suitable network model.

---

## 2. Conversation history and decisions so far

1. The student first explored research directions involving synthetic data, reconstruction, and machine learning for physical or biological systems.
2. The student then became interested in continuing established papers by following discussion-section suggestions for future work.
3. The relevant paper was *A Theory of Synaptic Neural Balance: From Local to Global Order* by Pierre Baldi, Antonios Alexos, Ian Domingo, and Alireza Rahmansetayesh.
4. Section 12.4 motivated neuromorphic work. It says local weight-balancing operations may be meaningful in physical systems; local balance could potentially serve as a proxy for global progress; hardware constraints could make balanced weights useful; and balancing may help minimize energy given the known connection between ReLU scaling, firing rates, and SNN energy.
5. Initial ideas were proposed around measuring SNN activity, SynOps, and accuracy after balancing; comparing against established scaling methods; and testing hardware constraints such as low-precision weights.
6. The student contacted Professor Baldi. He explicitly suggested adding three research questions: energy efficiency on actual/particular neuromorphic hardware, possible relation between synaptic balance and E/I balance, and interleaving dropout and balancing. He called these real research questions and said the student should make progress independently. If progress is made, he will introduce the student to one of his graduate students.
7. The immediate practical recommendation was to start with a small controlled experiment, not draft a paper immediately. Verify the method, build a baseline, collect a preliminary table/plot, and then use evidence to choose the strongest direction.

No claims of a confirmed original contribution have been established. Literature novelty remains unverified.

---

## 3. Source paper and the meaning of Section 12.4

### Core paper

Pierre Baldi, Antonios Alexos, Ian Domingo, Alireza Rahmansetayesh. **A Theory of Synaptic Neural Balance: From Local to Global Order.**

- arXiv: https://arxiv.org/abs/2405.09688
- PDF: https://arxiv.org/pdf/2405.09688
- Published version: *Artificial Intelligence*, volume 346 (2025), article 104360, DOI https://doi.org/10.1016/j.artint.2025.104360

Use the published version as well as the arXiv version during the literature review. The arXiv PDF viewed in this conversation was version 4, dated 2024-10-31. Section 12.4 appears on printed page 37.

### What synaptic neural balance means

For a hidden ReLU neuron, one simple L2 definition is:

\[
A_j = \sum_{i\in IN(j)} w_{ij}^2, \qquad B_j = \sum_{k\in OUT(j)} w_{jk}^2.
\]

The neuron is balanced under this cost when \(A_j=B_j\). This is a relationship between the cost of **incoming and outgoing weights of an individual neuron**. It is not the same concept as excitatory/inhibitory balance.

ReLU positive homogeneity permits a neuron-level rescaling in an ordinary feed-forward network. With \(\lambda>0\), scale the neuron's incoming weights by \(\lambda\) and its outgoing weights by \(1/\lambda\). The network input-output function is unchanged under the relevant assumptions. For the L2 example, minimizing

\[
\lambda^2 A_j + \lambda^{-2} B_j
\]

produces \(\lambda^*=(B_j/A_j)^{1/4}\), making the rescaled incoming and outgoing squared-weight sums equal. Bias terms and all outgoing connections must be handled correctly in an implementation. General regularizers change the formula.

**Critical limitation:** preserving a conventional ANN's function does not guarantee the exact same behavior once it is converted to a spiking network with finite thresholds, time steps, quantization, and spike discretization. SNN behavior and thresholds must be evaluated explicitly.

### Section 12.4's proposal, in plain language

The section's argument has several parts:

- A physical neural system may only permit local weight updates, yet the desired network behavior is global.
- Synaptic balancing uses local information, so it is a candidate operation for biological or neuromorphic systems.
- A neuron might be able to monitor its local balance even when it cannot compute the global training loss.
- Hardware constraints may make particular distributions of weights or learning algorithms preferable.
- Earlier SNN research shows that ReLU homogeneity, scaling, and firing rates are linked to SynOps and estimated energy.
- Therefore, balancing *may* help minimize energy, but the paper leaves this as a possibility, not an established result.

Section 12.3 also notes that the evidence for the specific form of synaptic neural balance in biological brains is limited. Do not present the proposed method as an established biological mechanism.

Section 12.2 separately likens balancing to a regularization tool similar in spirit to dropout. The advisor's explicit interest in interleaving dropout and balancing should be treated as a real experimental question, not as evidence that their combination is already validated.

---

## 4. Three research directions explicitly suggested by the advisor

### Track A: Does synaptic balance lead to more energy-efficient neuromorphic systems?

**Question:** Under what hardware and SNN-conversion conditions, if any, does balancing reduce energy while preserving useful task performance?

Potential outcomes:

- Lower spike-triggered SynOps at matched accuracy.
- A better accuracy-versus-SynOps Pareto curve.
- Better robustness to weight quantization or restricted threshold calibration.
- No benefit after standard threshold calibration, which would be a meaningful limitation if rigorously demonstrated.
- Hardware-specific effects, meaning any conclusions must identify the architecture and operating assumptions.

Do not treat SynOps as identical to joules. Operation counts are an energy proxy. Hardware energy can additionally include neuron state updates, memory traffic, routing, static power, input encoding, and other costs.

### Track B: Does synaptic balance lead to or favor excitation/inhibition balance?

**Question:** In a network with explicitly defined excitatory and inhibitory populations, does applying synaptic balancing change or promote a measurable E/I balance statistic?

This is a separate conceptual problem. Synaptic balance equates a neuron's incoming and outgoing weight costs. E/I balance usually concerns excitatory and inhibitory input currents, conductances, activities, or their relationship over time. Equal sums of squared weights do not imply equal excitatory and inhibitory currents.

A meaningful project would need:

- A specified spiking network model with explicit excitatory and inhibitory synapses or cell populations.
- A clear operational definition of E/I balance, such as separately measured excitatory and inhibitory currents or conductances and their time-dependent statistics.
- Explicit sign constraints, ideally a model consistent with Dale's law if biological interpretation is intended.
- Measurements before and after balancing, under matched activity and task conditions.
- Controls for network architecture, inhibition strength, thresholds, input drive, and random seeds.

This is potentially valuable but likely a separate study, because it requires a biologically meaningful model and careful measurement rather than simply using the same incoming/outgoing norm ratio.

### Track C: What happens when dropout and balancing are interleaved?

**Question:** How do dropout and balancing interact during training, and does that interaction affect the activity, accuracy, or energy proxy of the converted SNN?

Why it is promising:

- It is an explicit suggestion from the advisor.
- It can be tested initially in an ordinary ReLU ANN before adding SNN conversion complexity.
- It supports a clean factorial experiment: dropout off/on crossed with balancing off/on.
- SNN conversion and SynOps measurements let the study connect training dynamics to neuromorphic inference.

Why novelty is not automatic:

- Dropout is a standard method.
- Synaptic balance is already studied on its own.
- Sorbaro et al. (2020), a key SNN energy paper, already uses dropout in its experimental pipelines: 50% dropout before the fully connected layer for MNIST-DVS, and dropout in the CIFAR-10 setup. Thus, do not claim the first combination of dropout and energy-efficient SNN methods without a careful literature review. The contribution needs to be specifically about **how dropout placement/rate/schedule interacts with synaptic balancing**, and what this does to the balance-energy-accuracy relationship.

### Recommended priority

Start with Track C as the manageable initial experiment and track its effects on the energy metrics in Track A. Keep Track B as a separate follow-up or parallel literature question, not a required component of the first paper.

This prioritization is a recommendation based on feasibility, not a decision made by the advisor. The advisor has not endorsed a specific final experimental design yet.

---

## 5. Working research formulation

### Draft working title

**Interleaving Synaptic Balance and Dropout for Efficient Spiking Neural Network Inference**

Alternative title if results are more descriptive than algorithmic:

**On the Interaction Between Synaptic Neural Balance, Dropout, and Synaptic Operations in Converted Spiking Networks**

Avoid putting “energy-efficient” in the title unless the result really improves an energy-related measure and the evaluation supports the claim. If only simulated SynOps are measured, say SynOps or computational efficiency rather than claiming direct hardware energy savings.

### Primary research question

Does applying synaptic neural balancing before or during training, alone and in combination with dropout, improve the accuracy-versus-SynOps trade-off after conversion to an SNN, relative to established scaling and energy-aware training baselines?

### Initial hypotheses to test

- **H1:** Balancing changes the distribution of hidden-unit weight scales and may change firing rates or SynOps after conversion.
- **H2:** The interaction between dropout and balancing is not necessarily additive. Balancing may help, hurt, or have no measurable effect on dropout-trained models.
- **H3:** Any observed benefit may depend on threshold calibration, quantization, simulation duration, network architecture, or where dropout is applied.
- **H4:** A benefit in floating-point operation counts may disappear after standard SNN calibration or on hardware that does not exploit event sparsity.

These hypotheses are deliberately falsifiable. Do not write them as established facts.

### Main experimental output

A set of accuracy-versus-SynOps curves and a table that includes conventional ANN performance, converted-SNN performance, spikes/SynOps, firing-rate summaries, quantization settings, and (if available) hardware-measured energy/latency.

A publishable contribution should be a new, reproducible finding or method, not merely evidence that the transformations run.

---

## 6. Experimental plan

### Phase 0: Literature review and scope freeze

Before coding the full study:

1. Read the exact versions listed in the literature section below, including supplementary information and official implementations where available.
2. Write down the exact baseline algorithm, network architecture, SNN neuron dynamics, calibration method, SynOps definition, datasets, seeds, and dropout settings in Sorbaro et al.
3. Look for published work combining weight balancing with dropout, event-driven inference, spiking networks, and energy-aware optimization.
4. Check both the original Baldi paper and the published journal version for any differences relevant to Section 12.4.
5. Create a one-page “gap statement” that names what the closest existing papers already show and what remains untested.
6. Do not claim originality until the literature search supports it.

Deliverable: a related-work matrix with paper, method, dataset, balancing/scaling approach, dropout use, SNN conversion, energy metric, hardware used, main result, and the gap left open.

### Phase 1: Build and verify the balance operation

Start with a small ReLU network in PyTorch. MNIST can be used as a simple implementation test, but it should not be the only benchmark for a final SNN efficiency paper.

For each hidden neuron, calculate the incoming and outgoing costs. For the L2 case, apply the transform with \(\lambda=(B/A)^{1/4}\): incoming weights and the neuron's incoming bias are scaled by \(\lambda\); all outgoing weights from that neuron are scaled by \(1/\lambda\). Carefully handle shared weights, branches, and architecture-specific parameters.

Verification requirements:

- Original and balanced ANN outputs agree to numerical tolerance on random inputs and real validation examples before quantization.
- The intended balance-deficit statistic improves after balancing.
- Zero or near-zero incoming/outgoing costs are handled safely; do not divide by zero or let an arbitrary epsilon dominate without documentation.
- For convolution, define whether balancing operates per channel, filter, spatial neuron, or some other group. Do not casually apply the fully connected formula to a convolutional tensor without deriving the correct transformation.
- Inspect how biases and any normalization layers are treated. Batch normalization, residual connections, tied weights, and non-homogeneous activations can invalidate naive assumptions.
- Save before/after weights and regression tests.

Use the official Neural-Balance code as a starting point: https://github.com/antonyalexos/Neural-Balance . First inspect its license, README, implementation, versions, and reproduction status. Do not assume this repo implements the desired SNN experiments; it supports the original balancing research.

### Phase 2: Reproduce a known SNN energy baseline

Sorbaro et al. (2020) is the key direct baseline. Its paper and implementation are important because the study uses both MNIST-DVS and CIFAR-10, discusses weight scaling and robust layer-wise normalization, and explicitly optimizes a SynOp loss with activation quantization. It uses Sinabs for spiking simulations.

Paper: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full  
Code linked from the paper: https://gitlab.com/aiCTX/synoploss

Suggested progression:

1. Establish a small working ANN-to-SNN conversion pipeline.
2. Reproduce at least one known scaling baseline and its SynOps calculation.
3. If possible, reproduce a small part of Sorbaro's reported accuracy-SynOps trade-off.
4. Record framework/library versions, hardware, seeds, and any differences from the paper.

MNIST-DVS is a good first research benchmark because it was used by Sorbaro et al. and represents genuine event-camera data. For reproducibility, note their preprocessing: event recordings are divided into chunks of 3,000 events; training uses accumulated event frames; event polarity is discarded; the test SNN receives event frames at 1 ms resolution; and the network state is reset between chunks. The paper's MNIST-DVS network used 50% dropout before the fully connected layer during training. In its CIFAR-10 setup, the input layer uses dropout/augmentation and convolutional layers use dropout. Exact paper details should be checked in the Methods section before trying to reproduce the reference result.

CIFAR-10 is a second benchmark to test generality after the pipeline works. It is more complex and may expose different interactions between model width, activation sparsity, and balancing.

### Phase 3: Run the core factorial experiment

Start with a 2-by-2 design:

| Group | Dropout | Synaptic balancing | Purpose |
|---|---|---|---|
| A | Off | Off | Base network |
| B | On | Off | Dropout effect alone |
| C | Off | On | Balancing effect alone |
| D | On | On | Interaction between the two |

Use the same architecture, data splits, optimizer, training budget, and random seeds. Evaluate at least these balancing schedules:

- No balancing.
- Full balance before training.
- Partial balancing interleaved with training, such as once per epoch, matching the type of schedule investigated in the Baldi/Alexos work.
- Consider post-training balance as an additional diagnostic because it preserves the conventional ANN function but may change SNN conversion behavior.

For dropout, start with the architecture's standard rate/location and one or two justified alternatives rather than testing an enormous grid. Dropout is disabled at ordinary inference unless deliberately being studied as an inference-time stochastic method. The principal effect in this project is expected to be on the weights learned during training, then on the converted SNN.

Important details:

- Balance the full model parameters at a specified point in training, not the temporarily masked set of connections of one dropout realization, unless that is deliberately a separate method.
- Standard dropout acts on activations during training. Scaling a ReLU neuron's activations and inversely scaling outgoing weights can commute with a fixed multiplicative mask under simple assumptions, so the novelty must be in the changed optimization trajectory and downstream SNN behavior, not a claim that dropout mathematically breaks all scale equivalence.
- Explicitly state whether balancing occurs before optimizer steps, after optimizer steps, at epoch boundaries, or after the final epoch. These timings are not interchangeable.
- Tune hyperparameters on the validation set only. Reserve the test set for the final comparison.

### Phase 4: SNN conversion and threshold/calibration controls

Convert each trained ANN using the same SNN model, conversion procedure, time horizon, encoding, threshold calibration, and reset behavior. Document whether weights and biases are transferred directly, whether weights are quantized, and how layer thresholds are selected.

This phase has a major confound: **balancing is function-preserving in an ideal ReLU ANN, but it changes internal scales. SNN conversion with finite thresholds and time steps can respond differently to those internal scales.**

Use at least two threshold/calibration conditions:

1. **Fixed or constrained calibration:** keep a specified threshold procedure fixed across models, or simulate a deployment setting where per-neuron recalibration is expensive or unavailable.
2. **Standard recalibration:** apply the same calibration procedure independently to each model using only training/calibration data.

Interpretation:

- If balancing helps only when thresholds are fixed, report it as a benefit under that constraint, not a universal improvement.
- If recalibration removes the effect, balancing may be redundant with calibration for that architecture.
- If balancing improves operation counts but damages accuracy, report the Pareto trade-off rather than focusing on the lowest SynOps point.
- Check that output-layer decisions, biases, threshold scaling, membrane initialization, and reset rules are handled consistently.

A useful negative control is random function-preserving rescaling of hidden units. This can determine whether the effect is particular to balancing or to rescaling more generally. Do not select random factors after seeing the test results.

### Phase 5: Efficiency measurement

**Primary algorithmic metric:** SynOps per example and accuracy-SynOps Pareto curve.

For a layer, a common approximation is output spike count multiplied by synaptic fan-out. Use the same definition across every method. NeuroBench distinguishes dense SynOps from effective operations that account for nonzero activations/weights, and separates MACs from accumulates where appropriate. Document which form is used.

Recommended measurements:

- Test accuracy and, where appropriate, per-class metrics.
- SynOps per sample and per simulation window.
- Spike count/rate per layer and firing-rate distribution.
- Silent-neuron fraction and activity sparsity.
- Accuracy as a function of SNN simulation duration.
- Weight range and quantization error if quantization is tested.
- Inference wall-clock latency, number of parameters, and memory usage.
- Training runtime and additional balancing overhead.
- Measured or estimated energy only with a documented method and hardware-specific assumptions.

Do not directly call a SynOps reduction a measured energy reduction. It is an energy-related operation-count proxy. If no physical neuromorphic hardware or validated energy model is available, state that the work measures operations and estimates efficiency, not joules.

If actual hardware is available, use batch size 1 or another clearly specified workload, warm up the system, isolate host and device overhead, log idle/active power as appropriate, and repeat measurements. Check whether input encoding, communication, memory movement, and static power dominate. The balancing operation's energy cost matters if it is performed on-device; if performed offline before deployment, it is a one-time cost that should be reported separately from repeated inference cost.

### Phase 6: Hardware constraint experiment

Only add this once the core pipeline works. Choose one constraint at a time:

- Low-precision weights (e.g., 8-bit) with a documented quantization method.
- Fixed or restricted thresholds.
- Limited calibration data.
- A specific fan-out/memory constraint if the selected hardware motivates it.

Compare against a suitable existing quantization/rescaling method if claiming a benefit from balancing under quantization. Cross-layer equalization is an important related method in conventional quantization research. Do not add several constraints simultaneously until single-factor effects are understood.

### Phase 7: Optional E/I experiment

Treat this as a new branch, not a box to check in the primary paper.

Build or select an SNN with explicit excitatory and inhibitory neuron populations and synapse sign constraints. Measure excitatory and inhibitory currents/conductances over time and define the target E/I statistic before running experiments. Then assess whether balancing changes it relative to controls.

Do not equate \(\sum w_{in}^2 = \sum w_{out}^2\) with E/I balance. Squared weights erase sign and do not reflect incoming activity rates or conductance dynamics. You may need to preserve Dale's law and separately analyze recurrent versus feed-forward connections.

---

## 7. Experimental baselines that should be compared

At minimum, the paper needs controls that isolate the contributions of balancing, dropout, calibration, and SNN-specific energy optimization.

1. **Plain ANN-to-SNN conversion:** no synaptic balancing and no extra energy objective.
2. **Dropout only:** standard dropout, same rates/locations as the balancing-plus-dropout method.
3. **Balancing only:** pre-training and/or interleaved balancing, with dropout disabled.
4. **Balancing plus dropout:** proposed combination.
5. **Established SNN weight/threshold scaling:** follow the relevant conversion baseline and Sorbaro's scaling baseline.
6. **SynOp-loss + quantization-aware training:** direct energy-related baseline from Sorbaro et al. (2020), subject to reproduction feasibility.
7. **Random function-preserving rescaling:** diagnostic control for the possibility that arbitrary internal rescaling, not balance specifically, drives the outcome.
8. **Optional standard post-training quantization/equalization:** required if making claims about low-precision hardware advantages.

Keep model architecture and train/evaluation budgets aligned wherever possible. Do not compare your method's best hyperparameter configuration against an under-tuned baseline.

---

## 8. Critical caveats and likely failure modes

### A. Synaptic balance is not excitation/inhibition balance

They are distinct definitions. Incoming and outgoing weight costs ignore synapse signs and activity. E/I analysis needs explicit excitatory/inhibitory currents or related dynamical measures.

### B. Function-preserving ANN scaling may not preserve SNN behavior

The SNN has thresholds, finite time, reset dynamics, spike discretization, and potentially quantized weights. Changes in layer or neuron scale can alter firing rates and accuracy. This is both the project's opportunity and its major confound.

### C. Threshold calibration may erase the purported benefit

A scale change can alter a neuron's input magnitude. If thresholds are scaled or recalibrated accordingly, activity may return to a similar regime and energy gains may vanish. That would be a useful result. Compare fixed and standardized recalibrated conditions.

### D. Prior SNN energy work already uses dropout

Sorbaro et al. (2020) explicitly use dropout in their MNIST-DVS and CIFAR-10 experimental pipelines. The novelty cannot be “the first to combine dropout with SNN energy optimization.” It must identify an untested, precise interaction involving synaptic balance.

### E. Low SynOps is not automatically low energy

Hardware may pay for memory, neuron state updates, routing, input encoding, control, and static power. SynOps are a useful proxy but not a universal energy metric. State the scope of any energy claim precisely.

### F. A simulation on CPU/GPU is not a neuromorphic-hardware measurement

A software SNN simulator measures algorithm behavior and operation counts. It does not establish energy used by a physical neuromorphic chip. Real-hardware claims require access and a sound measurement protocol, or a validated hardware-specific energy model.

### G. Balance may have no benefit, or may hurt

Balancing can change SNN spike rates and quantization behavior, but it could also increase activity in some layers, hurt accuracy, have no measurable effect after calibration, or add more preprocessing/training cost than its benefit. Do not selectively report only favorable layers or runs.

### H. Dropout has several definitions and placements

Input dropout, hidden-unit dropout, channel dropout, and DropConnect are different interventions. Sorbaro's settings are not the same across datasets. Specify rates, placement, inverted-dropout convention, and whether masks are used only during training.

### I. Balancing with biases and complex architectures is error-prone

A ReLU neuron with a bias generally requires its incoming affine parameters to be transformed consistently. All outgoing edges need inverse scaling. Batch normalization, layer normalization, skip/residual connections, tied weights, convolutional channels, recurrent connections, and output layers require careful analysis. Begin with a simple feed-forward architecture and expand only after invariance tests pass.

### J. Cost function and balancing definition must be fixed

Balancing depends on the regularizer/cost definition. L1, L2, and other Lp choices are not interchangeable. Use one definition in the first experiment and explain its mathematical and hardware rationale.

### K. Architecture-dependent behavior

A result on a small fully connected ReLU network may not extend to convolutional SNNs, recurrent SNNs, or hardware-specific network mappings. Validate at least one more architecture or dataset before making broad claims.

### L. Dataset and evaluation caveats

MNIST is useful for debugging but likely too easy by itself for a strong efficiency paper. MNIST-DVS and CIFAR-10 connect to the direct baseline literature. Use official splits, keep validation separate from test, and ensure identical examples/evaluation time horizons across methods.

### M. Statistical variance

SNN conversion and training can be sensitive to seeds, data order, calibration samples, thresholds, and time horizons. Use multiple paired random seeds. Five seeds can be a pilot; final sample size should be based on variability and available compute. Report confidence intervals or another appropriate uncertainty estimate.

### N. Training efficiency versus inference efficiency

These are different questions. Balancing during training has a computational cost. If weights are balanced offline, inference doesn't execute the balancing operation. Report training runtime/overhead separately from repeated inference cost and explain any amortization assumption.

### O. Do not confuse a null result with a failed project

If balancing adds nothing after calibration, that helps delimit the claim in Section 12.4. A well-controlled result showing when it does and does not help can be meaningful, provided the question is relevant and comparisons are strong.

---

## 9. Metrics, analysis, and figure plan

### Pre-specify primary outcomes

- **Primary:** accuracy versus SynOps, with common conversion settings.
- **Secondary:** layer-wise spike counts/rates, silent neuron percentage, activity sparsity, accuracy at varying simulation duration, and weight/threshold dynamic ranges.
- **System costs:** inference latency, memory, and training/preprocessing time.
- **Energy:** physical measurement or clearly labeled estimate only if justified.

### Balance metric

Record the original paper's style of balance deficit and optionally a normalized measure. For example, the per-neuron raw L2 deficit can be based on \((A_j-B_j)^2\). A normalized measure such as

\[
d_j=\frac{|A_j-B_j|}{A_j+B_j+\epsilon}
\]

is often easier to compare across neurons but is an additional analysis choice, not the paper's exact definition. Report which one is primary and do sensitivity checks on the epsilon or low-norm cases.

### Suggested figures

1. **Balance verification:** histograms of per-neuron balance deficit before and after the transformation.
2. **ANN function preservation:** maximum absolute output difference before and after balancing.
3. **Factorial interaction plot:** accuracy and activity for dropout on/off crossed with balancing on/off.
4. **Accuracy vs SynOps Pareto curves:** with standard scaling and SynOp-loss baselines.
5. **Layer-wise activity analysis:** spike count/rate per layer, not only aggregate totals.
6. **Threshold sensitivity:** compare fixed versus recalibrated thresholds.
7. **Rate/quantization sensitivity:** if included, show one constraint at a time.
8. **Multiple-seed uncertainty:** show variation across runs instead of cherry-picked exemplars.

Avoid comparing heatmaps/plots that use different axes or color scales. State all conventions explicitly.

---

## 10. Reproducibility and implementation requirements

- Pin Python and library versions; save a lockfile/environment file.
- Use fixed random seeds and record hardware, OS, CUDA, and framework versions.
- Separate train, validation, calibration, and test data roles.
- Save model weights and all conversion/calibration settings.
- Automate the full evaluation script so every method is evaluated identically.
- Log model architecture, dropout position/rate, balancing interval, balancing norm, threshold calibration, quantization, time steps, dataset preprocessing, accuracy, SynOps, latency, and any energy estimate.
- Do not alter the test protocol after examining final results.
- Release code/configs and record differences from each baseline paper.
- Verify that the balance operation is applied to the intended parameters in the intended order. A bug in channel mapping or bias scaling can create apparent effects that are not scientific.

Suggested experiment record columns:

`method`, `seed`, `dataset`, `architecture`, `dropout_type`, `dropout_rate`, `balance_schedule`, `balance_norm`, `threshold_mode`, `quantization`, `inference_steps`, `test_accuracy`, `SynOps_per_sample`, `spike_count_by_layer`, `silent_fraction_by_layer`, `latency`, `parameter_count`, `training_time`, `balance_overhead`, `energy_value`, `energy_method`.

---

## 11. Hardware path and what can be claimed without hardware

### Software-first route

Begin with Python/PyTorch plus an SNN simulation framework. Sorbaro used Sinabs. Other frameworks may be considered, but choose one, lock versions, and ensure the operation-count convention is stable. NeuroBench offers hardware-independent and hardware-dependent benchmarking concepts and standardized activity/SynOps metrics.

This stage can support claims such as:

- Balancing changes layer-wise spike counts in the specified simulator.
- Balancing changes the simulated SNN accuracy-SynOps trade-off.
- The effect depends on threshold calibration, dropout, or quantization.

It cannot alone support claims such as:

- A physical neuromorphic chip uses X% less energy.
- A method is universally more energy efficient across hardware.

### Optional physical-hardware validation

If a suitable platform becomes available through the lab or advisor's graduate-student connection, validate a small number of models on actual hardware. Intel's Lava documentation describes software development for CPU/GPU and hardware paths; Loihi research hardware is not simply a generally purchasable commodity and access may depend on the Intel Neuromorphic Research Community or institutional access. Verify current access and supported functions before planning a project around Loihi.

Measure device power/energy according to the platform's supported profiler or measurement methodology. Report workload, batch size, warm-up, idle baseline, inference window, host-vs-device scope, and repeat count.

### Hardware-aware variant

If there is no physical access, an explicitly hardware-constrained evaluation can still be defensible: choose a known model of energy per event, add a documented weight-precision constraint, model memory/storage costs, and run a sensitivity analysis. Call it modeled energy or an energy proxy and state the assumptions.

---

## 12. Literature that must be read and compared against

This list is prioritized. It is not a claim that each paper is an exact competitor on every task. The related-work review should map which research question each paper supports and which methods it provides as a baseline.

### Tier 1: Essential to the core project

#### 1. Baldi et al. (2024 arXiv; published 2025), *A Theory of Synaptic Neural Balance: From Local to Global Order*

- arXiv: https://arxiv.org/abs/2405.09688
- Journal DOI: https://doi.org/10.1016/j.artint.2025.104360
- Why: Foundational definition, scaling/balancing operations, convergence, limitations, Sections 12.2 to 12.4.
- Required action: Read the published version as well as the arXiv PDF. Identify exactly which claims are theorem-backed and which are hypotheses for future neuromorphic work.

#### 2. Alexos, Domingo, and Baldi (2025), *Improving Deep Learning Speed and Performance Through Synaptic Neural Balance*, AAAI 2025

- Official proceedings: https://ojs.aaai.org/index.php/AAAI/article/view/33683
- DOI: https://doi.org/10.1609/aaai.v39i15.33683
- Code: https://github.com/antonyalexos/Neural-Balance
- Why: Already experimentally tests full balancing before training and partial balancing interleaved with SGD; reports training speed, accuracy, and data-scarce settings. These are not new contributions for the proposed SNN paper.
- Required action: Understand implementation and reproduce at least one relevant method or result.

#### 3. Sorbaro, Liu, Bortone, and Sheik (2020), *Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications*, Frontiers in Neuroscience

- DOI: https://doi.org/10.3389/fnins.2020.00662
- Full text: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full
- Code linked by paper: https://gitlab.com/aiCTX/synoploss
- Why: Closest direct baseline for SNN energy-related optimization. It studies MNIST-DVS and CIFAR-10, scaling, SynOp loss, activation quantization/QReLU, and Sinabs SNN simulation. It explicitly uses dropout in its experimental models.
- Required action: Record the exact architecture, dropout placement/rate, conversion, time steps, SynOps formula, threshold/weight scaling and calibration protocol. The proposed paper must compare to appropriate parts of this work and distinguish its novelty from the SynOp-loss/quantization approach.

#### 4. Rueckauer et al. (2017), *Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification*

- Frontiers: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00682/full
- DOI: https://doi.org/10.3389/fnins.2017.00682
- Why: Standard ANN-to-SNN conversion and robust layer-wise scaling/normalization baseline.
- Required action: Understand threshold and activation scaling and how these choices interact with neuron-wise balance.

#### 5. NeuroBench (2025), *The NeuroBench Framework for Benchmarking Neuromorphic Computing Algorithms and Systems*

- Nature Communications: https://www.nature.com/articles/s41467-025-56739-4
- Synaptic operation metric documentation: https://neurobench.readthedocs.io/en/latest/metrics/workload_metrics/synaptic_operations.html
- Why: Provides guidance for reproducible, fair neuromorphic algorithm and system benchmarking. It distinguishes dense operations from effective operations, and MACs from accumulates where applicable.
- Required action: Align metric definitions and reporting with appropriate NeuroBench conventions where possible.

### Tier 2: Essential to the dropout and balance interaction claim

#### 6. Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR

- Paper: https://www.jmlr.org/papers/v15/srivastava14a.html
- Why: Standard dropout definition, training/inference semantics, and baseline.
- Required action: State exactly which dropout form and location is used.

#### 7. Baldi and Sadowski (2014), *The Dropout Learning Algorithm*, Artificial Intelligence

- DOI: https://doi.org/10.1016/j.artint.2014.02.004
- Why: Mathematical analysis of dropout, including stochastic gating and effects in different network classes. Especially relevant given the same research group's interest in balancing as a training/regularization operation.
- Required action: Read for possible theoretical intuition and accurate claims about the interaction. Do not assume the mathematical interaction has already been solved by the paper.

#### 8. Salehinejad and Valaee (online 2021; journal 2022), *EDropout: Energy-Based Dropout and Pruning of Deep Neural Networks*, IEEE Transactions on Neural Networks and Learning Systems

- PubMed record: https://pubmed.ncbi.nlm.nih.gov/33830931/
- DOI: https://doi.org/10.1109/TNNLS.2021.3069970
- Why: Energy-related dropout/pruning is relevant if the proposed novelty is an interaction between dropout, balancing, and operation cost. It is not necessarily an SNN-specific baseline, so use it for related work and only compare experimentally where the task is appropriate.

#### 9. Sorbaro et al. (2020), again, especially the dropout details

The dropout use in this paper is a critical novelty caveat. For MNIST-DVS, it uses 50% dropout before the fully connected layer during training. For CIFAR-10, its setup includes dropout in the convolutional network and input augmentation/dropout. Therefore, a claim that no one has studied dropout with energy-optimized SNN conversion would be inaccurate.

### Tier 3: Important if studying biologically motivated synaptic balance or E/I balance

#### 10. Stock, Harvey, Ocko, and Ganguli (2022), *Synaptic Balancing: A Biologically Plausible Local Learning Rule That Provably Increases Neural Network Noise Robustness Without Sacrificing Task Performance*, PLOS Computational Biology

- DOI: https://doi.org/10.1371/journal.pcbi.1010418
- Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC9522011/
- Why: A distinct synaptic balancing method in nonlinear recurrent networks, focused on robustness and local plasticity. It is not identical to Baldi et al.'s balancing algorithm, so distinguish definitions and goals carefully.

#### 11. van Vreeswijk and Sompolinsky (1996), *Chaos in Neuronal Networks with Balanced Excitatory and Inhibitory Activity*, Science

- DOI: https://doi.org/10.1126/science.274.5293.1724
- Why: Foundational E/I-balanced network dynamics. Relevant only if an E/I study is actually included.

#### 12. Froemke (2015), *Plasticity of Cortical Excitatory-Inhibitory Balance*, Annual Review of Neuroscience

- DOI: https://doi.org/10.1146/annurev-neuro-071714-034002
- Why: Neuroscience perspective on the biological concept. Useful for defining what E/I balance does and does not mean.

#### 13. Field et al. (2020), *Heterosynaptic Plasticity Determines the Set Point for Cortical Excitatory-Inhibitory Balance*, Neuron

- Why: Relevant to homeostatic mechanisms and the relation between synaptic changes and E/I balance. Verify the complete citation and DOI during the formal literature review.

### Tier 4: Hardware constraints and quantization, if that becomes part of the paper

#### 14. Ji et al. (2016), *NEUTRAMS: Neural Network Transformation and Co-design under Neuromorphic Hardware Constraints*, MICRO

- PDF: https://pacman.cs.tsinghua.edu.cn/~cwg/papers_cwg/micro16.pdf
- Why: Hardware-aware neural network transformation, mapping, and co-design. Relevant if the proposed method is presented as hardware-constraint-aware rather than a purely algorithmic rescaling method.

#### 15. Nagel et al. (2019), *Data-Free Quantization Through Weight Equalization and Bias Correction*, ICCV

- Official open access: https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html
- Why: Related scaling/equalization for low-precision inference. Important if the paper claims that synaptic balancing improves quantized deployments.

#### 16. Rueckauer et al. (2017) and Sorbaro et al. (2020)

These are also the closest relevant scaling/normalization baselines for this part of the research.

#### 17. Current hardware energy measurement literature

As of 2026, also review newer papers that measure or model per-event energy across processors, neuromorphic chips, and in-memory computing. One current example surfaced in the initial search is *A Hardware-Grounded Energy Taxonomy for Comparing Deep and Spiking Neural Network Inference on Edge Platforms* (PMC article record: https://pmc.ncbi.nlm.nih.gov/articles/PMC13504384/). Verify its publication details, methodology, assumptions, and relevance before citing it. Use it as a prompt to update the energy-measurement literature, not as a substitute for primary hardware studies.

---

## 13. Practical next-step checklist

### First actions

- [ ] Download/read the published 2025 version of the synaptic neural balance paper, focusing on Sections 12.2, 12.3, 12.4 and the references to Sorbaro and Rueckauer.
- [ ] Read Alexos et al. (AAAI 2025) and inspect its official code.
- [ ] Read Sorbaro et al. (2020), its supplementary material if any, and the linked SynOp-loss implementation.
- [ ] Create a comparison spreadsheet for the papers in Section 12.
- [ ] Decide on an initial simple ReLU architecture and reproduce a baseline before adding proposed changes.
- [ ] Write unit tests showing ANN outputs stay the same after balancing.
- [ ] Verify an SNN conversion and SynOps counting pipeline before running the full matrix.
- [ ] Run a small 2x2 dropout-by-balancing pilot with paired seeds.
- [ ] Produce the first report: accuracy, balance deficit, spike counts/SynOps, threshold setting, and implementation notes.
- [ ] After obtaining a concrete result, prepare a concise update for Professor Baldi and ask whether a graduate-student connection or design feedback would be appropriate.

### Do not do yet

- Do not claim physical energy savings before measuring energy or using a validated hardware-specific model.
- Do not claim that synaptic balance implies E/I balance.
- Do not claim dropout plus balancing is novel without checking Sorbaro and the broader literature.
- Do not start with a full biologically detailed E/I model and neuromorphic hardware simultaneously.
- Do not build every possible architecture, dropout rate, regularizer, quantization scheme, and threshold policy into the first experiment.
- Do not optimize experimental thresholds against the test set.
- Do not write the paper around positive results before knowing whether the experiment works.

---

## 14. What would count as a strong result?

A convincing result would do more than show that balancing changes internal neuron norms.

Potential strong contributions include:

1. **Empirical characterization:** establishing the conditions under which balance does and does not improve accuracy-SynOps trade-offs after ANN-to-SNN conversion.
2. **A methodological contribution:** a balancing schedule or combined dropout-balancing method that improves the trade-off over well-tuned baselines, with multiple seeds and consistent results across datasets or architectures.
3. **An insight into calibration:** demonstrating that the effect is explained by or survives threshold calibration, and identifying when balance gives something beyond existing scaling methods.
4. **A hardware-specific result:** demonstrating a meaningful advantage on a specific constrained neuromorphic platform, with measured energy/latency and a documented accounting method.
5. **E/I finding:** if studied separately, showing a reproducible relation between synaptic balance and a well-defined E/I dynamical statistic in a suitable model, rather than equating the two definitions.

A null result can be useful if it shows, for example, that apparent gains vanish after standard calibration or are an artifact of particular rescaling. The paper would need to frame this as a careful empirical characterization and use strong baselines.

---

## 15. Final recommended strategy

**Start with one small, controlled experiment, and connect it to the advisor's questions without trying to answer all of them at once.**

1. Verify synaptic balance on a small ReLU network.
2. Reproduce a standard ANN-to-SNN conversion and its operation-count accounting.
3. Cross dropout off/on with balance off/on, and measure the learned models' downstream SNN accuracy-SynOps trade-off.
4. Compare against established scaling and SynOp-loss/quantization baselines, using Sorbaro et al. as the central reference.
5. Examine threshold calibration as a likely confound.
6. Add a second dataset and, if the results justify it, one hardware constraint.
7. Treat E/I balance as a separate follow-up unless a specific model makes it tractable.
8. Share reproducible preliminary results with the advisor. His offer to introduce the student to a graduate student is contingent on making progress, so the most useful next contact is one backed by code, plots, and a clearly stated question.

The guiding principle is: **measure first, establish the baseline, isolate the effect, then make the claim.** The Section 12.4 claim is a motivation and hypothesis. The research contribution must be an experimentally justified answer to when and why synaptic balance does or does not help.

---

## 16. Sources and online references

The URLs are grouped by topic above; the main entry points are collected here for convenience:

- Baldi et al., *A Theory of Synaptic Neural Balance*: https://arxiv.org/abs/2405.09688
- Published version, *Artificial Intelligence* (2025): https://doi.org/10.1016/j.artint.2025.104360
- Alexos et al., AAAI 2025 balancing paper: https://ojs.aaai.org/index.php/AAAI/article/view/33683
- Official Neural-Balance repository: https://github.com/antonyalexos/Neural-Balance
- Sorbaro et al., SNN energy optimization: https://doi.org/10.3389/fnins.2020.00662
- Sorbaro et al. code: https://gitlab.com/aiCTX/synoploss
- Rueckauer et al., ANN-to-SNN conversion: https://doi.org/10.3389/fnins.2017.00682
- NeuroBench paper: https://www.nature.com/articles/s41467-025-56739-4
- NeuroBench SynOps definition: https://neurobench.readthedocs.io/en/latest/metrics/workload_metrics/synaptic_operations.html
- Srivastava et al., dropout: https://www.jmlr.org/papers/v15/srivastava14a.html
- Baldi and Sadowski, dropout learning algorithm: https://doi.org/10.1016/j.artint.2014.02.004
- EDropout: https://doi.org/10.1109/TNNLS.2021.3069970
- Stock et al., biologically plausible synaptic balancing: https://doi.org/10.1371/journal.pcbi.1010418
- NEUTRAMS: https://pacman.cs.tsinghua.edu.cn/~cwg/papers_cwg/micro16.pdf
- Quantization by weight equalization: https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html
- Lava documentation: https://lava-nc.org/developer_guide.html
