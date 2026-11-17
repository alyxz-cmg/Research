# Research Handoff: Synaptic Neural Balance in Convolutional Neural Networks

**Purpose:** Give a future agent or research collaborator enough context to continue this project without reconstructing the conversation.
**Last updated:** 2026-11-17
**Primary advisor:** Professor Pierre Baldi, UC Irvine.

---

## 1. Project History and the Pivot

### The Original Theory and Assumptions
This project builds on Baldi et al. (2025), *A Theory of Synaptic Neural Balance: From Local to Global Order*. For neural networks with positively homogeneous activation functions ($f(\lambda x) = \lambda f(x)$ for $\lambda > 0$, such as ReLU) and compatible graph topology, multiplying a hidden unit's incoming weights and bias by $\lambda > 0$ while dividing its outgoing weights by $\lambda$ preserves the exact input-output mapping of the network. While balancing alone does not alter immediate training or test accuracy at the moment of transformation, interleaving it with gradient descent acts as a function-preserving optimization intervention that reshapes subsequent gradient steps.

### Completed Preliminary Direction: Two-Hidden-Layer MLP & SNN Proxy Pilot
Based on Section 12.4 of Baldi et al. (2025), we completed a controlled $2 \times 2$ factorial pilot (5 paired seeds: `42–46`) on a two-hidden-layer fully connected `784-256-128-10` ReLU network trained on MNIST for 60 epochs with stateless SGD (`lr = 0.001`), followed by ANN-to-SNN conversion:
1. **ANN Optimization Interaction:** Without balancing, dropout reduced 60-epoch test accuracy by `-1.34` percentage points (`89.66%` to `88.32%`). When interleaved with once-per-epoch partial balancing, the dropout penalty shrank to `-0.20` percentage points (`90.85%` to `90.65%`), yielding a positive factorial interaction ($\Delta_{\text{interaction}} = +1.146$ percentage points, 95% CI `[+0.88, +1.41]` pp, $p = 0.00023$). Interleaved balancing also accelerated early validation accuracy (at Epoch 10, `D - B = +18.10` pp) and converged to a normalized balance deficit of $\approx 10^{-6}$ with all 384 hidden neurons remaining balance-eligible.
2. **SNN Software Operation-Count Proxy Results:** Under tied layer-wise recalibrated thresholds at $T = 50$ timesteps, the combined dropout + balance condition (`Group D`) achieved a significant factorial interaction in hidden event-count SynOps/sample ($\Delta_{\text{interaction}} = -17,202.26$ hidden SynOps/sample, 95% CI `[-22,980.65, -11,423.87]`, $p = 0.001169$), reducing hidden operations by `-6,856.84` SynOps/sample (`-2.83%`) relative to dropout alone (`Group B`) while improving converted SNN test accuracy by `+2.29` percentage points. *(Note: These are software operation-count proxies, not physical hardware energy measurements in joules.)*

### Near-Term Priority: Standard Convolutional Networks with Shared Weights
Note that Section 10 of Baldi et al. (2025) already evaluated CIFAR-10 using *locally connected* layers (without shared convolutional kernels). We are keeping the SNN conversion work frozen as a completed preliminary phase (until neuromorphic hardware access is available) and prioritizing a distinct ANN optimization question: **Does the synergistic interaction between interleaved synaptic balancing and dropout generalize to standard Convolutional Neural Networks (CNNs) with shared weights?**

---

## 2. Theoretical Challenge: CNN Weight Sharing, Pooling & Cost Domains

Section 8.3 of Baldi et al. (2025) warns that standard convolutional weight sharing complicates individual neuronal balancing because all spatial neurons in an output channel share the same kernel parameters. Furthermore, Appendix B proves convergence for tied layer balancing in standard layered feedforward networks, which does not automatically cover shared convolutional kernels, spatial pooling, or dropout without explicit formulation.

1. **Coordinated Layer Balancing (Primary Method, Section 8.3):** Apply a single shared multiplier $\lambda_\ell > 0$ to all weights $W^{(\ell)}$ and biases $b^{(\ell)}$ of hidden layer $\ell$, and scale the next layer's incoming weights $W^{(\ell+1)}$ by $1/\lambda_\ell$ (leaving $b^{(\ell+1)}$ unscaled).
2. **Pooling Commutativity Restriction:** While `MaxPool2d` is not invariant to arbitrary per-location scaling ($\max(\lambda_1 a_1, \lambda_2 a_2) \neq \lambda \max(a_1, a_2)$), it is strictly positively homogeneous whenever all spatial positions inside each pooling window share the same positive multiplier $\lambda > 0$. Both layer-wide scaling (`balance_mode="layer"`) and channel-wide scaling (`balance_mode="channel"`, kept as a separate secondary investigation) satisfy this condition.
3. **Parameter-Space $L_2$ vs. Connection-Graph Synaptic $L_2$ Cost Domains:**
   * **Unique-Parameter Domain (`cost_domain="parameter"`, primary baseline):** Defines $A_\ell^{\text{param}} = \|W^{(\ell)}\|_F^2 + \|b^{(\ell)}\|_2^2$ and $B_\ell^{\text{param}} = \|W^{(\ell+1)}\|_F^2$, minimizing the standard trainable-parameter $L_2$ regularizer.
   * **Connection-Graph Domain (`cost_domain="connection_graph"`, Section 8.2 Eq. 8.1):** Weights each shared kernel parameter by its exact spatial connection multiplicity $M_\ell^{\text{in}}(u,v)$ across valid receptive fields, accounting for spatial resolution changes across `MaxPool2d` and the transition to unshared `Linear` layers. Both deficits are logged in all experiments.

---

## 3. Research Hypotheses

* **H1 (CNN Interaction):** The synergistic interaction between interleaved synaptic balancing and dropout generalizes to standard convolutional networks with shared weights on CIFAR-10.
* **H2 (Optimal Dropout Rate):** Interleaved balancing shifts the optimal dropout probability in a CNN, tested via orthogonal 1D validation sweeps over convolutional and fully connected dropout rates.
* **H3 (Generalization Across Complexity):** The interaction between dropout and synaptic balancing generalizes across increasingly complex image-classification tasks and produces reproducible improvements in convergence speed, generalization gap, or final classification accuracy relative to either method independently.

---

## 4. Staged Execution Plan

1. **Stage 1 (Mathematical & Numerical Verification Gate):** Verify exact logit invariance in `float64` and `float32` (including active fixed-mask dropout), monotonic regularizer reduction in both `parameter` and `connection_graph` domains, explicit convergence flags, zero-cost unit accounting, and paired augmentation RNG isolation (`--stage 1`).
2. **Stage 2 (CIFAR-10 $2 \times 2$ Factorial Experiment):** Run a 1-seed smoke test (`seed=42`, 5 epochs), then a 5-seed pilot (`seeds=43..47`, 60 epochs) on a custom sequential VGG-style CNN (`32-64-Pool-128-128-Pool-256-10`), followed by a locked confirmatory run on 8 disjoint seeds (`seeds=48..55`).
3. **Stage 3 (Mechanistic Controls & H2 Sweep):** Compare `C` vs. `C_rand` and `D` vs. `D_rand` under matched random function-preserving rescaling, and run orthogonal 1D validation sweeps over dropout rates.
4. **Stage 4 (Architectural Scale-Up):** Extend to BatchNorm, residual skip connections, and CIFAR-100 / ImageNet subsets only after Stages 1–3 are established.