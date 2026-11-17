# Research Handoff: Synaptic Neural Balance in Convolutional Neural Networks

**Purpose:** Give a future agent or research collaborator enough context to continue this project without reconstructing the conversation.
**Last updated:** 2026-11-17
**Primary advisor:** Professor Pierre Baldi, UC Irvine.

---

## 1. Project History and the Pivot

### The Original Theory
This project stems from Baldi et al. (2025), *A Theory of Synaptic Neural Balance: From Local to Global Order*. The core concept is that homogeneous activation functions (like ReLU) permit a function-preserving parameter scaling. Specifically, multiplying incoming weights and biases by a factor $\lambda$ while dividing outgoing weights by $\lambda$ preserves the exact input-output mapping of the network. "Balancing" consists of optimizing these $\lambda$ factors (e.g., via the $L_2$ norm) so that the incoming and outgoing weight costs of each neuron are equal. 

### The MLP Proof-of-Concept (Completed)
Our initial plan was to test whether this balancing operation improved Spiking Neural Network (SNN) efficiency, based on a hypothesis in Section 12.4 of the theory paper. We completed a controlled 2x2 factorial pilot (5 seeds) on a fully connected `784-256-128-10` ReLU network trained on MNIST for 60 epochs.

The experiment was highly successful as a proof-of-concept, producing two main findings:
1.  **Optimization Interaction:** Balancing fundamentally changed the effect of dropout during standard ANN training. Dropout alone slowed learning and reduced test accuracy by -1.34 percentage points. When interleaved with balancing, this penalty shrank to just -0.20 percentage points (difference-in-differences $+1.15$ percentage points, $p=0.00023$).
2.  **SNN Operation Proxies:** The interaction carried over to the converted SNNs. The combined balance+dropout condition yielded the lowest hidden SynOps count at $T=50$ (recalibrated thresholds), demonstrating a statistically significant reduction compared to dropout alone while achieving higher test accuracy.

### The Pivot to CNNs
Following Professor Baldi's review, we are executing a strategic pivot. The MLP experiments on MNIST served their purpose by validating the mechanics, the dropout interaction, and the optimization trajectory changes. However, claiming a general optimization or energy benefit based solely on a simple, fully connected dataset is insufficient. 

Furthermore, true SNN energy efficiency claims require hardware-in-the-loop testing, not just software SynOp proxies. Therefore, **we are freezing the SNN conversion aspect as a completed preliminary investigation** and refocusing entirely on standard Artificial Neural Network (ANN) optimization in modern Convolutional Neural Networks (CNNs).

The new focus is testing whether the optimization and generalization benefits of the balance $\times$ dropout interaction survive the complexity of standard convolutional weight sharing on harder datasets like CIFAR-10.

---

## 2. Theoretical Challenge: CNN Weight Sharing

Applying synaptic balance to a CNN is not a trivial copy-paste from the MLP code. Section 8.3 of the original theory paper explicitly warns that standard convolutional weight sharing complicates balancing.

**The Problem:**
In a fully connected network, you can independently scale a single hidden neuron's incoming and outgoing weights. In a standard CNN, a single convolutional kernel is shared across all spatial locations in a feature map. You cannot balance a single spatial neuron independently without altering the shared kernel, which would disrupt the balance of all other spatial neurons using that same kernel.

**The Solution:**
The baseline approach for CNNs must be **Coordinated Layer Balancing**. Instead of balancing individual neurons, we must calculate a single shared scaling factor $\lambda$ that balances the entire convolutional layer (or filter channel), preserving the network's function by inversely scaling the subsequent layer's inputs. 

Any implementation must rigorously define:
1.  Which parameters are scaled (including biases).
2.  The exact mathematical definition of the incoming and outgoing costs under weight sharing.
3.  How to handle subsequent layers, particularly if the architecture includes Batch Normalization, residual skip connections, or max pooling.

---

## 3. New Research Hypotheses

*   **H1 (The CNN Interaction):** The synergistic interaction between interleaved synaptic balancing and dropout (where balancing accelerates learning and mitigates dropout's accuracy penalty) generalizes to standard convolutional networks with shared weights.
*   **H2 (Optimal Dropout):** Interleaved balancing shifts the optimal dropout probability in a CNN, allowing networks to benefit from stronger regularization without suffering severe convergence penalties.
*   **H3 (Generalization):** These effects persist across more challenging image classification datasets (e.g., CIFAR-10, CIFAR-100) compared to simple benchmarks like MNIST.

---

## 4. Staged Execution Plan

We will follow a strict, staged progression to ensure the mathematics are correct before running large-scale training.

### Stage 1: Mathematical Feasibility & CNN Validation
*   **Action:** Build a small, sequential CNN architecture. Derive the coordinated layer balancing (or channel-wise balancing) equations. 
*   **Validation:** Write numerical unit tests verifying that the CNN's logits remain identical (to float32 tolerance) before and after balancing, especially ensuring that biases and spatial pooling operations are handled correctly. 
*   **Gate:** Do *not* begin training experiments until the CNN balancing operation is mathematically proven and numerically verified.

### Stage 2: The Core CIFAR-10 Factorial Experiment
*   **Action:** Execute the 2x2 factorial design on CIFAR-10 using the validated sequential CNN.
    *   Group A: Baseline
    *   Group B: Dropout Only (at a fixed, pre-specified rate/location)
    *   Group C: Interleaved Balancing Only
    *   Group D: Dropout + Interleaved Balancing
*   **Controls:** Maintain paired seeds (5-8 seeds), identical data splits/augmentation, stateless SGD, and isolated random-number streams for initialization, dropout, and batch shuffling.
*   **Metrics:** Track training/validation learning curves, generalization gap, and final test accuracy to evaluate the $\Delta_{\text{interaction}}$.

### Stage 3: Mechanistic Controls & Optimal Dropout (H2)
*   **Action:** Introduce a matched random function-preserving rescaling control. This tests whether the performance gain is explicitly due to finding the "balanced" state, or simply a byproduct of arbitrary weight rescaling.
*   **Action:** Run a sweep of different dropout probabilities to test H2.

### Stage 4: Scale Up (H3)
*   **Action:** Only after Stages 1-3 yield robust results, extend the implementation to handle complex architectures (e.g., ResNet, which requires managing skip connections) and train on harder datasets like CIFAR-100 or an ImageNet subset.