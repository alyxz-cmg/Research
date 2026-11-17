### Research Handoff: Synergistic Regularization via Synaptic Balance and Dropout
**Purpose:** Give a future agent or research collaborator enough context to continue this project, reflecting the recent pivot from neuromorphic SNN efficiency to standard ANN/CNN accuracy and optimization dynamics.
**Last updated:** 2026-11-17
**Primary advisor:** Professor Pierre Baldi, UC Irvine. 

---

#### 1. Core Source Material
**Original Paper:** Pierre Baldi, Antonios Alexos, Ian Domingo, Alireza Rahmansetayesh. *A Theory of Synaptic Neural Balance: From Local to Global Order* (Artificial Intelligence, 2025).
*   **The Foundation:** The paper mathematically proves that scaling a neuron's incoming weights by $\lambda$ and outgoing weights by $1/\lambda$ strictly preserves a neural network's input-output function while altering the internal weight norms. 
*   **The Catalyst for this Project:** While Section 12.4 hypothesizes neuromorphic energy benefits, Sections 11 and 12.2 establish that balancing during training acts as a powerful regularizer that accelerates learning and improves test accuracy. Remark 9.11 explicitly notes that balancing can be interleaved with dropout. 

#### 2. The Original Plan (Neuromorphic Focus)
Initially, this project focused on **Section 12.4 (Neuromorphic Hardware)**. 
*   **Goal:** To determine if interleaving neuron-wise $L_2$ synaptic balancing with dropout during training improves the accuracy-versus-energy trade-off when the network is converted to a Spiking Neural Network (SNN).
*   **Metric:** We tracked Synaptic Operations (SynOps) as a software proxy for hardware energy.
*   **Hypothesis:** Balancing redistributes internal weight scales, which might synergize with dropout to reduce multi-spike bursting and create sparser, more efficient SNNs.

#### 3. The Experiment Performed
We designed a rigorously controlled $2 \times 2$ factorial pilot to isolate the interaction between balancing and dropout.
*   **Architecture:** A fully connected `784-256-128-10` ReLU Artificial Neural Network (ANN).
*   **Dataset:** Real MNIST (50k train, 5k val, 5k calib, 10k test).
*   **Training:** 60 epochs using stateless SGD (learning rate = 0.001) across 5 independent paired random seeds. 
*   **Conditions:**
    *   **Group A (Baseline):** Dropout Off, Balancing Off
    *   **Group B (Dropout Only):** Hidden Dropout ($p=0.5$), Balancing Off
    *   **Group C (Balance Only):** Dropout Off, Interleaved $L_2$ Balancing (1 sweep/epoch)
    *   **Group D (Interaction):** Hidden Dropout ($p=0.5$) + Interleaved $L_2$ Balancing

#### 4. Findings and Results (ANN Focus)
While we did find significant SNN efficiency improvements, the **standard ANN training dynamics** revealed a major, standalone discovery regarding how balancing alters the optimization trajectory of dropout:

1.  **Acceleration of Learning:** Balancing vastly accelerates early training. By Epoch 30, validation accuracy was +3.39% higher for Balance vs. Baseline, and +5.59% higher for Balance+Dropout vs. Dropout Alone.
2.  **Mitigation of the Dropout Penalty:** On this architecture, applying $p=0.5$ dropout over 60 epochs was too aggressive and harmed final ANN test accuracy (-1.34% penalty: 89.66% $\to$ 88.32%). However, when dropout was interleaved with balancing, **the network was rescued**. The dropout penalty shrank to just -0.20% (90.85% $\to$ 90.65%). 
3.  **Statistically Significant Interaction:** The difference-in-differences interaction for ANN test accuracy was **+1.15 percentage points** (Holm-adjusted $p=0.0234$). 
4.  **Mechanistic Proof:** The balancing operations succeeded perfectly alongside dropout, achieving a normalized balance deficit of $\sim 10^{-6}$ with zero dead neurons and negligible computational overhead (~36 ms total across 60 epochs).

#### 5. The Adjusted Research Plan (ANN / CNN Accuracy Pivot)
**The Feedback:** Upon reviewing the pilot, Professor Baldi confirmed the results are promising but noted that a 4-layer fully connected network on MNIST is "child's play." Furthermore, calculating SNN efficiency via a software proxy is interesting but ultimately disconnected from physical neuromorphic hardware measurements.

**The Pivot:** We are officially deferring SNN conversion and neuromorphic energy models to a distant final phase. **The primary research question is now focused purely on algorithmic optimization and generalization:** 
*How does interleaved synaptic balancing interact with dropout to improve accuracy and convergence in modern architectures on complex visual datasets?*

**New Hypotheses to Test:**
*   **H1:** The synergistic accuracy interaction between dropout and balancing scales to Convolutional Neural Networks (CNNs).
*   **H2:** Balancing alters the optimal dropout rate (e.g., allowing for much higher, more aggressive dropout probabilities without hurting convergence).
*   **H3:** The regularization benefits of the Dropout $\times$ Balance combination yield state-of-the-art performance improvements on harder benchmarks like CIFAR-10 and ImageNet compared to using either method alone.

#### 6. Next Steps Necessary
To graduate from the "sandbox" to a publication-ready machine learning pipeline, we must execute the following:

1.  **Derive and Implement CNN Balancing:**
    *   Mathematical balancing for standard fully connected layers is $A_j = \sum w_{ij}^2$ and $B_j = \sum w_{jk}^2$. 
    *   *Action:* We must explicitly define and implement how this translates to 2D convolutions (e.g., balancing per-channel, per-filter, or per-spatial-location) ensuring that the scaling remains function-preserving for the CNN.
2.  **Scale the Architecture and Dataset:**
    *   *Action:* Upgrade the PyTorch pipeline from a simple MLP to a standard CNN (e.g., VGG-style or a small ResNet).
    *   *Action:* Swap the dataset from MNIST to **CIFAR-10**.
3.  **Run the $2 \times 2$ Grid on CIFAR-10:**
    *   *Action:* Execute the exact same factorial experiment (Baseline, Dropout, Balance, Both) on the new CNN/CIFAR-10 setup to see if the $+1.15\%$ accuracy interaction holds or widens on a harder task.
4.  **Hardware / SNN (Deferred):**
    *   Keep the existing `ConvertedSNN` code archived. It will only be resurrected if we secure access to an actual neuromorphic chip simulator (like Intel Loihi) *after* the CNN math is proven.