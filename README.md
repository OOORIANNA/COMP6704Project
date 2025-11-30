# COMP6704Project

# Joint Optimization of Physical Wave Propagation and Neural Architecture for Leaky-Lamb-Wave-Based Sensing

**Author:** Zehua CAO (Student ID: 25064021r)  
**Affiliation:** The Hong Kong Polytechnic University  
**Course:** Optimization Methods (COMP6704)  

---

## 1. Project Overview

This project addresses the **inverse scattering problem** in non-contact environmental perception using guided acoustic waves. We reformulate the physical perception task as a **data-driven stochastic optimization problem**, bypassing the intractability of analytical inversion.

The framework employs a **bilevel optimization strategy**:
1.  **Design Space Optimization:** Determining the optimal discrete sensor configuration to maximize information entropy.
2.  **Parameter Optimization:** Approximating the inverse mapping $\Phi^{-1}$ by minimizing the regularized empirical risk of a deep neural architecture using the **Adam** stochastic solver.

---

## 2. Problem Formulation

The core optimization objective is to find the optimal network parameters $\theta^*$ that minimize the divergence between the predicted object states and the ground truth $y$, subject to $L_2$ regularization:

$$
\min_{\theta} J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathcal{X}_i), y_i) + \lambda \|\theta\|_2^2
$$

Where:
* $\mathcal{X}$: Stochastic input signals from the sensor array.
* $f_\theta$: The neural function approximator.
* $\mathcal{L}$: Cross-Entropy loss function (non-convex w.r.t $\theta$).
* $\text{Solver}$: Adaptive Moment Estimation (Adam).

---

## 3. Methodology

### 3.1 Neural Function Approximator
We utilize a hybrid deep neural network (integrating Convolutional and BiLSTM operators) to serve as the inverse operator. This architecture is designed to capture both local oscillating features and global temporal dependencies inherent in Lamb wave signals.

### 3.2 Stochastic Optimization
The **Adam** algorithm is employed to navigate the sparse and non-convex loss landscape. Its adaptive learning rate mechanism is critical for handling the high variance characteristic of transient wave packets, ensuring convergence to a robust flat minimum.

---

## 4. Implementation & Usage

The implementation is based on **MATLAB** using the Deep Learning Toolbox.

### Prerequisites
* MATLAB R2021a or later
* Deep Learning Toolbox

### Execution
The entry point for the optimization pipeline is `main.m` (or `data_classification_3chl.m`). The script performs the following:
1.  **Data Loading & Preprocessing:** Reshapes raw acoustic signals into multi-channel time-series sequences.
2.  **K-Fold Cross-Validation:** Partitions the dataset to evaluate generalization.
3.  **Stochastic Optimization:** Trains the neural approximator using the `adam` solver.
4.  **Evaluation:** Outputs classification accuracy and confusion matrices.

To reproduce the results, simply run:
```matlab
>> data_classification_3chl
