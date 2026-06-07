# Methodological Contrast: SGC vs. DOMINANT

*Reference: `lecture3 dominant.py`*

## Overview
While both approaches rely on the normalized Laplacian $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ to capture structural context, their objectives and mechanisms are fundamentally opposed.

| Dimension | Our Pipeline (SGC + MLP) | DOMINANT |
| :--- | :--- | :--- |
| **Objective** | Supervised (Discriminative) | Unsupervised (Generative / Reconstruction) |
| **Loss Function** | Cross-Entropy F1 Optimization | $s(v) = \alpha \|x - \hat{x}\|^2 + (1-\alpha) \|A - \hat{A}\|^2$ |
| **Mechanism** | Sparse Message Passing $\rightarrow$ Bottleneck | GCN Encoder $\rightarrow$ Dual Decoders |
| **Memory Complexity** | $O(|E|)$ (Sparse) | $O(|V|^2)$ (Dense Structure Decoder) |

## The Generative vs. Discriminative Trade-off
DOMINANT's reconstruction error is highly sensitive to the capacity of the latent bottleneck. If illicit and licit nodes share similar intrinsic dimensionalities (`nb3_autoencoders`), reconstruction error fails to separate them. 

Because the Elliptic dataset provides ground-truth labels at every timestep, our pipeline trades the label-free flexibility of DOMINANT for the precision of a **discriminative F1 objective**, explicitly optimizing the boundary between licit and illicit manifolds. Furthermore, the $O(|V|^2)$ dense adjacency requirement of DOMINANT's structure decoder ($\sim 154$ GB for the 203k nodes) violates the schema constraints, making sparse SGC propagation structurally necessary.
