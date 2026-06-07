# GCN vs. SGC: Theoretical Contrast

## The Full GCN Layer (Kipf & Welling)
The canonical Graph Convolutional Network (GCN) layer applies a feature transformation and a non-linear activation at every step of message passing:
$$H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

where:
- $\tilde{A} = A + I$ (Adjacency with self-loops)
- $\tilde{D}$ is the diagonal degree matrix
- $W^{(l)}$ is the learned weight matrix at layer $l$
- $\sigma$ is the activation function (e.g., ReLU)

*Reference: `lecture3 dominant.py`, `nb5_gans_graphs.ipynb`*

## Our SGC Pipeline (Simplified Graph Convolution)
Our pipeline deliberately decouples the spatial aggregation from the feature transformation:
1. **Pre-computation (Aggregation):** We apply the normalized Laplacian iteratively without learned weights or non-linearities:
   $$X_{prop} = \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \right)^k X$$
2. **Transformation (Classifier):** A Multi-Layer Perceptron (MLP) head learns the decision boundary from the pre-computed features.

## Why This Matters for Elliptic
- **Optimization Stability:** The Elliptic dataset requires rapid adaptation across 49 timesteps. The decoupled SGC avoids the gradient instability of deep GCNs.
- **Low-Pass Filtering:** The repeated multiplication by the normalized adjacency acts as a low-pass filter over the graph manifold, effectively capturing homophily without the overhead of end-to-end backpropagation through the structure.
