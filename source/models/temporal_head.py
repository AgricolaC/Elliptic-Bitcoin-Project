import torch
import torch.nn as nn
from typing import Tuple
from config import Config

class SnapshotEmbedder(nn.Module):
    """
    Compresses a full snapshot's aggregate graph features into a single summary vector
    using Attention-Based Pooling to focus on anomalous node signatures.
    
    If use_mlp_head is True, this is a 2-layer MLP. Otherwise, a single Linear layer.
    """
    def __init__(self, in_dim: int, embed_dim: int, cfg: Config):
        super().__init__()
        self.use_mlp = cfg.use_mlp_head
        
        # Attention mechanism to weight nodes instead of naive mean pooling
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
        
        if self.use_mlp:
            self.net = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, embed_dim)
            )
        else:
            self.net = nn.Linear(in_dim, embed_dim)
            
    def forward(self, prop_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prop_features: (N, D) tensor of propagated features for all nodes in the snapshot.
        Returns:
            (embed_dim,) tensor representing the snapshot embedding.
        """
        # Calculate attention scores for each node
        attn_scores = self.attention(prop_features)  # (N, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (N, 1)
        
        # Weighted sum of node features based on attention (attention-based pooling)
        # LEAKAGE GUARD: prop_features must NOT include future information.
        # This is ensured because the graph is snapshot-specific and 
        # only labels from < tau are used.
        snapshot_summary = (prop_features * attn_weights).sum(dim=0)
        
        return self.net(snapshot_summary)


class TemporalLSTM(nn.Module):
    """
    LSTM that processes a sequence of snapshot embeddings to model temporal dynamics.
    Outputs the sequence of hidden states.
    """
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        # batch_first=True means input shape is (batch, seq, feature)
        # Since we process one temporal sequence of graphs, batch=1.
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (T, embed_dim) tensor of historical snapshot embeddings.
        Returns:
            (T, hidden_dim) tensor of LSTM hidden states.
        """
        # Add batch dimension: (1, T, embed_dim)
        embeddings = embeddings.unsqueeze(0)
        out, _ = self.lstm(embeddings)
        # Remove batch dimension: (T, hidden_dim)
        return out.squeeze(0)


class SnapshotEMA(nn.Module):
    """
    Exponential moving average of snapshot embeddings.
    h_t = alpha * e_t + (1 - alpha) * h_{t-1}

    No learnable parameters — alpha is fixed. This is the honest baseline
    the LSTM must beat. If it doesn't, that's a clean finding: the LSTM's
    extra capacity memorized ~26 training points rather than learning
    generalizable dynamics.
    """
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns (T, embed_dim) hidden states, same interface as TemporalLSTM."""
        T, D = embeddings.shape
        states = torch.zeros_like(embeddings)
        if T > 0:
            states[0] = embeddings[0]
            for t in range(1, T):
                states[t] = self.alpha * embeddings[t] + (1 - self.alpha) * states[t-1]
        return states


class LSTMConditionedHead(nn.Module):
    """
    Final node classifier.
    Concatenates the current snapshot's LSTM hidden state with the node's
    propagated features.
    """
    def __init__(self, node_in_dim: int, temporal_hidden_dim: int, cfg: Config, n_classes: int = 2):
        super().__init__()
        from models.classifier import SGCHead
        # The SGCHead will take the concatenated feature vector
        combined_dim = node_in_dim + temporal_hidden_dim
        self.head = SGCHead(combined_dim, cfg, n_classes=n_classes)
        
    def forward(self, node_features: torch.Tensor, temporal_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, D) tensor of propagated features for N nodes.
            temporal_context: (H,) tensor of the LSTM hidden state for the current snapshot.
        Returns:
            (N, n_classes) classification logits.
        """
        # Broadcast temporal context to all nodes: (N, H)
        N = node_features.size(0)
        temporal_context_expanded = temporal_context.unsqueeze(0).expand(N, -1)
        
        # Concatenate: (N, D + H)
        combined = torch.cat([node_features, temporal_context_expanded], dim=1)
        return self.head(combined)
