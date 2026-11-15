"""SignBridge AI: GRU Sequence Classifier Architecture for Closed-Set ISL Sentence Recognition.

Designed for variable-length 229-dimensional landmark feature sequences.
Implements mask-aware temporal aggregation to strictly exclude zero-padded frames.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = type("nn", (), {"Module": object})()  # type: ignore
    F = None  # type: ignore
    pack_padded_sequence = None  # type: ignore
    pad_packed_sequence = None  # type: ignore
    _TORCH_AVAILABLE = False


class SignBridgeGRUClassifier(nn.Module):
    """Bidirectional Gated Recurrent Unit (GRU) Sequence Classifier.
    
    Attributes:
        input_dim: Number of input features per frame (229 for Candidate B).
        hidden_dim: Number of features in the GRU hidden state.
        num_layers: Number of recurrent layers.
        num_classes: Number of sentence categories in the closed-set vocabulary.
        bidirectional: Whether the GRU is bidirectional.
        dropout: Dropout probability applied between layers and before head.
    """

    def __init__(
        self,
        input_dim: int = 229,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 101,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ) -> None:
        if not _TORCH_AVAILABLE:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_classes = num_classes
            self.bidirectional = bidirectional
            self.dropout_rate = dropout
            return

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.dropout_rate = dropout

        # Input feature projection and normalization layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout if num_layers > 1 else 0.0),
        )

        # Recurrent sequence encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        direction_factor = 2 if bidirectional else 1
        gru_output_dim = hidden_dim * direction_factor

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: Any,
        lengths: Any,
    ) -> Any:
        """Forward pass over a batch of variable-length padded sequences."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to execute SignBridgeGRUClassifier forward pass.")

        lengths_cpu = lengths.cpu()
        batch_size = features.size(0)

        # 1. Project input features
        proj_features = self.input_proj(features)  # (B, T_max, hidden_dim)

        # 2. Pack padded sequence
        packed_in = pack_padded_sequence(
            proj_features,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        # 3. GRU Recurrent Encoding
        packed_out, _ = self.gru(packed_in)

        # 4. Unpack sequence
        unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # 5. Extract endpoints
        if self.bidirectional:
            forward_out = unpacked_out[:, :, :self.hidden_dim]
            backward_out = unpacked_out[:, :, self.hidden_dim:]

            batch_indices = torch.arange(batch_size, device=features.device)
            last_indices = lengths - 1

            final_forward = forward_out[batch_indices, last_indices, :]
            initial_backward = backward_out[batch_indices, 0, :]

            seq_embedding = torch.cat([final_forward, initial_backward], dim=-1)
        else:
            batch_indices = torch.arange(batch_size, device=features.device)
            last_indices = lengths - 1
            seq_embedding = unpacked_out[batch_indices, last_indices, :]

        # 6. Classification projection
        seq_embedding = self.dropout(seq_embedding)
        logits = self.classifier(seq_embedding)

        return logits

    def predict_proba(
        self,
        features: Any,
        lengths: Any,
    ) -> Any:
        """Inference helper returning normalized class probabilities."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for predict_proba.")
        self.eval()
        with torch.no_grad():
            logits = self.forward(features, lengths)
            return F.softmax(logits, dim=-1)

    @classmethod
    def from_config(cls, config: Dict[str, Union[int, float, bool]]) -> "SignBridgeGRUClassifier":
        return cls(
            input_dim=int(config.get("input_dim", 229)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 1)),
            num_classes=int(config.get("num_classes", 101)),
            bidirectional=bool(config.get("bidirectional", True)),
            dropout=float(config.get("dropout", 0.2)),
        )

    def get_config(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout_rate,
        }
