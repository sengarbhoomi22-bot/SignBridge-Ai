"""SignBridge AI: GRU Sequence Classifier Architecture for Closed-Set ISL Sentence Recognition.

Designed for variable-length 229-dimensional landmark feature sequences.
Implements mask-aware temporal aggregation to strictly exclude zero-padded frames.

Pooling strategies:
  endpoint  — Concatenates forward GRU final-state and backward GRU initial-state.
              Default; retains full backward compatibility with legacy checkpoints.
  mean      — Mask-aware temporal average over all valid frames (Experiment 1).
  attention — Learned additive attention over valid frames only (Experiment 2).
              A single linear scoring layer maps each GRU hidden state to a scalar
              score, which is then masked and normalized via softmax before computing
              a weighted sum of the hidden states.
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
        pooling_strategy: str = "endpoint",
    ) -> None:
        if pooling_strategy not in ("endpoint", "mean", "attention"):
            raise ValueError(
                f"Unsupported pooling_strategy '{pooling_strategy}'. "
                f"Expected 'endpoint', 'mean', or 'attention'."
            )

        if not _TORCH_AVAILABLE:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.num_classes = num_classes
            self.bidirectional = bidirectional
            self.dropout_rate = dropout
            self.pooling_strategy = pooling_strategy
            return

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.pooling_strategy = pooling_strategy

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

        # Learned temporal attention scorer (instantiated only when pooling_strategy == "attention").
        # Maps each GRU output frame h_t ∈ R^{gru_output_dim} to a scalar score e_t ∈ R.
        # Keeping this conditional ensures strict state_dict backward compatibility with legacy
        # checkpoints (endpoint and mean) which do not contain attention_scorer weights.
        if self.pooling_strategy == "attention":
            self.attention_scorer = nn.Linear(gru_output_dim, 1, bias=True)
        else:
            self.attention_scorer = None

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
        return_attention: bool = False,
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
        unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_max, gru_output_dim)

        # 5. Extract representation via selected pooling strategy
        attn_weights = None
        if self.pooling_strategy == "endpoint":
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
        elif self.pooling_strategy == "mean":
            # Mask-aware temporal average pooling over valid sequence frames (0 .. lengths[b] - 1)
            max_t = unpacked_out.size(1)
            t_range = torch.arange(max_t, device=features.device).unsqueeze(0)  # (1, T_max)
            lengths_expanded = lengths.unsqueeze(1)  # (B, 1)
            pad_mask = (t_range < lengths_expanded).float().unsqueeze(-1)  # (B, T_max, 1)

            # Sum over valid frames only; zero-padded frames are zeroed out by pad_mask
            sum_pooled = torch.sum(unpacked_out * pad_mask, dim=1)  # (B, gru_output_dim)
            lengths_clamped = torch.clamp(lengths.unsqueeze(1).float(), min=1.0)  # (B, 1)
            seq_embedding = sum_pooled / lengths_clamped  # (B, gru_output_dim)
        elif self.pooling_strategy == "attention":
            # Learned additive temporal attention pooling.
            #
            # For each sample b and each valid timestep t in [0, T_b):
            #   e_t = attention_scorer(h_t)          scalar energy       (B, T_max, 1)
            #   e_t[t >= T_b] = -inf                 mask padded frames
            #   α_t = softmax(e_t, dim=1)            normalized weights  (B, T_max, 1)
            #   z   = Σ_t α_t * h_t                 weighted sum        (B, gru_output_dim)
            #
            # Guarantees:
            #   - Padded frames receive exactly zero attention weight (-inf → softmax → 0).
            #   - All valid frames remain eligible, including those where hands are absent.
            #   - Σ_t α_t = 1 strictly over valid frames only.
            #   - Output dimension = gru_output_dim = hidden_dim * 2 = 256.
            max_t = unpacked_out.size(1)

            # Compute scalar energy scores for every frame: (B, T_max, 1)
            energy = self.attention_scorer(unpacked_out)  # (B, T_max, 1)

            # Build boolean padding mask: True for valid frames, False for padded frames
            t_range = torch.arange(max_t, device=features.device).unsqueeze(0)   # (1, T_max)
            lengths_expanded = lengths.unsqueeze(1)                               # (B, 1)
            valid_mask = (t_range < lengths_expanded).unsqueeze(-1)               # (B, T_max, 1)

            # Set padded frame energies to -inf so softmax drives their weight to 0
            energy = energy.masked_fill(~valid_mask, float("-inf"))

            # Normalize over the time dimension → α sums to 1 over valid frames per sample
            attn_weights = torch.softmax(energy, dim=1)  # (B, T_max, 1)

            # Weighted sum of GRU outputs over valid frames
            seq_embedding = torch.sum(attn_weights * unpacked_out, dim=1)  # (B, gru_output_dim)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # 6. Classification projection
        seq_embedding = self.dropout(seq_embedding)
        logits = self.classifier(seq_embedding)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_attention_weights(
        self,
        features: Any,
        lengths: Any,
    ) -> Any:
        """Returns normalized temporal attention weights (B, T_max, 1) or None if not attention."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for get_attention_weights.")
        if self.pooling_strategy != "attention":
            return None
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(features, lengths, return_attention=True)
            return attn_weights

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
    def from_config(cls, config: Dict[str, Union[int, float, bool, str]]) -> "SignBridgeGRUClassifier":
        return cls(
            input_dim=int(config.get("input_dim", 229)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 1)),
            num_classes=int(config.get("num_classes", 101)),
            bidirectional=bool(config.get("bidirectional", True)),
            dropout=float(config.get("dropout", 0.2)),
            pooling_strategy=str(config.get("pooling_strategy", "endpoint")),
        )

    def get_config(self) -> Dict[str, Union[int, float, bool, str]]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout_rate,
            "pooling_strategy": self.pooling_strategy,
        }
