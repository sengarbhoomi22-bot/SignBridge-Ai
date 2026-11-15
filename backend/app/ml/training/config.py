"""SignBridge AI: ML Training Configuration Dataclass and Serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for GRU model training."""

    # Reproducibility
    seed: int = 42

    # Model architecture
    input_dim: int = 229
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2
    num_classes: int = 101

    # Optimization
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0

    # Dataset paths and filtering
    data_dir: str = "data/processed/isl_csltr_tensors_v2"
    min_frames: int = 10
    label_key: str = "sentence_text"
    filter_suitability: bool = True

    # Artifact persistence
    checkpoint_dir: str = "models/checkpoints"
    model_name: str = "best_gru_model.pt"

    # Execution device
    device: str = "auto"  # 'auto', 'cpu', 'cuda'

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Instantiates configuration from dictionary."""
        # Filter keys to match dataclass fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves configuration to JSON file."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainingConfig":
        """Loads configuration from JSON file."""
        target_path = Path(filepath)
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
