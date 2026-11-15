"""SignBridge AI Machine Learning Subsystem.

Provides:
- ISLLandmarkDataset: PyTorch Dataset for loading preprocessed ISL-CSLTR tensors.
- LabelEncoder: Deterministic label mapping fitted strictly on training data.
- SequenceBatchCollator: Padding-aware batch collation with separate masks.
- SignBridgeGRUClassifier: Recurrent neural sequence model for ISL sentence recognition.
- SignBridgeInferenceEngine: High-level inference engine with confidence verification.
- Training routines and metrics.
"""

from backend.app.ml.collate import SequenceBatchCollator
from backend.app.ml.dataset import ISLLandmarkDataset
from backend.app.ml.inference import SignBridgeInferenceEngine
from backend.app.ml.labels import LabelEncoder
from backend.app.ml.metrics import compute_classification_metrics, compute_confusion_matrix
from backend.app.ml.models.gru_classifier import SignBridgeGRUClassifier
from backend.app.ml.training.config import TrainingConfig
from backend.app.ml.training.evaluate import evaluate_majority_class_baseline, evaluate_model
from backend.app.ml.training.train import train_gru

__all__ = [
    "ISLLandmarkDataset",
    "LabelEncoder",
    "SequenceBatchCollator",
    "SignBridgeGRUClassifier",
    "SignBridgeInferenceEngine",
    "TrainingConfig",
    "train_gru",
    "evaluate_model",
    "evaluate_majority_class_baseline",
    "compute_classification_metrics",
    "compute_confusion_matrix",
]
