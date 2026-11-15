"""SignBridge AI: Step 3 Phase L — Comprehensive Unit Tests for ML Components.

Covers:
1. LabelEncoder (fit, transform, inverse_transform, unknown class handling, save/load).
2. SequenceBatchCollator (variable lengths, zero-padding, separate sequence padding mask).
3. Classification metrics (accuracy, precision, recall, F1, confusion matrix, edge cases).
4. TrainingConfig serialization and validation.
5. Majority-class baseline computation.
6. SignBridgeInferenceEngine input validation and threshold logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from backend.app.ml.collate import SequenceBatchCollator
from backend.app.ml.labels import LabelEncoder
from backend.app.ml.metrics import compute_classification_metrics, compute_confusion_matrix
from backend.app.ml.training.config import TrainingConfig
from backend.app.ml.training.evaluate import evaluate_majority_class_baseline


# ---------------------------------------------------------------------------
# 1. LabelEncoder Tests
# ---------------------------------------------------------------------------
def test_label_encoder_fit_transform() -> None:
    labels = ["cat", "dog", "apple", "cat", "banana"]
    encoder = LabelEncoder()
    encoder.fit(labels)

    assert encoder.num_classes == 4
    assert encoder.classes_ == ["apple", "banana", "cat", "dog"]

    encoded = encoder.transform(["cat", "apple", "dog"])
    assert encoded == [2, 0, 3]

    decoded = encoder.inverse_transform(encoded)
    assert decoded == ["cat", "apple", "dog"]


def test_label_encoder_unknown_label_raises() -> None:
    encoder = LabelEncoder().fit(["apple", "banana"])
    with pytest.raises(KeyError, match="not present"):
        encoder.transform(["unknown_fruit"])


def test_label_encoder_save_and_load(tmp_path: Path) -> None:
    encoder = LabelEncoder().fit(["hello", "world", "sign"])
    save_file = tmp_path / "labels.json"
    encoder.save(save_file)

    loaded = LabelEncoder.load(save_file)
    assert loaded.num_classes == 3
    assert loaded.classes_ == encoder.classes_
    assert loaded.transform(["world"]) == encoder.transform(["world"])


# ---------------------------------------------------------------------------
# 2. SequenceBatchCollator Tests
# ---------------------------------------------------------------------------
def test_sequence_batch_collator_variable_lengths() -> None:
    collator = SequenceBatchCollator(feature_dim=229)

    sample1 = {
        "features": np.ones((15, 229), dtype=np.float32) * 1.5,
        "presence_mask": np.ones((15, 4), dtype=np.float32),
        "label_id": 0,
        "label_str": "sentence_a",
        "metadata": {"clip": "clip_1"},
    }
    sample2 = {
        "features": np.ones((25, 229), dtype=np.float32) * 2.5,
        "presence_mask": np.ones((25, 4), dtype=np.float32),
        "label_id": 1,
        "label_str": "sentence_b",
        "metadata": {"clip": "clip_2"},
    }

    batch = collator([sample1, sample2])

    feat = batch["features"]
    lengths = batch["lengths"]
    pad_mask = batch["padding_mask"]

    # Convert to numpy if PyTorch tensors
    feat_np = feat.numpy() if hasattr(feat, "numpy") else feat
    lens_np = lengths.numpy() if hasattr(lengths, "numpy") else lengths
    mask_np = pad_mask.numpy() if hasattr(pad_mask, "numpy") else pad_mask

    assert feat_np.shape == (2, 25, 229)
    assert list(lens_np) == [15, 25]

    # Check padding in sample 1
    assert np.all(feat_np[0, :15, :] == 1.5)
    assert np.all(feat_np[0, 15:, :] == 0.0)  # Padded frames must be zero
    assert np.all(mask_np[0, :15] == 1.0)     # Valid frames
    assert np.all(mask_np[0, 15:] == 0.0)     # Padded frames mask = 0.0

    # Check sample 2 has no padding
    assert np.all(feat_np[1, :25, :] == 2.5)
    assert np.all(mask_np[1, :] == 1.0)


def test_sequence_batch_collator_empty_raises() -> None:
    collator = SequenceBatchCollator(feature_dim=229)
    with pytest.raises(ValueError, match="empty batch"):
        collator([])


# ---------------------------------------------------------------------------
# 3. Metrics Tests
# ---------------------------------------------------------------------------
def test_compute_confusion_matrix() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2  # class 0: 2 correctly predicted
    assert cm[1, 1] == 1  # class 1: 1 correct, 1 predicted as 2
    assert cm[1, 2] == 1
    assert cm[2, 2] == 1  # class 2: 1 correct, 1 predicted as 1
    assert cm[2, 1] == 1


def test_classification_metrics_perfect() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["total_samples"] == 6


def test_classification_metrics_empty() -> None:
    metrics = compute_classification_metrics([], [], num_classes=3)
    assert metrics["accuracy"] == 0.0
    assert metrics["total_samples"] == 0


# ---------------------------------------------------------------------------
# 4. TrainingConfig Tests
# ---------------------------------------------------------------------------
def test_training_config_defaults_and_save(tmp_path: Path) -> None:
    cfg = TrainingConfig(seed=123, hidden_dim=64, epochs=10)
    assert cfg.input_dim == 229
    assert cfg.hidden_dim == 64
    assert cfg.epochs == 10

    cfg_file = tmp_path / "config.json"
    cfg.save(cfg_file)

    loaded = TrainingConfig.load(cfg_file)
    assert loaded.seed == 123
    assert loaded.hidden_dim == 64
    assert loaded.epochs == 10


# ---------------------------------------------------------------------------
# 5. Majority-Class Baseline Tests
# ---------------------------------------------------------------------------
def test_evaluate_majority_class_baseline() -> None:
    train_labels = [0, 0, 0, 1, 2]  # Class 0 is majority (3/5)
    eval_labels = [0, 1, 0, 2]      # Ground truth has two 0s, one 1, one 2

    res = evaluate_majority_class_baseline(
        train_labels=train_labels,
        eval_labels=eval_labels,
        num_classes=3,
        class_names=["apple", "banana", "cherry"],
    )

    assert res["majority_class_id"] == 0
    assert res["majority_class_name"] == "apple"
    # Predicting 0 for all 4 gives 2 correct out of 4 (Accuracy = 0.50)
    assert res["accuracy"] == 0.50
    assert res["total_samples"] == 4


# ---------------------------------------------------------------------------
# 6. Inference Validation Logic Tests
# ---------------------------------------------------------------------------
def test_inference_input_validation() -> None:
    from backend.app.ml.inference import SignBridgeInferenceEngine

    class MockModel:
        input_dim = 229
        def to(self, d: Any) -> "MockModel":
            return self
        def eval(self) -> None:
            pass

    engine = SignBridgeInferenceEngine(
        model=MockModel(),  # type: ignore
        label_encoder=LabelEncoder(classes=["c0", "c1"]),
        min_frames=10,
    )

    # Valid sequence
    valid_seq = np.zeros((15, 229), dtype=np.float32)
    is_ok, err, arr = engine.validate_input(valid_seq)
    assert is_ok is True
    assert err is None
    assert arr is not None and arr.shape == (15, 229)

    # Wrong feature dimension
    wrong_dim = np.zeros((15, 100), dtype=np.float32)
    is_ok, err, _ = engine.validate_input(wrong_dim)
    assert is_ok is False
    assert "Feature dimension mismatch" in str(err)

    # Too short
    short_seq = np.zeros((5, 229), dtype=np.float32)
    is_ok, err, _ = engine.validate_input(short_seq)
    assert is_ok is False
    assert "below minimum threshold" in str(err)

    # Contains NaN
    nan_seq = np.zeros((15, 229), dtype=np.float32)
    nan_seq[2, 5] = np.nan
    is_ok, err, _ = engine.validate_input(nan_seq)
    assert is_ok is False
    assert "NaN or Infinite" in str(err)
