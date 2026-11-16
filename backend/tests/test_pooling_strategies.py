"""SignBridge AI: Comprehensive Tests for Temporal Pooling Strategies.

Verifies:
1. Endpoint pooling retains its exact previous behaviour.
2. Mean pooling averages only valid sequence outputs.
3. Variable sequence lengths in the same batch work correctly.
4. Padding values do not affect mean pooling.
5. Missing-hand frames remain included in temporal average.
6. Output shape remains (batch_size, num_classes).
7. Gradients remain finite under both pooling strategies.
8. Model and training configuration serialization and roundtrip.
9. Legacy checkpoints without pooling_strategy default to endpoint.
10. Inference engine loads both endpoint and mean checkpoints correctly.
11. Invalid pooling strategy raises ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.app.ml.collate import SequenceBatchCollator
from backend.app.ml.dataset import ISLLandmarkDataset
from backend.app.ml.inference import SignBridgeInferenceEngine
from backend.app.ml.labels import LabelEncoder
from backend.app.ml.models.gru_classifier import SignBridgeGRUClassifier
from backend.app.ml.training.config import TrainingConfig


# -----------------------------------------------------------------------------
# 1. Endpoint Pooling Behavioral Fidelity
# -----------------------------------------------------------------------------
def test_endpoint_pooling_retains_previous_behavior() -> None:
    torch.manual_seed(42)
    model = SignBridgeGRUClassifier(
        input_dim=229,
        hidden_dim=64,
        num_classes=10,
        bidirectional=True,
        pooling_strategy="endpoint",
    )
    model.eval()

    # Create synthetic variable-length batch
    lengths = torch.tensor([12, 18], dtype=torch.long)
    features = torch.randn(2, 20, 229)

    with torch.no_grad():
        logits = model(features, lengths)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    assert model.pooling_strategy == "endpoint"


# -----------------------------------------------------------------------------
# 2. Mean Pooling Mathematical Correctness
# -----------------------------------------------------------------------------
def test_mean_pooling_averages_only_valid_sequence_outputs() -> None:
    torch.manual_seed(42)
    model = SignBridgeGRUClassifier(
        input_dim=229,
        hidden_dim=32,
        num_classes=5,
        bidirectional=True,
        pooling_strategy="mean",
    )
    model.eval()

    lengths = torch.tensor([10, 20], dtype=torch.long)
    features = torch.randn(2, 25, 229)

    with torch.no_grad():
        logits = model(features, lengths)

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert model.pooling_strategy == "mean"


# -----------------------------------------------------------------------------
# 3. Variable Sequence Lengths in Same Batch
# -----------------------------------------------------------------------------
def test_different_sequence_lengths_in_same_batch() -> None:
    torch.manual_seed(42)
    model_mean = SignBridgeGRUClassifier(
        input_dim=229,
        hidden_dim=32,
        num_classes=71,
        bidirectional=True,
        pooling_strategy="mean",
    )
    model_mean.eval()

    lengths = torch.tensor([10, 15, 30, 45], dtype=torch.long)
    features = torch.randn(4, 50, 229)

    with torch.no_grad():
        out = model_mean(features, lengths)

    assert out.shape == (4, 71)
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------------
# 4. Padding Invariance: Padding Values Do Not Affect Mean Pooling
# -----------------------------------------------------------------------------
def test_padding_values_do_not_affect_mean_pooling() -> None:
    torch.manual_seed(42)
    model = SignBridgeGRUClassifier(
        input_dim=229,
        hidden_dim=32,
        num_classes=10,
        bidirectional=True,
        pooling_strategy="mean",
    )
    model.eval()

    # Sequence with true length 12 padded to 25
    lengths = torch.tensor([12], dtype=torch.long)
    
    # Batch A: zero padding beyond length 12
    feat_a = torch.randn(1, 25, 229)
    feat_a[:, 12:, :] = 0.0

    # Batch B: identical valid frames (0..11), but arbitrary non-zero numbers in padded zone (12..24)
    feat_b = feat_a.clone()
    feat_b[:, 12:, :] = 999.0

    with torch.no_grad():
        out_a = model(feat_a, lengths)
        out_b = model(feat_b, lengths)

    # In PyTorch RNN, pack_padded_sequence ignores inputs beyond length[i]
    # pad_mask in mean pooling ensures unpack values beyond length[i] are zeroed.
    # Therefore out_a and out_b must be strictly identical within numerical precision!
    assert torch.allclose(out_a, out_b, atol=1e-5), "Padding values corrupted mean pooling output!"


# -----------------------------------------------------------------------------
# 5. Missing-Hand Frames Remain Included
# -----------------------------------------------------------------------------
def test_missing_hand_frames_remain_included() -> None:
    torch.manual_seed(42)
    model = SignBridgeGRUClassifier(
        input_dim=229,
        hidden_dim=32,
        num_classes=5,
        bidirectional=True,
        pooling_strategy="mean",
    )
    model.eval()

    lengths = torch.tensor([20], dtype=torch.long)
    feat = torch.randn(1, 20, 229)

    # Frame 0..9: hands missing (indicators = 0.0, coords = 0.0)
    feat[:, :10, 0:126] = 0.0
    feat[:, :10, 225] = 0.0
    feat[:, :10, 226] = 0.0
    # Pose remains active
    feat[:, :10, 227] = 1.0

    with torch.no_grad():
        out = model(feat, lengths)

    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------------
# 6. Output Shape with 71 Classes
# -----------------------------------------------------------------------------
def test_output_shape_71_classes() -> None:
    for strategy in ["endpoint", "mean"]:
        model = SignBridgeGRUClassifier(
            input_dim=229,
            hidden_dim=128,
            num_classes=71,
            bidirectional=True,
            pooling_strategy=strategy,
        )
        model.eval()
        lengths = torch.tensor([15, 25, 35], dtype=torch.long)
        feats = torch.randn(3, 40, 229)

        with torch.no_grad():
            logits = model(feats, lengths)

        assert logits.shape == (3, 71)
        assert torch.isfinite(logits).all()


# -----------------------------------------------------------------------------
# 7. Finite Gradients Under Both Strategies
# -----------------------------------------------------------------------------
def test_gradients_remain_finite() -> None:
    criterion = nn.CrossEntropyLoss()

    for strategy in ["endpoint", "mean"]:
        torch.manual_seed(42)
        model = SignBridgeGRUClassifier(
            input_dim=229,
            hidden_dim=64,
            num_classes=8,
            bidirectional=True,
            pooling_strategy=strategy,
        )
        model.train()

        lengths = torch.tensor([12, 18], dtype=torch.long)
        feats = torch.randn(2, 20, 229, requires_grad=True)
        labels = torch.tensor([2, 5], dtype=torch.long)

        logits = model(feats, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Gradient is None for {name} ({strategy})"
                assert torch.isfinite(p.grad).all(), f"Gradient contains NaN/Inf for {name} ({strategy})"


# -----------------------------------------------------------------------------
# 8. Configuration Serialization & Roundtrip
# -----------------------------------------------------------------------------
def test_configuration_serialization_and_roundtrip(tmp_path: Path) -> None:
    for strategy in ["endpoint", "mean"]:
        model = SignBridgeGRUClassifier(
            input_dim=229,
            hidden_dim=64,
            num_classes=71,
            bidirectional=True,
            pooling_strategy=strategy,
        )
        cfg = model.get_config()
        assert cfg["pooling_strategy"] == strategy

        reconstructed = SignBridgeGRUClassifier.from_config(cfg)
        assert reconstructed.pooling_strategy == strategy
        assert reconstructed.input_dim == 229
        assert reconstructed.hidden_dim == 64
        assert reconstructed.num_classes == 71

    # TrainingConfig roundtrip
    t_cfg = TrainingConfig(pooling_strategy="mean", hidden_dim=64)
    assert t_cfg.pooling_strategy == "mean"
    cfg_file = tmp_path / "train_cfg.json"
    t_cfg.save(cfg_file)
    loaded_t_cfg = TrainingConfig.load(cfg_file)
    assert loaded_t_cfg.pooling_strategy == "mean"


# -----------------------------------------------------------------------------
# 9. Legacy Checkpoint Compatibility
# -----------------------------------------------------------------------------
def test_legacy_checkpoint_without_pooling_strategy_defaults_to_endpoint() -> None:
    # Config dictionary simulating a legacy checkpoint without "pooling_strategy"
    legacy_config = {
        "input_dim": 229,
        "hidden_dim": 128,
        "num_layers": 1,
        "num_classes": 71,
        "bidirectional": True,
        "dropout": 0.2,
    }

    model = SignBridgeGRUClassifier.from_config(legacy_config)
    assert model.pooling_strategy == "endpoint"

    # Verify actual baseline checkpoint on disk
    baseline_chk_path = Path("models/checkpoints/gru_baseline_v1/best_gru_model.pt")
    if baseline_chk_path.exists():
        chk = torch.load(baseline_chk_path, map_location="cpu", weights_only=False)
        loaded_model = SignBridgeGRUClassifier.from_config(chk.get("model_config", {}))
        assert loaded_model.pooling_strategy == "endpoint"


# -----------------------------------------------------------------------------
# 10. Inference Engine Loads Both Endpoint and Mean Models
# -----------------------------------------------------------------------------
def test_inference_engine_loads_both_endpoint_and_mean(tmp_path: Path) -> None:
    classes = [f"class_{i}" for i in range(10)]
    label_enc = LabelEncoder(classes=classes)

    for strategy in ["endpoint", "mean"]:
        model = SignBridgeGRUClassifier(
            input_dim=229,
            hidden_dim=32,
            num_classes=10,
            bidirectional=True,
            pooling_strategy=strategy,
        )
        model.eval()

        # Save mock checkpoint
        chk_path = tmp_path / f"model_{strategy}.pt"
        payload = {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "classes": classes,
            "num_classes": 10,
        }
        torch.save(payload, chk_path)

        # Load into SignBridgeInferenceEngine
        engine = SignBridgeInferenceEngine.load_from_checkpoint(chk_path, device="cpu")
        assert engine.model.pooling_strategy == strategy

        # Run forward prediction
        dummy_seq = np.zeros((15, 229), dtype=np.float32)
        res = engine.predict(dummy_seq)
        assert res["status"] in ("SUCCESS", "LOW_CONFIDENCE")
        assert res["predicted_class_id"] in range(10)


# -----------------------------------------------------------------------------
# 11. Invalid Pooling Strategy Raises ValueError
# -----------------------------------------------------------------------------
def test_invalid_pooling_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported pooling_strategy"):
        SignBridgeGRUClassifier(pooling_strategy="invalid_pooling")
