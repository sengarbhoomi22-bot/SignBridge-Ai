"""SignBridge AI: Reusable Sequence Inference Engine for ISL Sentence Recognition.

Performs:
1. Checkpoint loading and configuration alignment.
2. Input sequence integrity validation (T, 229), minimum duration, NaN/Inf checks.
3. Softmax prediction and configurable confidence thresholding.
4. Top-K probability extraction and low-confidence clarification flagging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

from backend.app.ml.labels import LabelEncoder
from backend.app.ml.models.gru_classifier import SignBridgeGRUClassifier


class SignBridgeInferenceEngine:
    """Production-grade inference interface for the trained GRU sequence model."""

    def __init__(
        self,
        model: SignBridgeGRUClassifier,
        label_encoder: LabelEncoder,
        device: str = "cpu",
        min_frames: int = 10,
        default_confidence_threshold: float = 0.50,
    ) -> None:
        self.device = torch.device(device) if torch is not None else device
        self.model = model.to(self.device) if torch is not None else model
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.label_encoder = label_encoder
        self.min_frames = min_frames
        self.default_confidence_threshold = default_confidence_threshold

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        default_confidence_threshold: float = 0.50,
    ) -> "SignBridgeInferenceEngine":
        """Loads a trained model, its configuration, and label mapping from a checkpoint file."""
        if torch is None:
            raise RuntimeError("PyTorch is required for loading model checkpoints.")

        chk_path = Path(checkpoint_path)
        if not chk_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {chk_path}")

        map_location = torch.device(device)
        checkpoint = torch.load(chk_path, map_location=map_location, weights_only=False)

        model_config = checkpoint.get("model_config", {})
        model = SignBridgeGRUClassifier.from_config(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(map_location)
        model.eval()

        # Load label mapping
        classes = checkpoint.get("classes", [])
        if not classes:
            # Fallback to adjacent label_mapping.json if present
            alt_label_path = chk_path.parent / "label_mapping.json"
            if alt_label_path.exists():
                with open(alt_label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    classes = data.get("classes", [])

        label_encoder = LabelEncoder(classes=classes)
        training_config = checkpoint.get("config", {})
        min_frames = int(training_config.get("min_frames", 10))

        return cls(
            model=model,
            label_encoder=label_encoder,
            device=device,
            min_frames=min_frames,
            default_confidence_threshold=default_confidence_threshold,
        )

    def validate_input(
        self,
        sequence: Union[np.ndarray, "torch.Tensor"],
    ) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
        """Validates incoming sequence tensor for dimension, length, and numerical integrity."""
        if sequence is None:
            return False, "Input sequence is None.", None

        if hasattr(sequence, "cpu") and hasattr(sequence, "numpy"):
            arr = sequence.detach().cpu().numpy()
        elif isinstance(sequence, np.ndarray):
            arr = sequence
        else:
            try:
                arr = np.asarray(sequence, dtype=np.float32)
            except Exception as e:
                return False, f"Could not convert input to numpy array: {e}", None

        if arr.ndim != 2:
            return False, f"Expected 2D array of shape (T, 229), but got {arr.ndim}D array with shape {arr.shape}.", None

        t_frames, feat_dim = arr.shape
        if feat_dim != self.model.input_dim:
            return False, f"Feature dimension mismatch: expected {self.model.input_dim}, got {feat_dim}.", None

        if t_frames < self.min_frames:
            return False, f"Sequence length {t_frames} is below minimum threshold ({self.min_frames} frames).", None

        if not np.isfinite(arr).all():
            return False, "Input sequence contains NaN or Infinite values.", None

        return True, None, arr.astype(np.float32)

    def predict(
        self,
        sequence: Union[np.ndarray, "torch.Tensor"],
        confidence_threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Performs sentence classification with quality gating and confidence thresholding."""
        if torch is None:
            raise RuntimeError("PyTorch is required for inference.")

        thresh = confidence_threshold if confidence_threshold is not None else self.default_confidence_threshold

        is_valid, error_msg, valid_arr = self.validate_input(sequence)
        if not is_valid or valid_arr is None:
            return {
                "status": "INVALID_INPUT",
                "error": error_msg,
                "predicted_class_id": -1,
                "predicted_label": "",
                "confidence": 0.0,
                "is_low_confidence": True,
                "num_frames": 0,
                "top_k": [],
            }

        t_frames = valid_arr.shape[0]

        # Prepare batch-size-one tensor
        feat_tensor = torch.from_numpy(valid_arr).unsqueeze(0).to(self.device)  # (1, T, 229)
        length_tensor = torch.tensor([t_frames], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(feat_tensor, length_tensor)  # (1, num_classes)
            probs = F.softmax(logits, dim=-1).squeeze(0)     # (num_classes,)

        top_k_val, top_k_idx = torch.topk(probs, min(top_k, len(probs)))
        top_k_scores = top_k_val.cpu().tolist()
        top_k_classes = top_k_idx.cpu().tolist()

        best_class_id = int(top_k_classes[0])
        best_confidence = float(top_k_scores[0])

        try:
            predicted_label = self.label_encoder.inverse_transform([best_class_id])[0]
        except Exception:
            predicted_label = f"Class_{best_class_id}"

        is_low_conf = best_confidence < thresh
        status = "LOW_CONFIDENCE" if is_low_conf else "SUCCESS"

        top_k_list = []
        for cid, score in zip(top_k_classes, top_k_scores):
            try:
                name = self.label_encoder.inverse_transform([cid])[0]
            except Exception:
                name = f"Class_{cid}"
            top_k_list.append({"class_id": cid, "label": name, "probability": round(score, 4)})

        return {
            "status": status,
            "predicted_class_id": best_class_id,
            "predicted_label": predicted_label,
            "confidence": round(best_confidence, 4),
            "is_low_confidence": is_low_conf,
            "confidence_threshold": thresh,
            "num_frames": t_frames,
            "top_k": top_k_list,
        }
