# SignBridge AI: Machine Learning Subsystem (`backend/app/ml/`)

This directory houses the first machine learning baseline for the SignBridge AI Indian Sign Language (ISL) sentence recognition pipeline.

---

## 1. Subsystem Architecture

```
backend/app/ml/
├── __init__.py               # Top-level exports
├── dataset.py                # ISLLandmarkDataset for loading .npz tensor archives
├── labels.py                 # LabelEncoder fitted strictly on training partition
├── collate.py                # SequenceBatchCollator (pads to batch max, creates separate padding mask)
├── metrics.py                # Macro Precision, Recall, F1, Accuracy, and Confusion Matrix
├── inference.py              # SignBridgeInferenceEngine with confidence gating
├── models/
│   ├── __init__.py           # Model factory
│   └── gru_classifier.py     # Mask-aware bidirectional GRU classifier
├── training/
│   ├── __init__.py           # Training package exports
│   ├── config.py             # TrainingConfig dataclass
│   ├── train.py              # Full epoch loop, early stopping, and checkpointing
│   └── evaluate.py           # Evaluation pipeline and majority-class baseline
└── README.md                 # This documentation
```

---

## 2. Feature Representation: Candidate B 3D (229 Dimensions)

Each video frame is processed into a 229-dimensional vector:
1. **Pose Landmarks (99 values):** 33 pose landmarks $(x, y, z)$ centered at the mid-shoulder anchor and scaled by bi-acromial distance.
2. **Left Hand (63 values):** 21 hand landmarks $(x, y, z)$ relative to the left wrist, scaled by shoulder distance. All coordinates are $0.0$ if the hand is not detected.
3. **Right Hand (63 values):** 21 hand landmarks $(x, y, z)$ relative to the right wrist, scaled by shoulder distance. All coordinates are $0.0$ if the hand is not detected.
4. **Presence Indicators (4 values):** $[m_{\text{left}}, m_{\text{right}}, m_{\text{pose}}, m_{\text{norm}}]$ indicating whether hands and pose were detected, and whether normalization was valid.

---

## 3. Crucial ML Design Principles

### A. Zero Data Fabrication
Missing hands are **never linearly interpolated or synthetically hallucinated**. They are set to $0.0$ and explicitly flagged by the presence indicators.

### B. Decoupled Padding Mask vs. Modality Indicators
In batches of variable-duration videos, shorter videos are padded with zeros to match the longest video in the batch.
- **Sequence Padding Mask:** A 2D mask of shape `(Batch_Size, T_max)` where `1.0` denotes real video frames and `0.0` denotes padded frames. This prevents the GRU from processing padded frames.
- **Modality Indicators:** Embedded in each frame (dimensions 225–228) indicating hand visibility.

### C. Strict Signer-Independent Partitioning
- **Training (Signers 1–5):** All parameter updates and LabelEncoder fitting happen strictly here.
- **Validation (Signer 6):** Used exclusively for model selection, hyperparameter tuning, and early stopping.
- **Test (Signer 7):** Held-out volunteer signer evaluated exactly once after training is complete.

---

## 4. Quick Start Usage

### Training from Script
```powershell
backend\.venv\Scripts\python.exe scripts/train_gru.py --epochs 50 --batch-size 16 --lr 0.001
```

### Inference
```python
from backend.app.ml.inference import SignBridgeInferenceEngine

# Load from trained checkpoint
engine = SignBridgeInferenceEngine.load_from_checkpoint("models/checkpoints/best_gru_model.pt")

# Predict on an incoming sequence of shape (T, 229)
result = engine.predict(landmark_sequence, confidence_threshold=0.60)
print(f"Predicted Sentence: {result['predicted_label']} (Confidence: {result['confidence']:.2f})")
if result["is_low_confidence"]:
    print("Warning: Low confidence prediction; prompting user for repetition.")
```
