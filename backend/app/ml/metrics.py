"""Evaluation metrics for sequence classification in SignBridge AI.

Pure NumPy / Python implementation of multi-class classification metrics:
- Overall Accuracy
- Macro / Micro Precision, Recall, F1
- Per-class Precision, Recall, F1, Support
- Confusion Matrix
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """Computes an integer confusion matrix of shape (num_classes, num_classes).
    
    Row index represents ground truth class; column index represents predicted class.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-class classification metrics.
    
    Args:
        y_true: Sequence of ground-truth integer class IDs.
        y_pred: Sequence of predicted integer class IDs.
        num_classes: Total number of classes.
        class_names: Optional sequence of human-readable class names.
        
    Returns:
        Dictionary containing overall accuracy, macro precision/recall/F1,
        per-class metrics, support, and confusion matrix.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    total_samples = len(y_t)
    if total_samples == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "total_samples": 0,
            "per_class": {},
            "confusion_matrix": [],
        }

    # Accuracy
    correct_predictions = int(np.sum(y_t == y_p))
    accuracy = float(correct_predictions / total_samples)

    # Confusion matrix
    cm = compute_confusion_matrix(y_t, y_p, num_classes)

    per_class_metrics: Dict[str, Dict[str, Union[float, int]]] = {}
    precisions = []
    recalls = []
    f1_scores = []

    for c in range(num_classes):
        c_name = class_names[c] if class_names and c < len(class_names) else str(c)
        tp = int(cm[c, c])
        fn = int(np.sum(cm[c, :]) - tp)
        fp = int(np.sum(cm[:, c]) - tp)
        support = tp + fn

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        per_class_metrics[c_name] = {
            "class_id": c,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "correct_samples": correct_predictions,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }
