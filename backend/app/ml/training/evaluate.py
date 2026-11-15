"""SignBridge AI: Model Evaluation and Majority-Class Reference Baseline."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore

from backend.app.ml.metrics import compute_classification_metrics


def evaluate_majority_class_baseline(
    train_labels: Sequence[int],
    eval_labels: Sequence[int],
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Computes a majority-class reference baseline.
    
    Determines the mode class in the training split strictly, and predicts it
    for all evaluation samples.
    
    Args:
        train_labels: Sequence of ground-truth integer labels from train set.
        eval_labels: Sequence of ground-truth integer labels from val/test set.
        num_classes: Total number of classes.
        class_names: Optional list of class names.
        
    Returns:
        Dictionary of classification metrics for the majority baseline.
    """
    if not train_labels:
        raise ValueError("Cannot compute majority class from empty training labels.")
    if not eval_labels:
        raise ValueError("Cannot evaluate majority class on empty evaluation labels.")

    # Find mode of training distribution
    counts = Counter(train_labels)
    majority_class_id, majority_count = counts.most_common(1)[0]
    majority_class_name = class_names[majority_class_id] if class_names and majority_class_id < len(class_names) else str(majority_class_id)

    # Predict majority class for all eval instances
    y_pred = [majority_class_id] * len(eval_labels)

    metrics = compute_classification_metrics(
        y_true=eval_labels,
        y_pred=y_pred,
        num_classes=num_classes,
        class_names=class_names,
    )
    metrics["baseline_type"] = "majority_class"
    metrics["majority_class_id"] = majority_class_id
    metrics["majority_class_name"] = majority_class_name
    metrics["majority_train_support"] = majority_count
    return metrics


def evaluate_model(
    model: Any,
    data_loader: Any,
    criterion: Any,
    device: Any,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluates a PyTorch sequence classifier on a given DataLoader.
    
    Args:
        model: PyTorch SignBridgeGRUClassifier instance.
        data_loader: PyTorch DataLoader yielding collated batches.
        criterion: PyTorch loss function (e.g. CrossEntropyLoss).
        device: PyTorch device ('cpu' or 'cuda').
        num_classes: Total number of classes.
        class_names: Optional list of class names.
        
    Returns:
        Dictionary containing validation/test loss and classification metrics.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation.")

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features, lengths)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1)
            y_true_all.extend(labels.cpu().tolist())
            y_pred_all.extend(preds.cpu().tolist())

    mean_loss = total_loss / max(total_samples, 1)
    metrics = compute_classification_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        num_classes=num_classes,
        class_names=class_names,
    )
    metrics["loss"] = round(mean_loss, 4)
    return metrics
