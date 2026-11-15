"""SignBridge AI: PyTorch GRU Training Pipeline with Early Stopping and Metric Tracking."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    AdamW = None  # type: ignore
    DataLoader = None  # type: ignore

from backend.app.ml.collate import SequenceBatchCollator
from backend.app.ml.dataset import ISLLandmarkDataset
from backend.app.ml.labels import LabelEncoder
from backend.app.ml.models.gru_classifier import SignBridgeGRUClassifier
from backend.app.ml.training.config import TrainingConfig
from backend.app.ml.training.evaluate import evaluate_model


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def select_device(preferred: str = "auto") -> Any:
    """Selects execution device with explicit user-facing feedback."""
    if torch is None:
        raise RuntimeError("PyTorch is required to select an execution device.")

    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    elif preferred == "cpu":
        device = torch.device("cpu")
        print("[DEVICE] Using CPU (explicitly requested)")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[DEVICE] CUDA detected: using GPU ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("[DEVICE] CUDA not detected: using CPU (Intel Core Ultra / Host)")
    return device


def train_gru(
    config: TrainingConfig,
    train_dataset: Optional[ISLLandmarkDataset] = None,
    val_dataset: Optional[ISLLandmarkDataset] = None,
    label_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, Any]:
    """Executes the full GRU baseline training loop.
    
    Args:
        config: TrainingConfig dataclass instance.
        train_dataset: Optional pre-instantiated training dataset.
        val_dataset: Optional pre-instantiated validation dataset.
        label_encoder: Optional pre-fitted LabelEncoder.
        
    Returns:
        Summary dictionary containing best validation metrics, history, and checkpoint path.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for training.")

    set_seed(config.seed)
    device = select_device(config.device)

    data_dir = Path(config.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    # 1. Dataset & Label Preparation
    if train_dataset is None:
        train_dataset = ISLLandmarkDataset(
            split_dir=train_dir,
            min_frames=config.min_frames,
            filter_suitability=config.filter_suitability,
            label_key=config.label_key,
        )

    if label_encoder is None:
        train_labels = train_dataset.get_labels()
        if not train_labels:
            raise ValueError(f"No valid training samples found in {train_dir}.")
        label_encoder = LabelEncoder().fit(train_labels)
        train_dataset.label_encoder = label_encoder

    if val_dataset is None and val_dir.exists():
        val_dataset = ISLLandmarkDataset(
            split_dir=val_dir,
            label_encoder=label_encoder,
            min_frames=config.min_frames,
            filter_suitability=config.filter_suitability,
            label_key=config.label_key,
        )

    num_classes = label_encoder.num_classes
    config.num_classes = num_classes

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label_mapping_path = checkpoint_dir / "label_mapping.json"
    label_encoder.save(label_mapping_path)

    # 2. DataLoaders
    collator = SequenceBatchCollator(feature_dim=config.input_dim)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    # 3. Model, Loss, Optimizer
    model = SignBridgeGRUClassifier(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=num_classes,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        pooling_strategy=getattr(config, "pooling_strategy", "endpoint"),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 4. Training Loop with Early Stopping
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    start_time = time.time()
    best_checkpoint_path = checkpoint_dir / config.model_name

    print(f"\n[TRAIN] Commencing training: {config.epochs} epochs, batch_size={config.batch_size}, classes={num_classes}")
    print(f"[TRAIN] Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset) if val_dataset else 0}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        train_samples = 0
        train_correct = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(features, lengths)
            loss = criterion(logits, labels)

            loss.backward()
            if config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            train_samples += bs
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == labels).sum().item())

        epoch_train_loss = round(running_loss / max(train_samples, 1), 4)
        epoch_train_acc = round(train_correct / max(train_samples, 1), 4)

        # Validation phase
        epoch_val_loss = None
        epoch_val_acc = None
        epoch_val_f1 = None

        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                class_names=label_encoder.classes_,
            )
            epoch_val_loss = val_metrics["loss"]
            epoch_val_acc = val_metrics["accuracy"]
            epoch_val_f1 = val_metrics["macro_f1"]

            # Model checkpointing on validation macro-F1
            if epoch_val_f1 > best_val_f1:
                best_val_f1 = epoch_val_f1
                best_epoch = epoch
                patience_counter = 0

                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "model_config": model.get_config(),
                    "best_val_metrics": val_metrics,
                    "num_classes": num_classes,
                    "classes": label_encoder.classes_,
                }
                torch.save(checkpoint_payload, best_checkpoint_path)
            else:
                patience_counter += 1

        history_entry = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc,
            "val_macro_f1": epoch_val_f1,
        }
        history.append(history_entry)

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            val_str = f" | Val Loss: {epoch_val_loss}, Val Acc: {epoch_val_acc}, Val F1: {epoch_val_f1}" if val_loader else ""
            print(f"Epoch [{epoch:03d}/{config.epochs:03d}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}{val_str}")

        if val_loader and patience_counter >= config.early_stopping_patience:
            print(f"[EARLY STOPPING] Validation macro-F1 did not improve for {config.early_stopping_patience} epochs. Halting at epoch {epoch}.")
            break

    elapsed_time = round(time.time() - start_time, 2)

    # Save training history and config
    config.save(checkpoint_dir / "training_config.json")
    with open(checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "total_epochs_trained": len(history),
        "training_duration_sec": elapsed_time,
        "checkpoint_path": str(best_checkpoint_path),
        "history": history,
        "num_classes": num_classes,
    }
    return summary
