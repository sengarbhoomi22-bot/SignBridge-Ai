"""SignBridge AI: Step 3 Phase F — GRU Baseline Model Training Script."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.app.ml.training.config import TrainingConfig
from backend.app.ml.training.train import train_gru


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SignBridge AI GRU baseline sequence classifier.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GRU hidden state dimension.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of recurrent layers.")
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectionality.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--pooling-strategy", type=str, default="endpoint", choices=["endpoint", "mean", "attention"], help="Sequence aggregation strategy ('endpoint', 'mean', or 'attention').")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--data-dir", type=str, default="data/processed/isl_csltr_tensors_v2", help="Path to processed tensors.")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Path to save checkpoints.")
    parser.add_argument("--model-name", type=str, default="best_gru_model.pt", help="Checkpoint filename.")

    args = parser.parse_args()

    config = TrainingConfig(
        seed=args.seed,
        input_dim=229,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
        pooling_strategy=args.pooling_strategy,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        device=args.device,
    )

    print("=" * 70)
    print("SignBridge AI — Step 3: Train GRU Sequence Baseline")
    print("=" * 70)
    print(f"Device Mode   : {config.device}")
    print(f"Data Dir      : {config.data_dir}")
    print(f"Epochs        : {config.epochs}")
    print(f"Batch Size    : {config.batch_size}")
    print(f"Learning Rate : {config.learning_rate}")
    print(f"Hidden Dim    : {config.hidden_dim} (Bi-GRU: {config.bidirectional})")
    print(f"Pooling Mode  : {config.pooling_strategy}")
    print()

    try:
        summary = train_gru(config)
        print("\n" + "=" * 70)
        print("Training Completed Successfully")
        print("=" * 70)
        print(f"Best Epoch            : {summary['best_epoch']}")
        print(f"Best Val Macro-F1     : {summary['best_val_macro_f1']:.4f}")
        print(f"Total Epochs Run      : {summary['total_epochs_trained']}")
        print(f"Duration (seconds)    : {summary['training_duration_sec']}s")
        print(f"Best Checkpoint Saved : {summary['checkpoint_path']}")
    except Exception as e:
        print(f"\n[FATAL] Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
