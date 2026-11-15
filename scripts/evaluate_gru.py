"""SignBridge AI: Step 3 Phase H — Model Evaluation and Benchmark Reporting.

Evaluates:
1. Majority-class reference baseline.
2. Trained SignBridge GRU baseline model on the held-out Signer 7 test partition.
3. Generates docs/baseline-model-evaluation-report.md and reports/baseline_evaluation_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Any, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None  # type: ignore

from backend.app.ml.collate import SequenceBatchCollator
from backend.app.ml.dataset import ISLLandmarkDataset
from backend.app.ml.inference import SignBridgeInferenceEngine
from backend.app.ml.training.evaluate import evaluate_majority_class_baseline, evaluate_model


def run_evaluation(
    checkpoint_path: Path = Path("models/checkpoints/best_gru_model.pt"),
    data_dir: Path = Path("data/processed/isl_csltr_tensors_v2"),
    device_name: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Runs single-shot held-out test evaluation against majority baseline."""
    print("=" * 70)
    print("SignBridge AI — Step 3: Phase H Model Evaluation on Held-Out Test Set")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dataset    : {data_dir}")
    print(f"Device     : {device_name}\n")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load inference engine to inspect model and label mapping
    engine = SignBridgeInferenceEngine.load_from_checkpoint(checkpoint_path, device=device_name)
    model = engine.model
    label_encoder = engine.label_encoder
    num_classes = label_encoder.num_classes

    # Load training dataset labels for majority baseline
    train_dir = data_dir / "train"
    train_dataset = ISLLandmarkDataset(train_dir, label_encoder=label_encoder, filter_suitability=True)
    train_labels = [sample["label_id"] for sample in train_dataset]

    # Load test dataset (Signer 7)
    test_dir = data_dir / "test"
    test_dataset = ISLLandmarkDataset(test_dir, label_encoder=label_encoder, filter_suitability=True)
    test_labels = [sample["label_id"] for sample in test_dataset]

    if not test_labels:
        raise ValueError(f"No usable test samples found in {test_dir}.")

    print(f"[EVAL] Train Samples for Baseline: {len(train_labels)}")
    print(f"[EVAL] Held-Out Test Samples (Signer 7): {len(test_labels)}")
    print(f"[EVAL] Closed-Set Classes: {num_classes}\n")

    # 1. Majority-Class Baseline
    maj_metrics = evaluate_majority_class_baseline(
        train_labels=train_labels,
        eval_labels=test_labels,
        num_classes=num_classes,
        class_names=label_encoder.classes_,
    )
    print(f"[BASELINE] Majority-Class Baseline Accuracy: {maj_metrics['accuracy']:.4f}, Macro-F1: {maj_metrics['macro_f1']:.4f}")

    # 2. GRU Model Evaluation
    collator = SequenceBatchCollator(feature_dim=model.input_dim)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(device_name)

    gru_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=label_encoder.classes_,
    )
    print(f"[MODEL]    GRU Sequence Model Accuracy     : {gru_metrics['accuracy']:.4f}, Macro-F1: {gru_metrics['macro_f1']:.4f}, Loss: {gru_metrics['loss']:.4f}")

    report_payload = {
        "status": "COMPLETED",
        "num_classes": num_classes,
        "test_samples_count": len(test_labels),
        "majority_class_baseline": maj_metrics,
        "gru_model": gru_metrics,
        "classes": label_encoder.classes_,
    }

    # Persist JSON report
    report_json = Path("reports/baseline_evaluation_report.json")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n[OK] Machine-readable evaluation report saved to {report_json}")

    # Generate Markdown Report
    md_path = Path("docs/baseline-model-evaluation-report.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SignBridge AI: Baseline Model Evaluation Report\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("This report presents the empirical performance of the first machine learning baseline for Indian Sign Language (ISL) sentence recognition, evaluated on the held-out Signer 7 test partition.\n\n")

        f.write("## 2. Comparative Benchmark Results\n\n")
        f.write("| Model / Approach | Overall Accuracy | Macro Precision | Macro Recall | Macro F1 | Test Loss |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(
            f"| **Majority-Class Baseline** | {maj_metrics['accuracy']:.4f} | "
            f"{maj_metrics['macro_precision']:.4f} | {maj_metrics['macro_recall']:.4f} | "
            f"{maj_metrics['macro_f1']:.4f} | N/A |\n"
        )
        f.write(
            f"| **SignBridge GRU Sequence Model** | {gru_metrics['accuracy']:.4f} | "
            f"{gru_metrics['macro_precision']:.4f} | {gru_metrics['macro_recall']:.4f} | "
            f"{gru_metrics['macro_f1']:.4f} | {gru_metrics['loss']:.4f} |\n"
        )

        f.write("\n## 3. Test Partition Characteristics\n\n")
        f.write(f"- **Evaluated Split:** Signer 7 (Held-out volunteer signer, strictly isolated from training/validation)\n")
        f.write(f"- **Total Evaluated Test Samples:** {len(test_labels)}\n")
        f.write(f"- **Correctly Classified Samples:** {gru_metrics['correct_samples']} / {gru_metrics['total_samples']}\n")
        f.write(f"- **Total Target Vocabulary Categories:** {num_classes}\n")

        f.write("\n## 4. Key Scientific Observations & Limitations\n\n")
        f.write("1. **Closed-Set Formulation:** The model is an isolated sequence classifier over the trained sentence categories, not an open-vocabulary continuous sign translator.\n")
        f.write("2. **Signer Independence:** Testing on Signer 7 measures generalization to a new, unseen signing physiology and movement rhythm.\n")
        f.write("3. **Small Sample Support:** Due to the single-take nature of several categories in the dataset, per-class test support is modest (~1 sample per category in test).\n")

    print(f"[OK] Markdown report generated at {md_path}")
    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SignBridge GRU model.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_gru_model.pt")
    parser.add_argument("--data-dir", type=str, default="data/processed/isl_csltr_tensors_v2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        run_evaluation(Path(args.checkpoint), Path(args.data_dir), args.device, args.batch_size)
    except Exception as e:
        print(f"[FATAL] Evaluation script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
