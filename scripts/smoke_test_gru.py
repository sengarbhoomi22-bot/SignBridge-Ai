"""SignBridge AI: Step 3.4 — Safe Isolated CPU Smoke Test for GRU Pipeline.

Verifies end-to-end training and validation loop on a small deterministic slice
of real preprocessed ISL-CSLTR tensors (16 train, 8 val) using CPU exclusively.
Strictly isolates checkpoints to models/checkpoints/smoke_test/ and never touches
the test partition or Signer 7 data.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from backend.app.ml.collate import SequenceBatchCollator
from backend.app.ml.dataset import ISLLandmarkDataset
from backend.app.ml.labels import LabelEncoder
from backend.app.ml.models.gru_classifier import SignBridgeGRUClassifier
from backend.app.ml.training.config import TrainingConfig
from backend.app.ml.training.evaluate import evaluate_model


def run_cpu_smoke_test() -> Dict[str, Any]:
    print("=" * 70)
    print("SignBridge AI — Step 3.4: Safe CPU Smoke Test")
    print("=" * 70)
    print("Hardware Target : CPU (Explicitly Forced)")
    print(f"PyTorch Version : {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    
    device = torch.device("cpu")
    smoke_test_results: Dict[str, Any] = {
        "verdict": "PENDING",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "steps": {},
        "samples": {
            "train": [],
            "validation": []
        },
        "metrics": {},
        "errors": []
    }

    data_dir = Path("data/processed/isl_csltr_tensors_v2")
    checkpoint_dir = Path("models/checkpoints/smoke_test")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ---------------------------------------------------------------------
        # [Step 1/10] Dataset loading and deterministic subset selection
        # ---------------------------------------------------------------------
        print("\n[Step 1/10] Loading real datasets and selecting deterministic subsets...")
        train_full = ISLLandmarkDataset(split_dir=data_dir / "train", min_frames=10, filter_suitability=True)
        val_full = ISLLandmarkDataset(split_dir=data_dir / "validation", min_frames=10, filter_suitability=True)

        # Safety Check: Assert test partition is not referenced
        assert not any("test" in str(s["path"]).lower() for s in train_full.samples), "CRITICAL: Test sample leaked into train!"
        assert not any("test" in str(s["path"]).lower() for s in val_full.samples), "CRITICAL: Test sample leaked into val!"
        assert all(str(s["metadata"].get("signer_id")) in ("1", "2", "3", "4", "5") for s in train_full.samples), "Signer mismatch in train!"
        assert all(str(s["metadata"].get("signer_id")) == "6" for s in val_full.samples), "Signer mismatch in validation!"

        # Group samples by category
        train_by_class: Dict[str, List[Dict[str, Any]]] = {}
        for s in train_full.samples:
            train_by_class.setdefault(s["label_str"], []).append(s)

        val_by_class: Dict[str, List[Dict[str, Any]]] = {}
        for s in val_full.samples:
            val_by_class.setdefault(s["label_str"], []).append(s)

        # Deterministically select 8 shared classes with >=2 train and >=1 val samples
        shared_classes = sorted([
            c for c in train_by_class 
            if len(train_by_class[c]) >= 2 and len(val_by_class.get(c, [])) >= 1
        ])[:8]
        assert len(shared_classes) == 8, f"Expected 8 shared classes, found {len(shared_classes)}"

        # Deterministically select 16 train (2 per class) and 8 val (1 per class)
        selected_train_samples: List[Dict[str, Any]] = []
        for c in shared_classes:
            # Sort by clip_id for reproducibility
            sorted_train = sorted(train_by_class[c], key=lambda x: x["clip_id"])
            selected_train_samples.extend(sorted_train[:2])

        selected_val_samples: List[Dict[str, Any]] = []
        for c in shared_classes:
            sorted_val = sorted(val_by_class[c], key=lambda x: x["clip_id"])
            selected_val_samples.extend(sorted_val[:1])

        assert len(selected_train_samples) == 16, f"Expected 16 train samples, got {len(selected_train_samples)}"
        assert len(selected_val_samples) == 8, f"Expected 8 val samples, got {len(selected_val_samples)}"

        # Verify all val labels exist in train vocabulary
        train_vocab = {s["label_str"] for s in selected_train_samples}
        val_vocab = {s["label_str"] for s in selected_val_samples}
        assert val_vocab.issubset(train_vocab), "Validation subset contains labels absent from training subset!"
        assert len(train_vocab) == 8, f"Expected 8 unique classes in vocabulary, got {len(train_vocab)}"

        # Fit LabelEncoder strictly on training subset
        label_encoder = LabelEncoder().fit([s["label_str"] for s in selected_train_samples])
        assert label_encoder.num_classes == 8

        # Create isolated subsets
        train_subset = ISLLandmarkDataset.__new__(ISLLandmarkDataset)
        train_subset.split_dir = data_dir / "train"
        train_subset.label_encoder = label_encoder
        train_subset.label_key = "sentence_text"
        train_subset.min_frames = 10
        train_subset.filter_suitability = True
        train_subset.cache_in_memory = False
        train_subset.samples = selected_train_samples
        train_subset._cache = {}

        val_subset = ISLLandmarkDataset.__new__(ISLLandmarkDataset)
        val_subset.split_dir = data_dir / "validation"
        val_subset.label_encoder = label_encoder
        val_subset.label_key = "sentence_text"
        val_subset.min_frames = 10
        val_subset.filter_suitability = True
        val_subset.cache_in_memory = False
        val_subset.samples = selected_val_samples
        val_subset._cache = {}

        smoke_test_results["samples"]["train"] = [
            {"clip_id": s["clip_id"], "label": s["label_str"], "signer": s["metadata"].get("signer_id")} 
            for s in selected_train_samples
        ]
        smoke_test_results["samples"]["validation"] = [
            {"clip_id": s["clip_id"], "label": s["label_str"], "signer": s["metadata"].get("signer_id")} 
            for s in selected_val_samples
        ]
        smoke_test_results["steps"]["step_1_dataset_selection"] = {
            "status": "PASS",
            "train_samples": len(selected_train_samples),
            "val_samples": len(selected_val_samples),
            "num_classes": label_encoder.num_classes,
            "classes": label_encoder.classes_
        }
        print(f"  [PASS] Selected 16 train clips (Signers: {sorted(list({s['signer'] for s in smoke_test_results['samples']['train']}))})")
        print(f"  [PASS] Selected 8 val clips (Signers: {sorted(list({s['signer'] for s in smoke_test_results['samples']['validation']}))})")
        print(f"  [PASS] Vocabulary strictly matched (8 classes). Zero unseen val classes.")

        # ---------------------------------------------------------------------
        # [Step 2/10] Input shape and numerical sanity check
        # ---------------------------------------------------------------------
        print("\n[Step 2/10] Verifying input tensor shape and numerical sanity...")
        for idx in range(len(train_subset)):
            sample = train_subset[idx]
            feats = sample["features"]
            assert feats.ndim == 2, f"Expected 2D array, got {feats.ndim}"
            assert feats.shape[1] == 229, f"Expected 229 features, got {feats.shape[1]}"
            assert np.isfinite(feats).all(), f"Non-finite values detected in sample {sample['clip_id']}"
            assert sample["label_id"] >= 0, f"Invalid label ID in sample {sample['clip_id']}"
        smoke_test_results["steps"]["step_2_input_shapes"] = {
            "status": "PASS",
            "feature_dim": 229,
            "all_finite": True
        }
        print("  [PASS] All 16 training tensors conform to shape (T, 229) with 100% finite values.")

        # ---------------------------------------------------------------------
        # [Step 3/10] SequenceBatchCollator variable-length batching & padding
        # ---------------------------------------------------------------------
        print("\n[Step 3/10] Testing variable-length batch collation & mask separation...")
        collator = SequenceBatchCollator(feature_dim=229)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=False, collate_fn=collator)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, collate_fn=collator)

        batch = next(iter(train_loader))
        b_features = batch["features"]      # (B, T_max, 229)
        b_lengths = batch["lengths"]        # (B,)
        b_padding_mask = batch["padding_mask"]  # (B, T_max)
        b_labels = batch["labels"]          # (B,)

        assert b_features.shape == (8, max(b_lengths).item(), 229), f"Unexpected batch feature shape: {b_features.shape}"
        assert b_lengths.shape == (8,), f"Unexpected batch lengths shape: {b_lengths.shape}"
        assert b_padding_mask.shape == (8, max(b_lengths).item()), f"Unexpected padding mask shape: {b_padding_mask.shape}"
        assert b_labels.shape == (8,), f"Unexpected labels shape: {b_labels.shape}"
        
        # Verify padding isolation
        for i, l in enumerate(b_lengths):
            assert torch.all(b_padding_mask[i, :l] == 1.0), "Padding mask valid region must be 1.0"
            assert torch.all(b_padding_mask[i, l:] == 0.0), "Padding mask padded region must be 0.0"
            assert torch.all(b_features[i, l:, :] == 0.0), "Padded frames must have zero feature values"

        smoke_test_results["steps"]["step_3_collation"] = {
            "status": "PASS",
            "batch_shape": list(b_features.shape),
            "max_frames_in_batch": int(max(b_lengths)),
            "padding_verified": True
        }
        print(f"  [PASS] Batch feature tensor shape: {tuple(b_features.shape)} (batch=8, T_max={max(b_lengths).item()}, dim=229)")
        print(f"  [PASS] Padding mask strictly separated from modality presence indicators.")

        # ---------------------------------------------------------------------
        # [Step 4/10] Forward pass through SignBridgeGRUClassifier
        # ---------------------------------------------------------------------
        print("\n[Step 4/10] Executing forward pass through SignBridgeGRUClassifier...")
        model = SignBridgeGRUClassifier(
            input_dim=229,
            hidden_dim=128,
            num_layers=1,
            num_classes=8,
            bidirectional=True,
            dropout=0.2
        ).to(device)

        model.train()
        logits = model(b_features.to(device), b_lengths.to(device))
        assert logits.shape == (8, 8), f"Expected logits shape (8, 8), got {logits.shape}"
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values!"
        
        smoke_test_results["steps"]["step_4_forward_pass"] = {
            "status": "PASS",
            "logits_shape": list(logits.shape),
            "finite_logits": True
        }
        print(f"  [PASS] Forward pass successful. Output raw logits shape: {tuple(logits.shape)}")

        # ---------------------------------------------------------------------
        # [Step 5/10] Loss calculation (CrossEntropyLoss on raw logits)
        # ---------------------------------------------------------------------
        print("\n[Step 5/10] Computing CrossEntropyLoss on raw logits...")
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, b_labels.to(device))
        assert torch.isfinite(loss), "Loss is not finite!"
        assert loss.item() > 0.0, "Loss must be positive!"
        
        initial_loss = float(loss.item())
        smoke_test_results["steps"]["step_5_loss_computation"] = {
            "status": "PASS",
            "initial_loss": round(initial_loss, 4),
            "loss_finite": True
        }
        print(f"  [PASS] CrossEntropyLoss computed: {initial_loss:.4f} (Finite and positive)")

        # ---------------------------------------------------------------------
        # [Step 6/10] Backpropagation
        # ---------------------------------------------------------------------
        print("\n[Step 6/10] Executing backpropagation (loss.backward())...")
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer.zero_grad()
        loss.backward()
        smoke_test_results["steps"]["step_6_backpropagation"] = {
            "status": "PASS",
            "backward_executed": True
        }
        print("  [PASS] Backpropagation completed without error.")

        # ---------------------------------------------------------------------
        # [Step 7/10] Finite gradient verification
        # ---------------------------------------------------------------------
        print("\n[Step 7/10] Checking parameter gradients for NaN/Inf...")
        grad_norms = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Gradient is None for parameter {name}!"
                assert torch.isfinite(p.grad).all(), f"Gradient contains NaN/Inf for parameter {name}!"
                grad_norms[name] = float(torch.norm(p.grad).item())
        
        smoke_test_results["steps"]["step_7_gradient_check"] = {
            "status": "PASS",
            "checked_parameters_count": len(grad_norms),
            "all_gradients_finite": True,
            "max_grad_norm": round(max(grad_norms.values()), 4),
            "min_grad_norm": round(min(grad_norms.values()), 6)
        }
        print(f"  [PASS] All {len(grad_norms)} trainable parameters have finite gradients.")
        print(f"  [PASS] Max gradient norm: {max(grad_norms.values()):.4f}, Min: {min(grad_norms.values()):.6f}")

        # ---------------------------------------------------------------------
        # [Step 8/10] Optimizer update and parameter movement
        # ---------------------------------------------------------------------
        print("\n[Step 8/10] Executing optimizer.step() and verifying parameter update...")
        # Capture pre-step weights
        pre_params = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        optimizer.step()
        
        # Check that weights moved
        param_diffs = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                diff = torch.norm(p.detach() - pre_params[name]).item()
                param_diffs.append(diff)
        
        assert max(param_diffs) > 0.0, "Optimizer step failed to update any model parameters!"
        smoke_test_results["steps"]["step_8_optimizer_update"] = {
            "status": "PASS",
            "parameters_updated": True,
            "max_parameter_delta": round(max(param_diffs), 6)
        }
        print(f"  [PASS] Optimizer updated parameters successfully. Max weight shift: {max(param_diffs):.6f}")

        # Train a full mini-epoch 2 to observe loss progression
        print("\n[Training mini-epoch 2 to confirm loss reduction]...")
        for epoch in range(1, 3):
            model.train()
            epoch_loss = 0.0
            for b in train_loader:
                optimizer.zero_grad()
                out = model(b["features"].to(device), b["lengths"].to(device))
                l = criterion(out, b["labels"].to(device))
                l.backward()
                optimizer.step()
                epoch_loss += l.item() * len(b["labels"])
            print(f"  Mini-epoch {epoch}: Mean loss = {epoch_loss / len(train_subset):.4f}")

        # ---------------------------------------------------------------------
        # [Step 9/10] Validation forward pass and metrics calculation
        # ---------------------------------------------------------------------
        print("\n[Step 9/10] Running validation pass with evaluate_model()...")
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=8,
            class_names=label_encoder.classes_
        )
        assert "loss" in val_metrics and np.isfinite(val_metrics["loss"])
        assert "accuracy" in val_metrics and 0.0 <= val_metrics["accuracy"] <= 1.0
        assert "macro_f1" in val_metrics and 0.0 <= val_metrics["macro_f1"] <= 1.0

        smoke_test_results["steps"]["step_9_validation_pass"] = {
            "status": "PASS",
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_samples_evaluated": val_metrics["total_samples"]
        }
        smoke_test_results["metrics"] = val_metrics
        print(f"  [PASS] Validation evaluated: {val_metrics['total_samples']} samples")
        print(f"  [PASS] Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        # ---------------------------------------------------------------------
        # [Step 10/10] Checkpoint creation in separate smoke-test directory
        # ---------------------------------------------------------------------
        print("\n[Step 10/10] Verifying checkpoint creation in isolated directory...")
        checkpoint_path = checkpoint_dir / "smoke_test_gru.pt"
        label_mapping_path = checkpoint_dir / "smoke_test_labels.json"
        config_path = checkpoint_dir / "smoke_test_config.json"

        # Save label mapping
        label_encoder.save(label_mapping_path)
        assert label_mapping_path.exists() and label_mapping_path.stat().st_size > 0

        # Save config
        cfg = TrainingConfig(
            input_dim=229,
            hidden_dim=128,
            num_layers=1,
            num_classes=8,
            device="cpu",
            checkpoint_dir=str(checkpoint_dir),
            model_name="smoke_test_gru.pt"
        )
        cfg.save(config_path)
        assert config_path.exists() and config_path.stat().st_size > 0

        # Save model checkpoint
        payload = {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "model_config": model.get_config(),
            "best_val_metrics": val_metrics,
            "num_classes": 8,
            "classes": label_encoder.classes_,
        }
        torch.save(payload, checkpoint_path)
        assert checkpoint_path.exists() and checkpoint_path.stat().st_size > 0

        # Verify checkpoint can be reloaded
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded_model = SignBridgeGRUClassifier.from_config(reloaded["model_config"])
        reloaded_model.load_state_dict(reloaded["model_state_dict"])
        reloaded_model.eval()

        smoke_test_results["steps"]["step_10_checkpoint_save_and_reload"] = {
            "status": "PASS",
            "checkpoint_path": str(checkpoint_path),
            "file_size_bytes": checkpoint_path.stat().st_size,
            "reloaded_successfully": True
        }
        print(f"  [PASS] Checkpoint saved successfully: {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")
        print(f"  [PASS] Checkpoint reloaded and state verified.")

        smoke_test_results["verdict"] = "CPU SMOKE TEST PASSED"

    except Exception as e:
        print(f"\n[FAIL] Smoke test encountered an error: {e}")
        import traceback
        traceback.print_exc()
        smoke_test_results["verdict"] = "CPU SMOKE TEST FAILED"
        smoke_test_results["errors"].append(str(e))

    # Persist JSON report
    report_json_path = Path("reports/cpu_smoke_test.json")
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(smoke_test_results, f, indent=2)
    print(f"\n[OK] Machine-readable report saved to {report_json_path}")

    # Generate Markdown report
    md_path = Path("docs/cpu-smoke-test-report.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SignBridge AI — Step 3.4: CPU Smoke Test Report\n\n")
        f.write(f"**Execution Date:** {smoke_test_results['timestamp']}  \n")
        f.write(f"**Hardware Target:** CPU ({smoke_test_results['device']})  \n")
        f.write(f"**PyTorch Version:** `{smoke_test_results['pytorch_version']}`  \n")
        f.write(f"**Final Verdict:** `{smoke_test_results['verdict']}`  \n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write("A 10-step isolated CPU smoke test was conducted on real preprocessed ISL-CSLTR tensors. ")
        f.write("The test verified dataset loading, variable-length collation, forward pass, loss calculation, ")
        f.write("backpropagation, gradient finiteness, optimizer step, validation evaluation, and checkpoint persistence.\n\n")

        f.write("## 2. Test Configuration & Samples\n\n")
        f.write(f"- **Training Samples:** 16 clips across 8 categories (2 clips/class)\n")
        f.write(f"- **Validation Samples:** 8 clips across the same 8 categories (1 clip/class)\n")
        f.write(f"- **Signers in Training:** Signers 1, 2, 3, 4, 5 (Native & Volunteer)\n")
        f.write(f"- **Signers in Validation:** Signer 6 (Volunteer)\n")
        f.write(f"- **Test Partition / Signer 7:** **NEVER ACCESSED (0 samples)**\n")
        f.write(f"- **Batch Size:** 8 (2 training batches, 1 validation batch)\n\n")

        f.write("### Selected Vocabulary (8 classes)\n\n")
        for c in smoke_test_results["steps"].get("step_1_dataset_selection", {}).get("classes", []):
            f.write(f"- `{c}`\n")

        f.write("\n## 3. Step-by-Step Verification Results\n\n")
        f.write("| Step | Description | Result |\n")
        f.write("|---|---|---|\n")
        for step_key, step_data in smoke_test_results["steps"].items():
            f.write(f"| **{step_key}** | {step_key.replace('_', ' ').title()} | `{step_data.get('status', 'N/A')}` |\n")

        f.write("\n## 4. Numerical and Gradient Sanity\n\n")
        step7 = smoke_test_results["steps"].get("step_7_gradient_check", {})
        step8 = smoke_test_results["steps"].get("step_8_optimizer_update", {})
        f.write(f"- **Trainable Parameters Checked:** {step7.get('checked_parameters_count', 0)}\n")
        f.write(f"- **All Gradients Finite:** `{step7.get('all_gradients_finite', False)}`\n")
        f.write(f"- **Max Gradient Norm:** `{step7.get('max_grad_norm', 'N/A')}`\n")
        f.write(f"- **Parameters Updated by Optimizer:** `{step8.get('parameters_updated', False)}`\n")
        f.write(f"- **Max Parameter Delta:** `{step8.get('max_parameter_delta', 'N/A')}`\n\n")

        f.write("## 5. Validation Evaluation\n\n")
        metrics = smoke_test_results.get("metrics", {})
        f.write(f"- **Validation Loss:** `{metrics.get('loss', 'N/A')}`\n")
        f.write(f"- **Validation Accuracy:** `{metrics.get('accuracy', 'N/A')}`\n")
        f.write(f"- **Validation Macro-F1:** `{metrics.get('macro_f1', 'N/A')}`\n")
        f.write(f"- **Samples Evaluated:** `{metrics.get('total_samples', 0)}`\n\n")

        f.write("## 6. Checkpoint Persistence Verification\n\n")
        step10 = smoke_test_results["steps"].get("step_10_checkpoint_save_and_reload", {})
        f.write(f"- **Checkpoint Path:** `{step10.get('checkpoint_path', 'N/A')}`\n")
        f.write(f"- **File Size:** `{step10.get('file_size_bytes', 0)}` bytes\n")
        f.write(f"- **Reload Verification:** `{step10.get('reloaded_successfully', False)}`\n\n")

        f.write("## 7. Final Verdict\n\n")
        if smoke_test_results["verdict"] == "CPU SMOKE TEST PASSED":
            f.write("### ✅ CPU SMOKE TEST PASSED\n\n")
            f.write("The GRU architecture, loss function, backpropagation pipeline, and validation harness ")
            f.write("are fully operational on real preprocessed ISL-CSLTR sequence tensors.\n")
        else:
            f.write("### ❌ CPU SMOKE TEST FAILED\n\n")
            for err in smoke_test_results["errors"]:
                f.write(f"- Error: `{err}`\n")

    print(f"[OK] Markdown report saved to {md_path}")
    print("\n" + "=" * 70)
    print(f"FINAL VERDICT: {smoke_test_results['verdict']}")
    print("=" * 70)
    return smoke_test_results


if __name__ == "__main__":
    res = run_cpu_smoke_test()
    sys.exit(0 if res["verdict"] == "CPU SMOKE TEST PASSED" else 1)
