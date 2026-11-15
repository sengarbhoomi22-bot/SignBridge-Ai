"""SignBridge AI: Step 3 Phase A — Comprehensive Pre-Flight ML Dataset Validator.

Validates the completed v2 preprocessing outputs before model training:
1. Directory structure and archive integrity.
2. Feature topology (T, 229) and numerical sanity (no NaN/Inf).
3. Zero-fabrication check: missing landmarks must be exactly 0.0.
   Feature layout (Candidate B 3D):
   - Left Hand:   [0:63]   (21 landmarks x 3)
   - Right Hand:  [63:126] (21 landmarks x 3)
   - Pose:        [126:225] (33 landmarks x 3)
   - Indicators:  [225:229] [m_left, m_right, m_pose, m_norm]
4. Modality presence indicators [m_left, m_right, m_pose, m_norm].
5. Signer isolation: Signers 1-5 in train, Signer 6 in val, Signer 7 exclusively in test.
6. Exclusion of quarantined and QC-failed clips from benchmark training.
   - For train: suitability_for_training == True and qc_pass == True
   - For val/test: qc_pass == True and signer_identity_verified == True (suitability_for_training is False by design)
7. Produces docs/ml-dataset-validation-report.md and reports/ml_dataset_validation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set
import numpy as np


def validate_ml_dataset(
    data_dir: Path = Path("data/processed/isl_csltr_tensors_v2"),
    expected_dim: int = 229,
    min_frames: int = 10,
) -> Dict[str, Any]:
    """Inspects and validates all processed tensors across train, validation, test, and quarantine."""
    print("=" * 70)
    print("SignBridge AI — Step 3: Phase A Pre-Flight ML Dataset Validation")
    print("=" * 70)
    print(f"Target Directory : {data_dir.resolve()}")
    print(f"Expected Feature Dimension : {expected_dim}")
    print(f"Minimum Sequence Length    : {min_frames} frames\n")

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed tensor directory does not exist: {data_dir}")

    splits = ["train", "validation", "test", "quarantine"]
    summary: Dict[str, Any] = {
        "status": "PASS",
        "data_dir": str(data_dir),
        "expected_dim": expected_dim,
        "splits": {},
        "global_checks": {
            "no_duplicate_clip_ids": True,
            "signer_7_strictly_isolated": True,
            "quarantine_excluded_from_train": True,
            "zero_fabrication_confirmed": True,
            "no_nan_or_inf": True,
        },
        "errors": [],
        "warnings": [],
    }

    seen_clip_ids: Dict[str, str] = {}
    all_categories: Set[str] = set()
    split_categories: Dict[str, Set[str]] = {s: set() for s in splits}

    for split in splits:
        split_path = data_dir / split
        if not split_path.exists():
            summary["splits"][split] = {"exists": False, "count": 0, "usable_count": 0}
            summary["warnings"].append(f"Split directory missing: {split}")
            continue

        npz_files = sorted(split_path.glob("*.npz"))
        split_stats = {
            "exists": True,
            "total_files": len(npz_files),
            "usable_for_ml": 0,
            "qc_failed": 0,
            "too_short": 0,
            "corrupted": 0,
            "signers": set(),
            "categories": set(),
            "min_frames_observed": 999999,
            "max_frames_observed": 0,
        }

        for npz_file in npz_files:
            clip_id = npz_file.stem
            # Duplicate check
            if clip_id in seen_clip_ids:
                summary["global_checks"]["no_duplicate_clip_ids"] = False
                summary["errors"].append(
                    f"Duplicate clip_id '{clip_id}' found in '{split}' (previously in '{seen_clip_ids[clip_id]}')"
                )
            else:
                seen_clip_ids[clip_id] = split

            try:
                with np.load(npz_file, allow_pickle=True) as archive:
                    if "features" not in archive or "metadata" not in archive:
                        split_stats["corrupted"] += 1
                        summary["errors"].append(f"{npz_file.name}: Missing features or metadata key.")
                        continue

                    features = archive["features"]
                    meta_raw = archive["metadata"]
                    meta = json.loads(str(meta_raw)) if isinstance(meta_raw, (str, np.ndarray)) else meta_raw

                # Dimension check
                if features.ndim != 2 or features.shape[1] != expected_dim:
                    summary["errors"].append(
                        f"{npz_file.name}: Invalid feature shape {features.shape}; expected (T, {expected_dim})."
                    )
                    split_stats["corrupted"] += 1
                    continue

                # NaN / Inf check
                if not np.isfinite(features).all():
                    summary["global_checks"]["no_nan_or_inf"] = False
                    summary["errors"].append(f"{npz_file.name}: Contains NaN or Inf values.")
                    split_stats["corrupted"] += 1
                    continue

                t_len = features.shape[0]
                split_stats["min_frames_observed"] = min(split_stats["min_frames_observed"], t_len)
                split_stats["max_frames_observed"] = max(split_stats["max_frames_observed"], t_len)

                # Zero-fabrication check
                # Correct Candidate B layout:
                # - Left hand:  [0:63]
                # - Right hand: [63:126]
                # - Pose:       [126:225]
                # - Indicators: [225:229] -> [m_left, m_right, m_pose, m_norm]
                left_indicator = features[:, 225]
                right_indicator = features[:, 226]
                left_hand_coords = features[:, 0:63]
                right_hand_coords = features[:, 63:126]

                # Where left indicator is 0.0, coordinates MUST be exactly 0.0
                left_missing_mask = (left_indicator == 0.0)
                if np.any(left_missing_mask):
                    if not np.all(left_hand_coords[left_missing_mask] == 0.0):
                        summary["global_checks"]["zero_fabrication_confirmed"] = False
                        summary["errors"].append(f"{npz_file.name}: Fabricated left hand landmarks detected when indicator=0.")

                right_missing_mask = (right_indicator == 0.0)
                if np.any(right_missing_mask):
                    if not np.all(right_hand_coords[right_missing_mask] == 0.0):
                        summary["global_checks"]["zero_fabrication_confirmed"] = False
                        summary["errors"].append(f"{npz_file.name}: Fabricated right hand landmarks detected when indicator=0.")

                # Signer and Category metadata
                signer_id = str(meta.get("signer_id", ""))
                split_stats["signers"].add(signer_id)
                category = meta.get("sentence_text", "").strip()
                if category:
                    split_stats["categories"].add(category)
                    all_categories.add(category)
                    split_categories[split].add(category)

                # Signer isolation checks
                if split == "train" and signer_id in ("6", "7"):
                    summary["errors"].append(f"{npz_file.name}: Validation/Test signer '{signer_id}' found in train split.")
                if split == "validation" and signer_id != "6":
                    summary["errors"].append(f"{npz_file.name}: Non-validation signer '{signer_id}' found in validation split.")
                if split == "test" and signer_id != "7":
                    summary["errors"].append(f"{npz_file.name}: Non-test signer '{signer_id}' found in test split.")
                if signer_id == "7" and split != "test":
                    summary["global_checks"]["signer_7_strictly_isolated"] = False
                    summary["errors"].append(f"{npz_file.name}: Signer 7 leaked into split '{split}'.")

                # Quality checks by partition
                is_suitable_for_train = meta.get("suitability_for_training", False)
                qc_pass = meta.get("qc_pass", False)
                signer_verified = meta.get("signer_identity_verified", False)

                if split == "train":
                    if not is_suitable_for_train or not qc_pass:
                        split_stats["qc_failed"] += 1
                        continue
                    if t_len < min_frames:
                        split_stats["too_short"] += 1
                        continue
                    split_stats["usable_for_ml"] += 1

                elif split in ("validation", "test"):
                    # Val and test samples must pass technical QC and have verified signer
                    # suitability_for_training is False by design to prevent data leakage into train
                    if not qc_pass or not signer_verified:
                        split_stats["qc_failed"] += 1
                        continue
                    if t_len < min_frames:
                        split_stats["too_short"] += 1
                        continue
                    split_stats["usable_for_ml"] += 1

                elif split == "quarantine":
                    if is_suitable_for_train:
                        summary["global_checks"]["quarantine_excluded_from_train"] = False
                        summary["errors"].append(f"{npz_file.name}: Quarantined clip marked as suitable for training.")

            except Exception as e:
                split_stats["corrupted"] += 1
                summary["errors"].append(f"{npz_file.name}: Exception during read: {e}")

        # Finalize split stats
        if split_stats["min_frames_observed"] == 999999:
            split_stats["min_frames_observed"] = 0
        split_stats["signers"] = sorted(list(split_stats["signers"]))
        split_stats["categories_count"] = len(split_stats["categories"])
        del split_stats["categories"]  # Avoid giant json
        summary["splits"][split] = split_stats

    summary["total_unique_categories"] = len(all_categories)
    summary["train_val_category_overlap"] = len(split_categories["train"].intersection(split_categories["validation"]))
    summary["train_test_category_overlap"] = len(split_categories["train"].intersection(split_categories["test"]))

    if summary["errors"]:
        summary["status"] = "FAIL"
    elif summary["warnings"]:
        summary["status"] = "PASS_WITH_WARNINGS"
    else:
        summary["status"] = "PASS"

    # Persist JSON report
    report_json_path = Path("reports/ml_dataset_validation.json")
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Machine-readable report saved to {report_json_path}")

    # Generate Markdown Report
    md_report_path = Path("docs/ml-dataset-validation-report.md")
    md_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# SignBridge AI: Pre-Flight ML Dataset Validation Report\n\n")
        f.write(f"**Overall Status:** `{summary['status']}`  \n")
        f.write(f"**Date:** November 2026  \n")
        f.write(f"**Dataset Directory:** `{data_dir}`  \n\n")

        f.write("## 1. Split Partition Summary\n\n")
        f.write("| Split | Total Files | Usable for ML | QC Failures | Too Short (<10f) | Signers Present | Categories |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in splits:
            st = summary["splits"].get(s, {})
            f.write(
                f"| **{s}** | {st.get('total_files', 0)} | {st.get('usable_for_ml', 0)} | "
                f"{st.get('qc_failed', 0)} | {st.get('too_short', 0)} | "
                f"{', '.join(st.get('signers', []))} | {st.get('categories_count', 0)} |\n"
            )

        f.write("\n## 2. Integrity & Scientific Verification Checks\n\n")
        for check, val in summary["global_checks"].items():
            f.write(f"- **{check}:** `{'PASS' if val else 'FAIL'}`\n")

        f.write(f"\n- **Total Unique Categories Across Splits:** {summary['total_unique_categories']}\n")
        f.write(f"- **Train-Validation Category Overlap:** {summary['train_val_category_overlap']}\n")
        f.write(f"- **Train-Test Category Overlap:** {summary['train_test_category_overlap']}\n")

        if summary["errors"]:
            f.write("\n## 3. Errors Detected\n\n")
            for err in summary["errors"][:20]:
                f.write(f"- ❌ {err}\n")
            if len(summary["errors"]) > 20:
                f.write(f"- ... and {len(summary['errors']) - 20} more errors.\n")

        if summary["warnings"]:
            f.write("\n## 4. Warnings\n\n")
            for warn in summary["warnings"]:
                f.write(f"- ⚠️ {warn}\n")

        f.write("\n## 5. Readiness Verdict\n\n")
        if summary["status"] == "PASS":
            f.write("✅ **APPROVED FOR MODEL TRAINING:** All splits strictly conform to Candidate B topology and signer isolation rules.\n")
        else:
            f.write("❌ **TRAINING BLOCKED:** Resolve detected integrity failures before initiating model training.\n")

    print(f"[OK] Markdown report generated at {md_report_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate processed ML dataset tensors.")
    parser.add_argument("--data-dir", type=str, default="data/processed/isl_csltr_tensors_v2")
    parser.add_argument("--expected-dim", type=int, default=229)
    parser.add_argument("--min-frames", type=int, default=10)
    args = parser.parse_args()

    try:
        res = validate_ml_dataset(Path(args.data_dir), args.expected_dim, args.min_frames)
        sys.exit(0 if res["status"] in ("PASS", "PASS_WITH_WARNINGS") else 1)
    except Exception as e:
        print(f"[FATAL] Validation script failed: {e}")
        sys.exit(2)
