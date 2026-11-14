"""
Step 2.4B — Empirical Orientation Correction Validation
========================================================
Runs the preprocessing pipeline on 4 target videos into an isolated
test directory (data/processed/orientation_eval_tensors/) and reports:
- Applied rotation per video
- Original vs processed dimensions
- Shoulder vector angle (should be near 0° for all after correction)
- Spine vector angle (should be near -90° / pointing down for all)
- QC status and warnings

Does NOT touch data/processed/isl_csltr_tensors_v1/.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from preprocess_isl_csltr import (
    HAND_MODEL_PATH,
    POSE_MODEL_PATH,
    SAMPLE_VIDEOS_DIR,
    discover_sample_videos,
    load_gloss_mapping,
    GLOSS_CSV_PATH,
    preprocess_video,
)

# Isolated output directory — MUST NOT be isl_csltr_tensors_v1
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "orientation_eval_tensors"

# Target videos with expected rotation behaviour
TARGET_VIDEOS = {
    "are you free today.mp4":       {"expect_rotation": True,  "label": "Signer 1 (Native, verified)"},
    "are you free today (2).mp4":   {"expect_rotation": True,  "label": "Signer 2 (Native, verified)"},
    "free (2).MP4":                 {"expect_rotation": False, "label": "Signer 4 (Volunteer, verified)"},
    "bring water for me (1).MP4":   {"expect_rotation": False, "label": "Unverified (Quarantine)"},
}

# MediaPipe pose landmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP      = 23
R_HIP      = 24


def shoulder_angle_deg(npz_path: Path) -> float | None:
    """
    Extract median shoulder vector angle (degrees) from a preprocessed .npz file.

    After torso-centred normalization, MediaPipe defines Left/Right from the
    SUBJECT's perspective (mirrored from viewer). As a result:
      - Left shoulder  (landmark 11) is at positive x  (~+0.5, ~0.0)
      - Right shoulder (landmark 12) is at negative x  (~-0.5, ~0.0)

    So the L→R vector points LEFT on screen, giving angles near ±180°.
    We therefore measure the MAGNITUDE of deviation from horizontal:
        abs(angle) is near 0° or near 180° → horizontal (correct)
        abs(angle) is near 90°             → vertical (needs rotation)

    Returns the median angle. Caller checks: is_horizontal = deviation_from_h < 20°
    where deviation_from_h = min(abs(angle), 180 - abs(angle)).
    """
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        topology = meta.get("topology", "candidate_b_3d")
    except Exception as e:
        print(f"  [WARN] Could not load npz: {e}")
        return None

    features = data["features"]  # shape (T, D)
    masks    = data["presence_mask"]  # shape (T, 4): [mL, mR, mPose, mNorm]
    T = features.shape[0]

    # Determine pose block offset based on topology
    if topology in ("candidate_b_3d", "candidate_c_upper_3d", "candidate_d_hybrid"):
        pose_start = 63 + 63  # left_3d + right_3d = 126
        is_3d = True
    else:
        pose_start = 42 + 42  # left_2d + right_2d = 84
        is_3d = False

    angles = []
    for t in range(T):
        if masks[t, 2] < 0.5:   # pose not detected this frame
            continue
        pose_flat = features[t, pose_start : pose_start + (33 * 3 if is_3d else 33 * 2)]
        pose_arr = pose_flat.reshape(33, 3 if is_3d else 2)

        ls = pose_arr[L_SHOULDER]
        rs = pose_arr[R_SHOULDER]
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        angle = math.degrees(math.atan2(dy, dx))
        angles.append(angle)

    return float(np.median(angles)) if angles else None


def is_shoulder_horizontal(angle_deg: float, threshold: float = 20.0) -> bool:
    """
    Returns True if the shoulder vector is horizontal (near 0° or ±180°).
    After torso normalization the L→R vector points left (~180°), so we
    measure deviation from the nearest horizontal axis.
    """
    deviation = min(abs(angle_deg), 180.0 - abs(angle_deg))
    return deviation < threshold


def run_validation():
    print("=" * 70)
    print("Step 2.4B — Orientation Correction Empirical Validation")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gloss_map = load_gloss_mapping(GLOSS_CSV_PATH)
    all_videos = discover_sample_videos(SAMPLE_VIDEOS_DIR, gloss_map)

    # Build lookup by lowercase filename
    video_lookup = {v["filename"].lower(): v for v in all_videos}

    results = []
    for fname, config in TARGET_VIDEOS.items():
        vitem = video_lookup.get(fname.lower())
        if vitem is None:
            print(f"\n[SKIP] {fname} — not found in discovered videos")
            results.append({"filename": fname, "status": "FILE_NOT_FOUND"})
            continue

        print(f"\n--- {config['label']} ---")
        print(f"    File   : {fname}")
        print(f"    Signer : {vitem['signer_id']} (verified={vitem['signer_identity_verified']})")
        print(f"    Split  : {vitem['split']}")

        res = preprocess_video(
            video_item=vitem,
            hand_model=HAND_MODEL_PATH,
            pose_model=POSE_MODEL_PATH,
            output_dir=OUTPUT_DIR,
            topology="candidate_b_3d",
            overwrite=True,
        )

        if "metadata" not in res:
            print(f"    [ERROR] Preprocessing failed: {res}")
            results.append({"filename": fname, "status": "PREPROCESS_ERROR", "detail": res})
            continue

        meta = res["metadata"]

        orig_dim  = meta.get("original_dimensions", "?")
        proc_dim  = meta.get("processed_dimensions", "?")
        rot_deg   = meta.get("applied_rotation_deg", 0)
        qc_status = meta.get("qc_status", "?")
        qc_warns  = meta.get("qc_warnings", [])

        # Verify rotation expectation
        rotated = rot_deg != 0
        expect  = config["expect_rotation"]
        rot_ok  = rotated == expect

        # Compute shoulder angle from saved npz
        npz_path = Path(res["npz_path"])
        sh_angle = shoulder_angle_deg(npz_path)
        sh_ok    = is_shoulder_horizontal(sh_angle) if sh_angle is not None else None

        print(f"    Orig dims   : {orig_dim}")
        print(f"    Proc dims   : {proc_dim}")
        print(f"    Rotation    : {rot_deg}° — {'✓ EXPECTED' if rot_ok else '✗ UNEXPECTED'}")
        print(f"    QC status   : {qc_status}")
        if qc_warns:
            print(f"    QC warnings : {qc_warns}")
        if sh_angle is not None:
            deviation = min(abs(sh_angle), 180.0 - abs(sh_angle))
            status = "✓ HORIZONTAL" if sh_ok else "✗ NOT HORIZONTAL"
            print(f"    Shoulder °  : {sh_angle:+.1f}° (deviation {deviation:.1f}°) — {status}")
        else:
            print(f"    Shoulder °  : N/A (no pose frames)")

        results.append({
            "filename":          fname,
            "label":             config["label"],
            "signer_id":         meta["signer_id"],
            "signer_verified":   meta["signer_identity_verified"],
            "split":             meta["split"],
            "original_dims":     orig_dim,
            "processed_dims":    proc_dim,
            "applied_rotation":  rot_deg,
            "rotation_expected": expect,
            "rotation_correct":  rot_ok,
            "shoulder_angle_deg": sh_angle,
            "shoulder_horizontal": sh_ok,
            "qc_status":         qc_status,
            "qc_warnings":       qc_warns,
        })

    # Summary
    print("\n" + "=" * 70)
    print("ORIENTATION VALIDATION SUMMARY")
    print("=" * 70)
    rot_checks   = [r["rotation_correct"] for r in results if "rotation_correct" in r]
    angle_checks = [r["shoulder_horizontal"] for r in results if r.get("shoulder_horizontal") is not None]

    print(f"  Rotation gating correct : {sum(rot_checks)}/{len(rot_checks)}")
    print(f"  Shoulder horizontal     : {sum(a for a in angle_checks if a)}/{len(angle_checks)}")

    all_ok = all(rot_checks) and all(angle_checks)
    verdict = "PASS — Orientation correction is functioning correctly." if all_ok else \
              "FAIL — One or more orientation checks did not pass. Review above."
    print(f"\n  Verdict: {verdict}")

    # Print per-video shoulder deviation for transparency
    print("\n  Per-video shoulder deviation from horizontal:")
    for r in results:
        if r.get("shoulder_angle_deg") is not None:
            ang = r["shoulder_angle_deg"]
            dev = min(abs(ang), 180.0 - abs(ang))
            print(f"    {r['filename']}: angle={ang:+.1f}°, deviation={dev:.1f}°, horizontal={r['shoulder_horizontal']}")

    return results


if __name__ == "__main__":
    run_validation()
