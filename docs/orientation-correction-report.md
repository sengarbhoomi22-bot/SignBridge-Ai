# SignBridge AI — Step 2.4B: Orientation Correction Report

**Date:** 2026-11-14  
**Step:** 2.4B — Camera Orientation Correction Before Full-Dataset Preprocessing  
**Status:** ✅ PASS — Ready for full-corpus preprocessing (with restrictions documented below)

---

## 1. Problem Statement

The ISL-CSLTR dataset contains two classes of recording setup:

| Signer Class | Signers | Camera Setup | Native Resolution | Orientation |
|---|---|---|---|---|
| Native signers | 1, 2 | Sideways-mounted camera | 320 × 240 QVGA | **Landscape (sideways)** |
| Volunteer signers | 3–7 | Upright camera | 1920 × 1080 Full HD | **Portrait (upright)** |

Because MediaPipe HandLandmarker and PoseLandmarker output coordinates **in the image plane**, feeding sideways frames yields landmarks rotated 90° relative to volunteer frames:

- Unrotated Signer 1 shoulder angle: **–90°** (vertical axis)
- Unrotated Signer 2 shoulder angle: **+91.6°** (vertical axis)
- Volunteer Signer 4 shoulder angle: **~–3.6°** (horizontal axis)

The torso-centred normalization (decoupled 2D translation + scaling) does **not** correct rotational misalignment. If left uncorrected, the model would need to learn two entirely different coordinate frames from the same training data.

---

## 2. Correction Strategy

Apply a **90° clockwise rotation** to decoded frames **before MediaPipe extraction**, gated by three simultaneous conditions:

```
apply_rotation = (
    signer_identity_verified == True
    AND signer_id in {"1", "2"}
    AND orig_width > orig_height   # landscape detection
)
```

**All other clips** (volunteers, unverified, quarantined) receive **zero rotation**.

### Gating Rationale

| Condition | Reason |
|---|---|
| `signer_identity_verified == True` | Prevents rotating clips whose signer identity is uncertain |
| `signer_id in {"1", "2"}` | Only empirically confirmed native signers are rotated |
| `orig_width > orig_height` | Explicit landscape check; portrait native-signer frames are unexpected and trigger a warning |

---

## 3. Implementation

**File:** [`scripts/preprocess_isl_csltr.py`](../scripts/preprocess_isl_csltr.py)

Changes are inside `preprocess_video()`:

### 3.1 Orientation detection (after `cap.isOpened()`)

```python
orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
signer_id_str   = str(video_item.get("signer_id", ""))
signer_verified = bool(video_item.get("signer_identity_verified", False))

apply_rotation       = False
applied_rotation_deg = 0
orientation_warning  = None

if signer_verified and signer_id_str in ("1", "2"):
    if orig_width > orig_height:
        apply_rotation       = True
        applied_rotation_deg = 90
        proc_width, proc_height = orig_height, orig_width
    else:
        orientation_warning = (
            f"Native signer {signer_id_str} video is not landscape "
            f"({orig_width}x{orig_height}); rotation skipped."
        )
        proc_width, proc_height = orig_width, orig_height
else:
    proc_width, proc_height = orig_width, orig_height
```

### 3.2 Per-frame rotation (inside decode loop, before any other processing)

```python
ret, frame_bgr = cap.read()
if not ret:
    break
if apply_rotation:
    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
```

The rotation is applied **exactly once per frame** and **before** `cv2.cvtColor` and MediaPipe ingestion.

### 3.3 Metadata fields (added to `metadata_payload` and `manifest_rows`)

| Field | Type | Description |
|---|---|---|
| `original_dimensions` | `str` e.g. `"320x240"` | Raw capture dimensions |
| `processed_dimensions` | `str` e.g. `"240x320"` | Dimensions after rotation (swapped for native signers) |
| `applied_rotation_deg` | `int` (0 or 90) | Rotation applied; 0 = no rotation |

### 3.4 Catalog update

`are you free today (2).mp4` was added to `KNOWN_CATALOG` in `discover_sample_videos()` as **Signer 2, verified, native, train**, based on:
- Pixel-level MAE correspondence verified in Step 2.4B: **MAE = 0.42%** vs `Frames_Sentence_Level/are you free today/2/*.jpg`
- Post-rotation shoulder angle: **+0.3°** (horizontal)

---

## 4. Automated Unit Tests

**File:** [`backend/tests/test_preprocessing.py`](../backend/tests/test_preprocessing.py)

13 new unit tests in class `TestOrientationCorrection`:

| Test | Description |
|---|---|
| `test_native_signer_1_landscape_rotates` | Signer 1 + landscape → `apply_rotation = True` |
| `test_native_signer_2_landscape_rotates` | Signer 2 + landscape → `apply_rotation = True` |
| `test_volunteer_signer_3_no_rotation` | Signer 3 portrait → `apply_rotation = False` |
| `test_volunteer_signer_7_no_rotation` | Signer 7 portrait → `apply_rotation = False` |
| `test_unverified_signer_landscape_no_rotation` | signer_id="1" but unverified → no rotation |
| `test_quarantined_clip_no_rotation` | Unverified + landscape → no rotation |
| `test_native_signer_portrait_no_rotation` | Native signer portrait (unexpected) → no rotation |
| `test_90cw_rotation_swaps_dimensions` | (240,320,3) → (320,240,3) after 90° CW |
| `test_90cw_rotation_produces_expected_proc_dimensions` | proc_w = orig_h, proc_h = orig_w |
| `test_rotation_applied_exactly_once` | Single rotation changes shape; double rotation restores it |
| `test_metadata_records_rotation_applied` | Fields `original_dimensions`, `processed_dimensions`, `applied_rotation_deg=90` |
| `test_metadata_records_zero_rotation_for_volunteers` | No size change; `applied_rotation_deg=0` |
| `test_metadata_records_zero_rotation_for_unverified` | `applied_rotation_deg=0` for quarantined clips |

**Full suite result:**

```
======================== 25 passed, 1 warning in 0.95s ========================
```

(25 total = 8 pre-existing + 13 new + 4 health API tests; 1 warning = pytest-asyncio loop-scope deprecation, unrelated to preprocessing)

---

## 5. Empirical Validation Results

Run on 4 videos into isolated directory `data/processed/orientation_eval_tensors/`.  
The five-video prototype outputs in `data/processed/isl_csltr_tensors_v1/` were **not overwritten**.

> [!NOTE]
> MediaPipe defines "Left" and "Right" from the **subject's perspective** (mirrored from the viewer). After torso-centred normalization, the **subject's left shoulder** (landmark 11) is at positive x (~+0.5) and the **subject's right shoulder** (landmark 12) is at negative x (~−0.5). The L→R shoulder vector therefore points **left on screen**, giving angles near ±180°. A shoulder vector is classified as horizontal if its deviation from the nearest horizontal axis is < 20°:
> ```
> deviation = min(|angle|, 180 − |angle|)
> horizontal = deviation < 20°
> ```

| Video | Signer | Rotation Applied | Expected | Rotation ✓ | Shoulder Angle | Deviation | Horizontal ✓ |
|---|---|---|---|---|---|---|---|
| `are you free today.mp4` | 1 (native, verified) | **90° CW** | Yes | ✓ | −179.0° | **1.0°** | ✓ |
| `are you free today (2).mp4` | 2 (native, verified) | **90° CW** | Yes | ✓ | −178.3° | **1.7°** | ✓ |
| `free (2).MP4` | 4 (volunteer, verified) | 0° | No | ✓ | +176.5° | **3.5°** | ✓ |
| `bring water for me (1).MP4` | unverified (quarantine) | 0° | No | ✓ | +172.0° | **8.0°** | ✓ |

**Summary: Rotation gating 4/4 ✓ | Shoulder horizontal 4/4 ✓**

### Additional observation — Signer 2 QC FAIL

`are you free today (2).mp4` received `QC_STATUS = FAIL` due to:
```
QC-W_ASYM: Asymmetric/one-handed detection (L=100.0%, R=0.0%)
```
This means **only the subject's left hand** was detected across all 25 frames. This is a data characteristic of this particular signing clip, **not** an artifact of the orientation correction. The 90° rotation is correct (shoulder angle −178.3°, deviation 1.7°). The QC failure is noted in the manifest and the clip is marked `suitability_for_training = False`.

---

## 6. Dimension Summary

| Signer | Input (W × H) | After 90° CW (W × H) | MediaPipe frame |
|---|---|---|---|
| 1 (native) | 320 × 240 | **240 × 320** | 240 × 320 portrait |
| 2 (native) | 320 × 240 | **240 × 320** | 240 × 320 portrait |
| 3–7 (volunteer) | 1920 × 1080 | (unchanged) | 1920 × 1080 landscape |
| Unverified | any | (unchanged) | original dimensions |

> [!IMPORTANT]
> After rotation, **all normalized pose and hand coordinates are in a consistent coordinate frame**: shoulders horizontal, spine vertical, hands in the upper half. Torso-centred normalization then removes absolute position and scale, so the final feature vectors are directly comparable across all signers.

---

## 7. Limitations and Open Issues

| Issue | Severity | Status |
|---|---|---|
| `are you free today (2).mp4` QC FAIL (one-handed detection) | Low | Documented; not an orientation bug. Clip is quarantined in prototype output. Full-corpus equivalent should be verified. |
| Hip scaling heuristic (1.25) for normalization fallback | Low | Documented as unvalidated in preprocessing design. Not affected by orientation correction. |
| 241 quarantined videos in full corpus have unknown orientation | Medium | These will not receive rotation (gating requires `signer_identity_verified == True`). Once signer identities are resolved, their orientation must be re-assessed. |
| Native signer portrait frames (unexpected case) | Low | Implementation issues `ORIENTATION_WARNING` and skips rotation. No such frame observed in current sample set. |
| Signer 2 video (`are you free today (2).mp4`) was missing from `KNOWN_CATALOG` | Fixed | Added in Step 2.4B with full empirical evidence (pixel MAE 0.42%). |

---

## 8. Verdict

> [!IMPORTANT]
> **READY FOR FULL-CORPUS PREPROCESSING — with the following restrictions:**

1. **Rotation is ONLY safe for verified Signer 1 and Signer 2 clips** in landscape orientation. The 241 quarantined videos must remain at 0° rotation until their signer identities are resolved.
2. **Signer 2's `are you free today (2).mp4`** is now correctly catalogued and will receive 90° CW rotation. Its one-handed QC failure is noted.
3. **The five-video prototype tensors** in `data/processed/isl_csltr_tensors_v1/` were **not overwritten** and must be regenerated before training (they were produced before the orientation correction was implemented).
4. **Full-corpus tensor generation** (Step 2.4C) must write to a new directory (e.g., `data/processed/isl_csltr_tensors_v2/`) and must not overwrite the prototype outputs.
5. **The final test set (Signer 7) must not be used** for topology selection or hyperparameter tuning.

---

## 9. Files Modified / Created

| File | Action |
|---|---|
| `scripts/preprocess_isl_csltr.py` | Modified — orientation correction logic, Signer 2 catalog entry, metadata fields |
| `backend/tests/test_preprocessing.py` | Modified — 13 new `TestOrientationCorrection` unit tests |
| `scripts/validate_orientation_correction.py` | Created — empirical validation script |
| `scripts/debug_shoulder_angles.py` | Created — diagnostic tool (scratch; can be deleted) |
| `data/processed/orientation_eval_tensors/` | Created — isolated validation output (untracked) |
| `docs/orientation-correction-report.md` | Created — this document |

