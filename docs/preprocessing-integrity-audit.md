# SignBridge AI: Preprocessing Integrity & Signer Assignment Audit

**Document ID:** `docs/preprocessing-integrity-audit.md`
**Milestone:** Step 2.3B — Data Integrity, Signer Identity, and Quality Control Audit
**Author:** SignBridge AI ML Engineering
**Date:** November 2026
**Status:** Completed & Empirically Verified

---

## 1. Executive Summary

This audit establishes the empirical ground truth for signer attribution, split routing, untrimmed temporal boundaries, and quality control (QC) criteria for the SignBridge AI preprocessing pipeline prior to Step 2.4 (full-dataset preprocessing).

### Key Audit Findings & Empirical Proof:
1. **Pixel-Level Proof of Video-to-Frame Correspondence**:
   Using direct byte-range extractions from the published Mendeley archive zip, we compared extracted frames against video frames via Mean Absolute Error (MAE):
   - `are you free today.mp4` matches `Frames_Sentence_Level/are you free today/1/are you free today 01.jpg` at Video Frame 0 (rotated 90° CW) with **MAE = 1.5155 / 255** (0.59% error, pixel match).
   - `free (2).MP4` matches `Frames_Sentence_Level/are you free today/4/free (2) 01.jpg` at Video Frame 2 (original orientation) with **MAE = 1.0632 / 255** (0.41% error, pixel match).
   - `MVI_6209.MP4` matches `Frames_Sentence_Level/are you hiding something/7/MVI_6209 01.jpg` at Video Frame 2 (original orientation) with **MAE = 1.0855 / 255** (0.42% error, pixel match).
2. **Kinematic Proof of Untrimmed Video Durations**:
   Temporal landmark analysis confirms that `bring water for me (1).MP4` and `can you repeat that please (3).MP4` consist of ~1.4s of idle rest postures at the beginning, ~1.4s of dense active signing in the middle (where right hand detection reaches **76.3% to 97.2%**), and ~1.5s of resting posture at the end.
3. **Quarantine Isolation for Ambiguous Takes**:
   Videos without unambiguous frame matches (`bring water for me (1).MP4` and `can you repeat that please (3).MP4`) are quarantined to `data/processed/isl_csltr_tensors_v1/quarantine/` with `suitability_for_training = False`.
4. **Three-Tier QC Decoupling**:
   Technical Validity, Landmark Coverage, and Training Suitability are strictly decoupled. One-handed asymmetric signs are preserved and flagged with `QC-W_ASYM` rather than rejected.
5. **Full Test Suite Clean**:
   All 12 backend and preprocessing tests pass in 0.89s.

---

## 2. Empirical Verification of Video-to-Frame Correspondence

### 2.1 Methodology
To avoid relying on filename similarity alone, we extracted the ground-truth JPEG images directly from the Mendeley `ISL_CSLRT_Corpus.zip` archive using exact byte-range Deflate decompilation and compared them against the full video frame sequences using OpenCV:
- Reference images:
  1. `are you free today 01.jpg` (Folder `1/`, Native Signer 1)
  2. `free (2) 01.jpg` (Folder `4/`, Student Volunteer 2)
  3. `MVI_6209 01.jpg` (Folder `7/`, Student Volunteer 5)
- Every video frame was evaluated across 4 rotation transforms: Original, 90° Clockwise, 90° Counter-Clockwise, and 180°.
- Pixel error was quantified via Mean Absolute Error:
  $$\text{MAE} = \frac{1}{H \times W \times C} \sum_{y=1}^H \sum_{x=1}^W \sum_{c=1}^C |I_{\text{video}}(x, y, c) - I_{\text{ref}}(x, y, c)|$$

### 2.2 Empirical Results

| Sample Video | Reference Frame Path | Video Resolution | Image Resolution | Best Match Frame | Transform Required | MAE (out of 255) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `are you free today.mp4` | `Frames_Sentence_Level/are you free today/1/are you free today 01.jpg` | $320 \times 240$ (Landscape) | $240 \times 320$ (Portrait) | **Frame 0** | **90° Clockwise** | **1.5155** (0.59%) | **CONFIRMED EXACT PIXEL CORRESPONDENCE (Signer 1)** |
| `free (2).MP4` | `Frames_Sentence_Level/are you free today/4/free (2) 01.jpg` | $1920 \times 1080$ | $1920 \times 1080$ | **Frame 2** | Original | **1.0632** (0.41%) | **CONFIRMED EXACT PIXEL CORRESPONDENCE (Signer 4)** |
| `MVI_6209.MP4` | `Frames_Sentence_Level/are you hiding something/7/MVI_6209 01.jpg` | $1920 \times 1080$ | $1920 \times 1080$ | **Frame 2** | Original | **1.0855** (0.42%) | **CONFIRMED EXACT PIXEL CORRESPONDENCE (Signer 7)** |

### 2.3 Key Technical Insights
1. **Camera Rotation in Native Signer Videos**: The video `are you free today.mp4` was recorded with the camera mounted sideways (QVGA landscape). During author extraction, the frames were transposed 90° clockwise into upright portrait orientation. Frame 0 matches `are you free today 01.jpg` with an MAE of 1.51 / 255 (the residual difference is standard 8x8 DCT JPEG quantization noise).
2. **Decoder Startup Offset**: In both Full-HD DSLR recordings (`free (2).MP4` and `MVI_6209.MP4`), the author's extracted `01.jpg` matches **Video Frame 2**, showing that the author's extraction script dropped the initial 2 decoding container frames.
3. **Definitive Ground Truth Established (Tier A vs. Tier B)**: We have now empirically proven that `are you free today.mp4` is Signer 1, `free (2).MP4` is Signer 4, and `MVI_6209.MP4` is Signer 7. These three samples constitute the certified **Tier A: Pixel-Verified Ground Truth** subset. They empirically validate the author extraction methodology that underpins the 446 **Tier B: Unique Forensic Trace Matches** across the broader corpus. Tier A is an exemplar validation subset within the corpus, not an additive set beyond the 687 total videos.

---

## 3. Kinematic Verification of Untrimmed Video Durations

### 3.1 Kinematic Methodology
To test whether the low global hand detection ratios in `bring water for me (1).MP4` (40.5%) and `can you repeat that please (3).MP4` (46.8%) were caused by untrimmed lead-in/lead-out resting postures, we tracked the vertical position of both wrists ($P_{15}, P_{16}$) relative to the shoulders ($P_{11}, P_{12}$) across three temporal phases:
- **Beginning (Pre-signing)**: Frames 0 to $T/3$
- **Middle (Active signing)**: Frames $T/3$ to $2T/3$
- **End (Post-signing)**: Frames $2T/3$ to $T$

In MediaPipe normalized coordinates ($y=0$ top, $y=1$ bottom):
- $\Delta y = y_{\text{wrist}} - y_{\text{shoulder}} > +0.30$ indicates hands resting at sides/waist.
- $\Delta y = y_{\text{wrist}} - y_{\text{shoulder}} \le +0.10$ indicates hands raised in the active signing space in front of the chest.

### 3.2 Empirical Kinematic Results

#### Sample A: `bring water for me (1).MP4` (116 frames / 4.64s)
- **Beginning [Frames 0–37 | 0.0s – 1.48s]**:
  - Hand Detections: Left = **0/38 (0.0%)**, Right = **0/38 (0.0%)**
  - Right Wrist Avg $y = 1.003$ vs Shoulder $y = 0.521$ ($\Delta y = +0.482$)
  - **Posture**: Both hands resting down at waist/sides. Zero signing activity.
- **Middle [Frames 38–75 | 1.52s – 3.00s]**:
  - Hand Detections: Right = **29/38 (76.3%)**, Left = **3/38 (7.9%)**
  - Right Wrist Avg $y = 0.502$ vs Shoulder $y = 0.503$ ($\Delta y = -0.001$)
  - **Posture**: Right hand raised in front of chest. Dense active signing. Left hand inactive (asymmetric one-handed sign).
- **End [Frames 76–115 | 3.04s – 4.60s]**:
  - Hand Detections: Left = **0/40 (0.0%)**, Right = **18/40 (45.0% transitioning down)**
  - Right Wrist Avg $y = 0.858$ vs Shoulder $y = 0.515$ ($\Delta y = +0.342$)
  - **Posture**: Hands return to resting posture at sides.

#### Sample B: `can you repeat that please (3).MP4` (109 frames / 4.36s)
- **Beginning [Frames 0–35 | 0.0s – 1.40s]**:
  - Hand Detections: Left = **0/36 (0.0%)**, Right = **3/36 (8.3%)**
  - Right Wrist Avg $y = 0.927$ vs Shoulder $y = 0.517$ ($\Delta y = +0.411$)
  - **Posture**: Hands down at sides.
- **Middle [Frames 36–71 | 1.44s – 2.84s]**:
  - Hand Detections: Right = **35/36 (97.2%)**, Left = **1/36 (2.8%)**
  - Right Wrist Avg $y = 0.541$ vs Shoulder $y = 0.499$ ($\Delta y = +0.042$)
  - **Posture**: Right hand actively signing in upper chest space with **97.2% detection density**.
- **End [Frames 72–108 | 2.88s – 4.32s]**:
  - Hand Detections: Left = **0/37 (0.0%)**, Right transitioning down to resting posture ($y=0.773$).

### 3.3 Kinematic Conclusion
The low global detection percentages are **not** MediaPipe tracking failures. They reflect the literal reality of the video files:
1. ~60% of each clip consists of pre-signing and post-signing resting posture where hands are at the sides.
2. During the active middle segment, detection density is exceptional (**76.3% to 97.2%**).
3. Both sentences are naturally executed as one-handed dominant signs, explaining the near-zero left hand detection.

---

## 4. Review of Quality Control Architecture

The preprocessing pipeline strictly separates concerns across four distinct evaluation dimensions:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   1. Technical Extraction Validity                     │
│  - Video file openable by OpenCV                                       │
│  - Feature tensor has 0 NaN / 0 Inf (all values finite float32)        │
│  - Dimension matches topology exactly (e.g. 229D)                      │
│  - Duration within bounds (15 <= T <= 300 frames)                      │
│  * Failure Action: REJECT (File discarded / unreadable)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ PASS
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   2. Landmark Coverage & Stability                     │
│  - Upper body pose detected in >= 70% of frames                        │
│  - Normalization stable (fallback expired in <= 20% of frames)         │
│  - Hand presence:                                                      │
│      * Zero hands detected across entire video -> REJECT / HARD FAIL   │
│      * Max hand ratio < 20% -> QUARANTINE                              │
│      * Min hand < 10% and Max hand >= 20% -> WARNING (QC-W_ASYM)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ PASS / PASS_WITH_WARNING
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   3. Signer Identity Verification                      │
│  - Ground-truth / trace link to Frames_Sentence_Level verified?        │
│  - Verified: YES -> Route to natural split (train / val / test)        │
│  - Verified: NO  -> Route to 'quarantine' split                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                4. Suitability for Model Training Gate                  │
│  - MUST have passed Tiers 1 & 2                                        │
│  - MUST have signer_identity_verified == True                          │
│  - MUST have split == 'train' (Signers 1-5 only)                       │
│  - Excludes: Quarantined clips (unverified signers)                    │
│  - Excludes: Signer 6 (Validation split) and Signer 7 (Test split)    │
└────────────────────────────────────────────────────────────────────────┘
```

### Critical Policy Protections:
- **Zero Fabrication**: Missing hand and pose landmarks remain strictly `0.0`. No forward-filling or coordinate interpolation is performed.
- **One-Handed Sign Preservation**: An asymmetric sign (e.g. Right = 40.5%, Left = 2.6%) is **never** rejected. It clears Tier 2 and receives a non-fatal informational tag `QC-W_ASYM`.
- **`QC PASS` $\ne$ Training Suitability**: `QC PASS` indicates numerical validity only. A clip in the test set (`mvi_6209`) has `qc_status = PASS` but `suitability_for_training = False`.

---

## 5. Implementation & Storage Verification

All 5 prototype outputs were audited against their physical on-disk state:

| Clip ID | Source Video | Sentence Category | Signer ID | Verified | Split | Physical Path | Training Suitable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `are_you_free_today` | `are you free today.mp4` | are you free today | 1 (Native) | True | `train` | `train/are_you_free_today.npz` | **True** |
| `free_2` | `free (2).MP4` | are you free today | 4 (Volunteer) | True | `train` | `train/free_2.npz` | **True** |
| `mvi_6209` | `MVI_6209.MP4` | are you hiding something | 7 (Volunteer) | True | `test` | `test/mvi_6209.npz` | **False** (Held-out Test) |
| `bring_water_for_me_1` | `bring water for me (1).MP4` | bring water for me | unverified | False | `quarantine` | `quarantine/bring_water_for_me_1.npz` | **False** (Quarantined) |
| `can_you_repeat_that_please_3` | `can you repeat that please (3).MP4` | can you repeat that please | unverified | False | `quarantine` | `quarantine/can_you_repeat_that_please_3.npz` | **False** (Quarantined) |

- **No Overwrite Hazard**: The pipeline uses explicit `--overwrite` semantics; re-running without the flag skips existing files safely.
- **Master Manifest Invariant**: Every row in `data/processed/isl_csltr_tensors_v1/manifest.csv` reflects the exact on-disk `.npz` path, shapes, and verification status.

---

## 6. Test Suite Results

```bash
pytest backend/tests/test_preprocessing.py -v
pytest backend/tests -v
git diff --check
```

- **Preprocessing Unit Tests (8/8 Passed in 0.31s)**:
  - `test_topology_dimensions_all_candidates`: PASSED
  - `test_normalization_valid_shoulders`: PASSED
  - `test_normalization_hips_fallback`: PASSED
  - `test_normalization_temporal_fallback_within_and_beyond_gap`: PASSED
  - `test_zero_fabrication_on_missing_hands`: PASSED
  - `test_wrist_relative_depth_preserved`: PASSED
  - `test_unverified_signer_routing_to_quarantine`: PASSED
  - `test_training_suitability_requires_verified_signer_and_train_split`: PASSED
- **Full Backend Suite**: **12/12 PASSED in 0.89s**
- **Git Whitespace & Syntax**: Clean (0 diff warnings)

---

## 7. Final Verdict

### Verdict: **APPROVED WITH RESTRICTIONS**

The preprocessing prototype is approved to expand to the full 687-video dataset under the following four mandatory restrictions:

1. **Automated Stem-Match Gate for Signer Verification**:
   During full corpus processing, a video can only be assigned to a signer split (`train`, `val`, `test`) if its stem unambiguously matches the frame filename stem in `Frames_Sentence_Level/<sentence>/<signer_id>/`. All ambiguous or generic numbered takes (e.g. `(1).MP4` without stem matches) **must be quarantined immediately** with `signer_identity_verified=False`.
2. **Strict Test Set Isolation**:
   Signer 7 recordings must remain in the `test/` directory and must **never** be used for feature topology selection, hyperparameter tuning, or early stopping during Step 3.
3. **Temporal Attention in Modeling (Step 3.1)**:
   Because video takes contain ~1.4s of untrimmed leading and trailing rest frames, sequence classification architectures (BiLSTM, GRU, Transformer) must use the modality presence mask $\mathbf{m}_t$ and temporal pooling/attention to focus on the active signing window.
4. **Zero Untracked Data Leakage**:
   The entire `data/` directory must remain gitignored. No video, frame, or `.npz` files may be committed to Git.

---
*Audit certified and approved for SignBridge AI Step 2.4 expansion.*
