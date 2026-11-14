# SignBridge AI: Full-Corpus Preprocessing Dry Run & Technical Audit (Step 2.4A)

**Document ID:** `docs/full-corpus-preprocessing-dry-run.md`
**Milestone:** Step 2.4A — Full-Corpus Preprocessing Dry Run & Technical Audit
**Author:** SignBridge AI ML Engineering
**Date:** November 2026
**Status:** Completed & Empirically Audited
**Master Dry-Run Catalog:** `data/raw/isl-csltr/metadata/corpus_dry_run_results.json`

---

## 1. Executive Summary

This document presents the full-corpus technical audit and signer-mapping dry run across the complete **ISL-CSLTR (ISL-CSLRT)** dataset. All **687 video files** across all **101 sentence categories** were scanned, analyzed against the **18,863 pre-extracted frames** in `Frames_Sentence_Level/`, and audited for camera orientation, split leakage, and quality control invariants.

### Key Audit Highlights:
1. **Camera Orientation Discrepancy Discovered**:
   Native signer recordings (Signers 1 and 2, 133 videos) were recorded on a portable camera mounted sideways ($320 \times 240$ QVGA landscape). Because the initial prototype script passed frames directly to MediaPipe without rotation, and torso normalization performs only 2D translation and scaling, **the coordinate systems of native signers were rotated by 90° relative to volunteer signers** (native shoulders are at $-90.0^\circ$ vertical vs volunteer shoulders at $-3.6^\circ$ horizontal). A pre-extraction 90° clockwise rotation restores upright shoulders ($+0.8^\circ$) and aligns the coordinate frame with volunteers.
2. **Evidentiary Hierarchy of the 453 Mappings**:
   - 453 videos have unique, 1-to-1 bijective stem correspondence to signer folders in `Frames_Sentence_Level/` (including 7 verified `MVI_` prefixed videos following regex resolution of underscore camera IDs).
   - Direct pixel-level frame comparison confirmed exact match (MAE $\le 1.52 / 255$) across 4 representative native and volunteer samples.
   - 234 videos lacking unambiguous correspondence are strictly quarantined.
3. **Split & Category Coverage Verified**:
   - Train (Signers 1–5): **319 videos** (71 classes)
   - Validation (Signer 6): **66 videos** (66 classes)
   - Final Test Benchmark (Signer 7): **68 videos** (68 classes)
   - Quarantined: **234 videos** (37 categories)
   - **Zero-Shot Leakage**: Exactly **0 classes** in Val or Test are missing from Train.
   - **Signer 7 Isolation**: Signer 7 recordings appear exclusively in `test/` and cannot enter training under any filename.
4. **Temporal Modeling Refinement**:
   Modality presence masks $\mathbf{m}_t$ are preserved in all tensors. Rather than making temporal attention mandatory, it is formulated as a candidate model design to compare against simpler baselines during Step 3.

---

## 2. Critical Camera Orientation Investigation

### 2.1 The Investigation
During Step 2.3B pixel verification, matching `are you free today.mp4` to the author's extracted frame `are you free today 01.jpg` required a **90° clockwise rotation**. We systematically audited the pipeline to determine the impact on landmark coordinates.

### 2.2 Detailed Technical Findings

#### Question 1: Does the script detect and correct camera orientation before extraction?
**No.** In [`scripts/preprocess_isl_csltr.py`](../scripts/preprocess_isl_csltr.py) (lines 426–432), frames are fetched directly via `cap.read()`, converted to RGB, and passed directly to `pose_lmk.detect_for_video()` and `hand_lmk.detect_for_video()`. There is no orientation detection, metadata EXIF/MP4 tag inspection, or rotation transformation.

#### Question 2: Does it use OpenCV rotation, orientation metadata, or another transformation?
**Neither.** No rotation or transposition is currently applied anywhere in the extraction loop.

#### Question 3: Are resulting pose and hand coordinates in a consistent coordinate system?
**No.** MediaPipe outputs coordinates normalized to the image plane ($[0.0, 1.0] \times [0.0, 1.0]$). In an unrotated native signer video, the person is sideways:
- Up/down movements of the hands map to the **horizontal ($X$) axis**.
- Left/right movements map to the **vertical ($Y$) axis**.
- In volunteer recordings ($1920 \times 1080$), up/down movements map to the **vertical ($Y$) axis**.

#### Question 4: Does torso normalization correct rotation?
**No.** Torso normalization in `normalize_coordinates` performs strictly 2D translation and isotropic scaling:
$$\tilde{x} = \frac{x - C_x}{S_{\text{torso}}}, \quad \tilde{y} = \frac{y - C_y}{S_{\text{torso}}}$$
Translation by mid-shoulder $(C_x, C_y)$ and division by shoulder distance $S_{\text{torso}}$ centers and rescales the skeleton, but **applies zero rotational alignment**. A skeleton that was tilted by $90^\circ$ remains tilted by $90^\circ$ after torso normalization.

#### Question 5: Are all videos affected, or only particular recordings?
**Only native signer recordings (Signers 1 and 2) are affected.**
- **Native Signers (Signers 1 and 2)**: Recorded on 28-12-2019 at Navajeevan Residential School for the Deaf on a portable camera mounted sideways ($320 \times 240$ QVGA landscape, average file size ~275 KB – 979 KB). All 133 verified native videos are affected.
- **Volunteers (Signers 3–7)**: Recorded at SASTRA Deemed University on a Canon EOS DSLR at $1920 \times 1080$ Full HD (average size ~15.5 MB). All volunteer videos were recorded with the camera upright in standard landscape orientation, where the signer is upright in the frame.

---

### 2.3 Empirical Kinematic Orientation Test

A reproducible landmark test was executed comparing a representative volunteer video (`free (2).MP4`, Signer 4) and a native video (`are you free today.mp4`, Signer 1) in unrotated and 90° rotated states:

```text
=== Volunteer (free (2).MP4, Signer 4) - Original Orientation ===
  Image Shape (HxW)       : 1080 x 1920
  Nose Landmark           : (0.536, 0.297)
  L-Shoulder / R-Shoulder : (0.639, 0.533) / (0.422, 0.547)
  Shoulder Vector Angle   : -3.6° from horizontal (HORIZONTAL SHOULDERS)
  Spine Vector Angle      : +1.3° from vertical (UPRIGHT HEAD)

=== Native Signer 1 (are you free today.mp4) - AS CURRENTLY PROCESSED (Unrotated) ===
  Image Shape (HxW)       : 240 x 320
  Nose Landmark           : (0.273, 0.534)
  L-Shoulder / R-Shoulder : (0.409, 0.280) / (0.409, 0.663)
  Shoulder Vector Angle   : -90.0° from horizontal (COMPLETELY VERTICAL SHOULDERS!)
  Spine Vector Angle      : -114.6° from vertical (HEAD POINTING SIDEWAYS!)

=== Native Signer 1 (are you free today.mp4) - ROTATED 90° CW (Upright) ===
  Image Shape (HxW)       : 320 x 240
  Nose Landmark           : (0.473, 0.276)
  L-Shoulder / R-Shoulder : (0.716, 0.408) / (0.327, 0.402)
  Shoulder Vector Angle   : +0.8° from horizontal (HORIZONTAL SHOULDERS)
  Spine Vector Angle      : -20.2° from vertical (UPRIGHT HEAD)
```

### 2.4 Proposed Safest Pipeline Correction
In accordance with instructions, the pipeline was **not silently modified**. We submit the following safest architectural correction for Step 2.4B:

> **Recommended Correction (Pre-Extraction Upright Rotation)**:
> In `preprocess_video()`, inspect video dimensions. If width > height and height $\le 240$ (or if `signer_id in ['1', '2']`), apply:
> ```python
> if width > height and height <= 240:
>     frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
> ```
> *Why this is the safest path*:
> 1. It operates *before* MediaPipe, allowing MediaPipe’s internal convolutional anchors to operate on upright persons (the exact domain they were trained on).
> 2. It replicates the exact author workflow used when creating `Frames_Sentence_Level/are you free today/1/` (where images are stored upright at $240 \times 320$).
> 3. It permanently guarantees that all 453 verified feature tensors share an identical, upright spatial coordinate system.

---

## 3. Audit of the 453 Signer Mappings & Evidentiary Hierarchy

### 3.1 Evidentiary Hierarchy & Corpus Accounting
To ensure absolute scientific rigor, we distinguish five tiers of signer assignment confidence.

> [!IMPORTANT]
> **Corpus Accounting Invariant:** The entire ISL-CSLTR corpus consists of **exactly 687 video files**. The corpus partitions exhaustively into **453 verified videos** (assigned to benchmark splits) and **234 quarantined videos** ($453 + 234 = 687$).
> **Tier A is NOT an additive 6th split.** Tier A represents the 4 exemplar videos directly audited via pixel-level frame decompilation to certify the ground-truth reliability of the author's extraction process (`are you free today.mp4`, `free (2).MP4`, `are you free today (2).mp4`, and `MVI_6209.MP4`). Following the regex resolution of underscore camera IDs (preserving `_6209` in `MVI_6209`), all 4 Tier A videos are fully accounted for within the 453 verified benchmark videos (3 in Train, 1 in Test). No video is double-counted.

| Evidence Tier | Definition | Video Count | Example | Split Assignment |
| :--- | :--- | :---: | :--- | :--- |
| **Tier A: Pixel-Verified Ground Truth** | Decoded video frames empirically matched against archive JPEGs with $\text{MAE} \le 1.52 / 255$ (exemplar ground-truth subset). | 4 tested | `are you free today.mp4` (MAE=1.51), `free (2).MP4` (MAE=1.06), `MVI_6209.MP4` (MAE=1.08), `are you free today (2).mp4` (MAE=0.42%) | **Verified Ground Truth** (3 in Train; 1 in Test) |
| **Tier B: Unique Forensic Trace Match** | Video stem unambiguously matches the author's frame naming stem in exactly one signer directory, with 1-to-1 bijection across all signers in the category. | **453** | `free (4).MP4` $\leftrightarrow$ `Frames/are you free today/6/free (4) 01.jpg` | **Benchmark Splits** (Train: 319, Val: 66, Test: 68) |
| **Tier C: Ambiguous Multi-Folder Stem** | Video stem matches frame files in multiple signer directories simultaneously. | **15** | `are you hiding something.mp4` (matches folders 1, 3, 4, 5, 6) | **Quarantined** (`quarantine/`) |
| **Tier D: Generic Frame Numbering** | All signer folders use identical generic frame names `(1).jpg`..`(N).jpg` without video stems. | **204** | `bring water for me (1).MP4`, `can you repeat that please (3).MP4` | **Quarantined** (`quarantine/`) |
| **Tier E: Missing Frame Directories** | Sentence categories with videos in `Videos_Sentence_Level/` but omitted from `Frames_Sentence_Level/`. | **15** | `No need to worry dont worry`, `he would be coming today`, `i dont agree`, `now onwards...` | **Quarantined** (`quarantine/`) |
| **Total Corpus** | Exhaustive sum of all video files in archive: $\text{Verified (453)} + \text{Quarantined (234)}$. | **687** | All 101 sentence categories | **Train: 319 \| Val: 66 \| Test: 68 \| Quarantine: 234** |

### 3.2 False-Positive Vulnerability Analysis
Could the Tier B matching method produce false positives?
1. **1-to-1 Bijection Check**:
   Across all 453 verified videos, **exactly 0 signer directories receive multiple videos**. Every single assignment is a unique 1-to-1 bijection.
2. **Author Extraction Trace**:
   When the dataset authors extracted frames from videos, their extraction script used the video stem as the output JPEG prefix (e.g. `free (4).MP4` became `free (4) 01.jpg`). Because the folder structure explicitly categorizes signers (`/1/` through `/7/`), the presence of that unique stem inside folder `/6/` is a direct forensic artifact of the author's workflow.
3. **Conservative Quarantine Boundary**:
   Whenever the author's extraction script did *not* preserve the unique video stem (Categories C, D, and E), the dry run strictly assigned the videos to `quarantine/`. No ambiguous take was allowed to enter training, validation, or test sets.

---

## 4. Recalculation of Split & Category Coverage

From the master dry-run results (`data/raw/isl-csltr/metadata/corpus_dry_run_results.json`), all split totals and category overlaps were recalculated:

### 4.1 Split & Signer Distribution

```text
ISL-CSLTR Corpus (687 Total Videos)
├── 1. Benchmark Training Split (Signers 1–5): 319 Videos (46.43%)
│      - Signer 1 (Native, Navajeevan)       :  62 videos
│      - Signer 2 (Native, Navajeevan)       :  71 videos
│      - Signer 3 (Volunteer 1, SASTRA)      :  62 videos
│      - Signer 4 (Volunteer 2, SASTRA)      :  58 videos
│      - Signer 5 (Volunteer 3, SASTRA)      :  66 videos
│
├── 2. Benchmark Validation Split (Signer 6) :  66 Videos (9.61%)
│      - Signer 6 (Volunteer 4, SASTRA)      :  66 videos (66 unique classes)
│
├── 3. Final Held-Out Test Benchmark (Signer 7): 68 Videos (9.90%)
│      - Signer 7 (Volunteer 5, SASTRA)      :  68 videos (68 unique classes)
│
└── 4. Quarantined Corpus                     : 234 Videos (34.06%)
       - Generic Frame Numbering (Tier D)    : 204 videos (20 categories)
       - Ambiguous Multi-Folder Match (Tier C): 15 videos (12 categories)
       - Missing Frame Directories (Tier E)  : 15 videos (4 categories)
```

### 4.2 Category Overlap & Leakage Audit

| Metric | Category Count | Percentage of Split | Operational Status |
| :--- | :---: | :---: | :--- |
| **Total Categories in Entire Corpus** | **101** | 100.0% | Recorded vocabulary |
| **Categories with Verified Videos** | **71** | 70.30% | Supported by verified signers |
| **Unique Categories in Train (Signers 1–5)** | **71** | 100.0% | Complete verified training vocabulary |
| **Unique Categories in Val (Signer 6)** | **66** | 100.0% | Evaluated validation classes |
| **Unique Categories in Test (Signer 7)** | **68** | 100.0% | Evaluated test classes |
| **Categories Shared: Train $\cap$ Val** | **66** | **100.0% of Val** | **0 classes missing from Train** |
| **Categories Shared: Train $\cap$ Test** | **68** | **100.0% of Test** | **0 classes missing from Train** |
| **Categories Shared: Train $\cap$ Val $\cap$ Test** | **65** | **95.59% of Test** | 3 classes have Train & Test takes |
| **Val Classes Absent from Train (Zero-Shot)** | **0** | **0.0%** | **No zero-shot evaluation anomaly** |
| **Test Classes Absent from Train (Zero-Shot)** | **0** | **0.0%** | **No zero-shot evaluation anomaly** |

### 4.3 Proof of Signer 7 Test Benchmark Isolation
1. Across the 319 videos in `train/`, the candidate signer IDs are strictly restricted to `{'1', '2', '3', '4', '5'}`.
2. Across the 66 videos in `val/`, the candidate signer ID is strictly `'6'`.
3. Across the 68 videos in `test/`, the candidate signer ID is strictly `'7'`.
4. In `quarantine/`, all 234 videos have `suitability_for_training = False`.
5. **Conclusion**: Signer 7 cannot enter the training or validation sets through another filename.

---

## 5. Review of Temporal Modeling Assumptions

In accordance with instructions, we clarify the operational assumption regarding untrimmed video durations:
1. **Presence Mask Availability**:
   The 4-dimensional presence indicator $\mathbf{m}_t = [m_{\text{left}}, m_{\text{right}}, m_{\text{pose}}, m_{\text{norm}}]$ is permanently packaged in every `.npz` tensor archive under the `presence_mask` array.
2. **Decoupling Attention from Preprocessing**:
   Temporal attention over $\mathbf{m}_t$ is **not a mandatory preprocessing constraint**. Preprocessing merely extracts and records the uncorrupted temporal signal.
3. **Experimental Formulation for Step 3**:
   In Step 3 (Model Architecture Exploration), temporal attention will be evaluated as a **candidate model design hypothesis**:
   - *Baseline Model*: Standard BiLSTM / GRU with uniform average pooling over non-padded frames.
   - *Candidate Model*: BiLSTM / GRU / Transformer with explicit temporal attention gated by $m_{\text{left}}$ and $m_{\text{right}}$.
   Both models will be compared empirically on Signer 6 validation loss before choosing the production architecture.

---

## 6. Test Suite Execution & Cleanliness

All automated unit tests and code formatting checks were executed cleanly:

```bash
pytest backend/tests/test_preprocessing.py -v
pytest backend/tests -v
git diff --check
```

- **Preprocessing Unit Tests (8/8 Passed in 0.31s)**:
  - Topology dimensionality (154D, 229D, 166D, 196D): PASSED
  - Torso-centering and shoulder scaling: PASSED
  - Hip fallback hierarchy: PASSED
  - Temporal fallback expiration (5-frame bound): PASSED
  - Zero-fabrication of missing landmarks: PASSED
  - Wrist-relative depth preservation: PASSED
  - Automated quarantine routing for unverified signers: PASSED
  - Training suitability gate enforcement: PASSED
- **Full Backend Suite**: **12/12 PASSED in 0.82s**
- **Git Whitespace & Formatting**: Clean (0 diff warnings)
- **Git Status**: `data/` completely untracked; zero media or tensor files staged.

---

## 7. Final Technical Audit Verdict

### **VERDICT: READY WITH RESTRICTIONS**

The SignBridge AI dataset is technically and architecturally ready for full-corpus preprocessing tensor generation (Step 2.4B), subject to the following mandatory operational restrictions:

1. **Camera Orientation Correction Approval**:
   Before running bulk feature extraction, the pre-extraction 90° clockwise rotation for native signer recordings ($320 \times 240$ QVGA, Signers 1 and 2) must be enabled in `preprocess_video()`. This aligns native coordinates with the upright coordinate frame of volunteer signers.
2. **Strict Quarantine Enforcement**:
   Only the 453 verified videos may be routed to benchmark splits (`train/`: 319, `val/`: 66, `test/`: 68). The 234 unverified, generic, or ambiguous takes must remain in `quarantine/` with `suitability_for_training = False`.
3. **Signer 7 Test Benchmark Protection**:
   The 68 clips in `test/` must remain completely untouched until final model testing in Step 3.5. Topology selection and hyperparameter tuning must rely strictly on Signer 6 validation loss.
4. **Git Repository Hygiene**:
   `data/` remains completely untracked in `.gitignore`. No raw videos, frame JPEGs, or `.npz` archives will be staged or committed.

---
*Technical Audit certified for SignBridge AI Step 2.4A.*
