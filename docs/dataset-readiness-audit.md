# SignBridge AI: Step 2.5A Dataset Readiness Audit Report

**Document ID:** `docs/dataset-readiness-audit.md`  
**Milestone:** Step 2.5A — Dataset Readiness Audit  
**Author:** SignBridge AI ML Engineering  
**Date:** November 2026  
**Status:** Audit Completed  
**Primary References:**  
- Full-Corpus Report: [`docs/full-corpus-preprocessing-report.md`](full-corpus-preprocessing-report.md)
- Full-Corpus Dry Run: [`docs/full-corpus-preprocessing-dry-run.md`](full-corpus-preprocessing-dry-run.md)
- Preprocessing Design: [`docs/preprocessing-design.md`](preprocessing-design.md)
- Master Manifest: [`data/processed/isl_csltr_tensors_v2/manifests/master_manifest.csv`](../data/processed/isl_csltr_tensors_v2/manifests/master_manifest.csv)
- Failed Videos Manifest: [`data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv`](../data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv)
- Quarantine Manifest: [`data/processed/isl_csltr_tensors_v2/manifests/quarantine_manifest.csv`](../data/processed/isl_csltr_tensors_v2/manifests/quarantine_manifest.csv)
- Master Catalog: [`data/raw/isl-csltr/metadata/corpus_dry_run_results.json`](../data/raw/isl-csltr/metadata/corpus_dry_run_results.json)

---

## 1. Executive Summary

This audit evaluates whether the **ISL-CSLTR (ISL-CSLRT)** dataset is genuinely ready for baseline machine-learning training. While the feature extraction pipeline (Candidate B 3D, 229D), orientation correction (gated 90° CW rotation for native signers), coordinate normalization (torso-centered, shoulder-scaled), and quality control filters are verified and functional, **the physical dataset currently available on local storage is completely insufficient for training a classification model**.

### Key Audit Findings:
1. **Physical Video Availability:** Exactly **6 unique MP4 video files** exist in local raw storage (`data/raw/isl-csltr/sample_videos/`). The remaining **449 eligible benchmark videos** reside in the published remote Mendeley archive (`ISL_CSLRT_Corpus.zip`, 8.29 GiB) and have not yet been downloaded to local disk.
2. **Processed Tensor Inventory:** Exactly **4 benchmark-eligible videos** have processed `.npz` feature tensors in `data/processed/isl_csltr_tensors_v2/`.
3. **Training Suitability Gate:** Of the 4 processed benchmark videos:
   - **2 tensors** pass all QC checks and are in the training split (`suitability_for_training = True`).
   - **1 tensor** passes QC but belongs to the held-out test split (`test/mvi_6209.npz`, Signer 7; `suitability_for_training = False`).
   - **1 tensor** failed the minimum-duration quality gate (`train/are_you_free_today_2.npz`, Signer 2; 25 frames $< 30$ minimum duration bound; `suitability_for_training = False`).
4. **Class Coverage Deficit:** The 2 usable training tensors belong to **the exact same single sentence class** (`'are you free today'`). There are **0 validation tensors** and **0 training tensors** for 70 of the 71 benchmark training classes.
5. **Readiness Verdict:** **FAIL (NOT READY FOR BASELINE TRAINING)**. Training a sentence-level classifier on 2 samples from 1 class is mathematically impossible. The data extraction must be expanded before training can commence.

---

## 2. Actual Dataset Availability

The published ISL-CSLTR dataset consists of 687 MP4 videos across 101 sentence categories. The full-corpus accounting is partitioned into 453 benchmark-eligible videos and 234 quarantined videos ($453 + 234 = 687$).

### 2.1 Physical On-Disk Video Inventory

An exhaustive filesystem scan of `data/raw/isl-csltr/` identifies exactly **6 unique video files** (all located in `data/raw/isl-csltr/sample_videos/`):

| Filename | File Size | Catalog Category | Signer ID | Partition | Local Availability Status |
| :--- | :---: | :--- | :---: | :--- | :--- |
| `are you free today.mp4` | 634,833 B | `are you free today` | Signer 1 (Native) | Benchmark (Train) | **Present locally** |
| `are you free today (2).mp4` | 57,618 B | `are you free today` | Signer 2 (Native) | Benchmark (Train) | **Present locally** |
| `free (2).MP4` | 11,245,586 B | `are you free today` | Signer 4 (Volunteer) | Benchmark (Train) | **Present locally** |
| `MVI_6209.MP4` | 13,930,893 B | `are you hiding something` | Signer 7 (Volunteer) | Benchmark (Test) | **Present locally** |
| `bring water for me (1).MP4` | 16,948,475 B | `bring water for me` | Unverified | Quarantine | **Present locally (Diagnostic)** |
| `can you repeat that please (3).MP4` | 15,819,619 B | `can you repeat that please` | Unverified | Quarantine | **Present locally (Diagnostic)** |

### 2.2 Pending Video Accounting
- **Total Benchmark Eligible Videos in Catalog:** **453**
- **Eligible Videos Present on Disk:** **4**
- **Missing Eligible Videos:** **449** (99.12% of the benchmark dataset)
- **Status of Missing Videos:** Explicitly accounted for in [`data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv`](../data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv) under `failure_stage: "FILE_DISCOVERY"` with reason `"Source video file not present in local filesystem (pending full archive download)"`.
- **Can the Original Archive Provide the Pending Videos?** Yes. All 449 pending videos are indexed with byte offsets and sizes in `data/raw/isl-csltr/metadata/csltr_file_list.json` and reside inside `ISL_CSLRT_Corpus.zip` on Mendeley S3. They simply have not been downloaded or extracted to local disk yet.

---

## 3. Processed Tensor Inventory (`data/processed/isl_csltr_tensors_v2/`)

The full-corpus v2 preprocessing pipeline produced 6 compressed `.npz` feature archives:

```text
data/processed/isl_csltr_tensors_v2/
├── train/
│   ├── are_you_free_today.npz       (Signer 1, 50 frames, 229D, QC: PASS, TrainSuit: True)
│   └── are_you_free_today_2.npz     (Signer 2, 25 frames, 229D, QC: FAIL [T < 30], TrainSuit: False)
├── validation/                      (0 tensors — EMPTY)
├── test/
│   └── mvi_6209.npz                 (Signer 7, 95 frames, 229D, QC: PASS, TrainSuit: False [Test Holdout])
└── quarantine/
    ├── bring_water_for_me_1.npz     (Unverified, 116 frames, 229D, QC: QUARANTINE_UNVERIFIED, TrainSuit: False)
    └── can_you_repeat_that_please_3.npz (Unverified, 109 frames, 229D, QC: QUARANTINE_UNVERIFIED, TrainSuit: False)
```

### 3.1 Tensor Breakdown by Suitability
- **Total Processed Tensors:** 6
- **Benchmark Tensors:** 4
- **Quarantine Diagnostic Tensors:** 2 (`suitability_for_training = False`)
- **Tensors Passing Preprocessing QC:** 3 (`are_you_free_today.npz`, `free_2.npz`, `mvi_6209.npz`)
- **Tensors Passing Minimum Duration Gate:** 3 (1 failed: `are_you_free_today_2.npz` has 25 frames $< 30$)
- **Tensors Usable for Training (`suitability_for_training == True`):** **EXACTLY 2**
  1. `train/are_you_free_today.npz`
  2. `train/free_2.npz`

---

## 4. Label and Class Distribution

### 4.1 Label Schema & Representation
- **Sentence Category:** Stored as normalized lowercase English text in clip metadata (e.g. `'are you free today'`, `'are you hiding something'`).
- **Sign Glosses:** Stored as uppercase gloss tokens in clip metadata, loaded from `data/raw/isl-csltr/metadata/ISL Corpus sign glosses.csv` (101 mapped pairs, e.g. `'YOU FREE TODAY'`).
- **Duplicate / Conflicting Labels:**
  - Duplicate sentences in gloss mapping: **0**.
  - Duplicate glosses: Exactly **1 collision** (`'WHAT YOU DO'` is mapped to both `'what are you doing'` and `'what do you do'`). For sentence-level classification, the target label must be the unique sentence text, not the sign gloss, to prevent label ambiguity.

### 4.2 Benchmark Vocabulary Scope
Although the raw dataset recorded 101 sentence categories:
- **Verified Benchmark Classes:** **71 classes** are supported by unambiguous single-signer recordings.
- **Quarantined Classes:** **30 classes** have all their takes quarantined (generic `(N).jpg` numbering or missing extracted frame directories).
- **Usable Benchmark Classes:** Only **71 classes** can genuinely be used in the benchmark. The claim of a "101-class sentence translation task" must be scoped down to 71 classes for verified continuous sign language recognition.

### 4.3 Class Distribution across Splits (Full Design vs. Physical Availability)

| Split | Full Benchmark Design Classes | Full Benchmark Design Videos | Physically Available Local Tensors | Usable Training Tensors |
| :--- | :---: | :---: | :---: | :---: |
| **Train (Signers 1–5)** | 71 classes | 319 videos | 3 videos (2 files) | **2 videos (1 class: `'are you free today'`)** |
| **Validation (Signer 6)** | 66 classes | 66 videos | **0 videos (0 classes)** | **0 videos** |
| **Test (Signer 7)** | 68 classes | 68 videos | 1 video (1 class: `'are you hiding something'`) | 0 videos (held-out) |
| **Quarantine** | 49 categories | 234 videos | 2 videos (2 categories) | 0 videos (quarantined) |

### 4.4 Usable Training Samples per Class
- **Classes with $\ge 1$ usable training tensor:** **1 class** (`'are you free today'`: 2 samples).
- **Classes with 0 usable training tensors:** **70 classes** (98.59% of training vocabulary).
- **Sample Distribution per Class in Full Catalog Design:**
  - 5 samples per class: 53 classes
  - 4 samples per class: 7 classes
  - 3 samples per class: 6 classes
  - 2 samples per class: 3 classes
  - 1 sample per class: 2 classes

---

## 5. Tensor Structure & Integrity

Physical inspection of all 6 generated `.npz` archives confirmed complete technical integrity:

| Tensor File | Sequence Length ($T$) | Feature Shape | Presence Mask Shape | Dtype | Finite (0 NaN/Inf) | Pose Ratio | Hands Ratio (L / R) | Norm Valid Ratio | Applied Rotation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train/are_you_free_today_2.npz` | 25 | $(25, 229)$ | $(25, 4)$ | `float32` | **True** | 100.0% | 100.0% / 0.0% | 100.0% | 90° CW |
| `train/are_you_free_today.npz` | 50 | $(50, 229)$ | $(50, 4)$ | `float32` | **True** | 100.0% | 84.0% / 54.0% | 100.0% | 90° CW |
| `train/free_2.npz` | 78 | $(78, 229)$ | $(78, 4)$ | `float32` | **True** | 100.0% | 41.0% / 55.1% | 100.0% | 0° (Upright) |
| `test/mvi_6209.npz` | 95 | $(95, 229)$ | $(95, 4)$ | `float32` | **True** | 100.0% | 75.8% / 44.2% | 100.0% | 0° (Upright) |
| `quarantine/bring_water_for_me_1.npz` | 116 | $(116, 229)$ | $(116, 4)$ | `float32` | **True** | 100.0% | 2.6% / 40.5% | 100.0% | 0° (Upright) |
| `quarantine/can_you_repeat_that_please_3.npz` | 109 | $(109, 229)$ | $(109, 4)$ | `float32` | **True** | 100.0% | 0.9% / 46.8% | 100.0% | 0° (Upright) |

### Key Structural Findings:
1. **Consistency:** All tensors strictly use the 229-dimensional Candidate B schema (99 pose + 63 left hand + 63 right hand + 4 indicators).
2. **Numeric Cleanliness:** Zero NaN, null, or infinite values in any feature or mask tensor.
3. **Decoupled Presence Indicators:** Strictly track whether articulators were observed ($m_{\text{left}}, m_{\text{right}}, m_{\text{pose}}, m_{\text{norm}}$).
4. **Orientation Invariant:** Both native signer videos were rotated 90° CW ($320 \times 240 \to 240 \times 320$), yielding horizontal shoulders ($+0.8^\circ$) matching volunteer coordinate geometry.
5. **Quality Gate Sensitivity:** The pipeline correctly caught and failed `are_you_free_today_2.npz` because its duration of 25 frames falls below the 30-frame threshold ($1.2\text{s}$ at 25 FPS).

---

## 6. Split Integrity

- **Signer 7 (Held-Out Test Set):** Strictly isolated. Exactly 1 Signer 7 tensor exists locally (`test/mvi_6209.npz`), correctly assigned to `test/` with `suitability_for_training = False`. Zero Signer 7 tensors exist in `train/` or `validation/`.
- **Validation Split (`validation/`):** **Empty (0 tensors)**. Signer 6 has 66 videos in the catalog design, but none have been extracted locally.
- **Quarantine Isolation:** Both processed quarantine tensors are located in `quarantine/` with `suitability_for_training = False`. None can enter training batches.

---

## 7. Assessment of Training Readiness

### Question 1: Are the current 2 training tensors sufficient to train a baseline model?
**No. Absolutely not.**
- Supervised classification requires at least two distinct classes to form a decision boundary and compute cross-entropy loss:
  $$\mathcal{L} = -\sum_{c=1}^C y_c \log \hat{y}_c$$
  With $C = 1$ class (`'are you free today'`), cross-entropy loss is identically zero ($\log 1 = 0$), gradients are zero, and no learning can occur.
- Even for binary classification, at least 2 distinct classes with multiple samples each are mathematically required.
- Attempting to train any machine learning model on 2 samples would be an exercise in trivial memorization, yielding zero predictive capability and zero scientific validity.

### Question 2: Can the dataset support sentence-level classification?
**Yes, in design; but not yet in physical data availability.**
- Once extracted, the 319 training videos cover 71 classes with an average of 4.5 samples per class (53 classes have 5 distinct signers).
- This is sufficient for few-shot or low-resource sequence classification (e.g. BiLSTM / GRU / Spatial-Temporal GCN).
- However, physical video extraction is a non-negotiable prerequisite.

### Question 3: Is the current train/validation/test split suitable?
**Yes.**
- The signer-independent partition (Signers 1–5 in Train, Signer 6 in Val, Signer 7 in Test) is methodologically sound and prevents person-specific memorization.
- Category overlap is 100%: all 66 validation classes and all 68 test classes are represented in the 71 training classes (0 zero-shot leakage anomalies).

### Question 4: Can the 101 sentence categories be used?
**No.**
- Only **71 categories** have verified benchmark recordings.
- The remaining **30 categories** are completely quarantined due to generic numbering or missing extracted frame directories in the archive.
- The classification head in Step 3 must be sized to **71 classes**, not 101 classes.

### Question 5: What minimum additional data preparation is necessary before training?
1. Extract or download the remaining **449 eligible benchmark videos** from the Mendeley archive (`Videos_Sentence_Level/`).
2. Run `preprocess_isl_csltr.py --mode full_corpus` to generate 229D tensors for all 453 eligible videos.
3. Validate that `data/processed/isl_csltr_tensors_v2/` contains $\ge 300$ usable training tensors across all 71 classes and $\ge 60$ validation tensors across Signer 6.

---

## 8. Blockers & Recommended Next Technical Step

### Primary Blocker
**Physical Video Data Starvation:** 449 of 453 benchmark source videos (99.12%) exist only as entries in `failed_videos.csv` because their MP4 files have not been unpacked locally from the 8.29 GiB Mendeley archive.

### Evaluation of Options
- **Option A (Extract remaining eligible videos):** Required for real training, but involves downloading or unpacking large video archives.
- **Option B (Fix dataset access or preprocessing):** The preprocessing pipeline itself has passed all tests and has no functional bugs.
- **Option C (Prepare training pipeline using a small smoke-test subset):** Highly recommended as the immediate next step. We can build the PyTorch `Dataset`, `DataLoader`, variable-length collate function with sequence padding, attention masking, and baseline BiLSTM model skeleton, testing it on the available local tensors to certify that the training pipeline code runs cleanly from end to end without crashing.
- **Option D (Proceed directly to baseline training):** Strictly rejected. Training cannot proceed with 2 samples from 1 class.

### Recommended Course of Action
**Adopt Option C immediately, followed by Option A:**
1. **Immediate Next Step (Step 2.5B / Step 3.1 Prep — Option C):**
   Implement the PyTorch data loader and baseline model skeleton (`torch.utils.data.Dataset`, padding collator, BiLSTM architecture). Execute a small dry-run / smoke-test forward pass using the available local tensors to verify tensor batching, mask collation, and gradient computation. Clearly document this as a pipeline verification smoke-test, claiming zero model accuracy.
2. **Bulk Extraction (Option A):**
   Obtain the remaining 449 eligible videos, run bulk preprocessing, and populate all 71 training classes and 66 validation classes before training the real baseline model.

---

## 9. Final Verdict

### **VERDICT: FAIL (NOT READY FOR BASELINE TRAINING)**

*Reasoning:* While the preprocessing v2 pipeline and feature tensor schema (229D) have earned a **PASS** on technical and numerical integrity, the dataset is **NOT READY FOR BASELINE TRAINING** because 449 of 453 benchmark videos (and 70 of 71 training classes) are physically absent from local disk. Proceed to data pipeline architecture and smoke-testing (Option C) before attempting real model training.

