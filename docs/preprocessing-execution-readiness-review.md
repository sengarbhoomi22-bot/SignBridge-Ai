# SignBridge AI: Preprocessing Execution Readiness Review

**Document ID:** `docs/preprocessing-execution-readiness-review.md`
**Milestone:** Step 2.5E $\to$ Step 2.5F Pre-Execution Technical Audit
**Author:** SignBridge AI ML Engineering
**Date:** November 2026
**Status:** Read-Only Audit Completed (Standing by for approval)
**Primary References:**
- Preprocessing Design: [`docs/preprocessing-design.md`](preprocessing-design.md)
- Extraction Audit: [`docs/dataset-extraction-audit-report.md`](dataset-extraction-audit-report.md)
- Full-Corpus Report: [`docs/full-corpus-preprocessing-report.md`](full-corpus-preprocessing-report.md)
- Extraction Manifest: [`data/raw/isl-csltr/metadata/extraction_manifest.csv`](../data/raw/isl-csltr/metadata/extraction_manifest.csv)
- Corpus Catalog: [`data/raw/isl-csltr/metadata/corpus_dry_run_results.json`](../data/raw/isl-csltr/metadata/corpus_dry_run_results.json)
- Preprocessing Pipeline: [`scripts/preprocess_isl_csltr.py`](../scripts/preprocess_isl_csltr.py)

---

## 1. Executive Summary & Readiness Verdict

Following the extraction of 449 benchmark-eligible videos in Step 2.5E, a rigorous, read-only technical audit was performed to determine whether the codebase, data directories, split routing, and quality-control hierarchy are **100% ready for full-corpus v2 preprocessing**.

### Overall Readiness Verdict: **PASS (APPROVED FOR FULL-CORPUS PREPROCESSING)**

Every required invariant was physically audited against disk state, source code, and catalog metadata. There are **zero blockers** and **zero data integrity anomalies**.

---

## 2. Comprehensive Audit Matrix

| # | Inspection Criterion | Verdict | Technical Evidence & Physical Verification |
| :---: | :--- | :---: | :--- |
| **1** | **All 453 eligible benchmark videos discoverable** | **PASS** | `preprocess_isl_csltr.py` scans `video_dirs = [RAW_DIR / "sample_videos", RAW_DIR / "Videos_Sentence_Level"]`. Script simulation discovered exactly **455 local videos** (453 benchmark-eligible + 2 quarantine diagnostics). Exactly **0** benchmark videos are missing. |
| **2** | **449 extracted + 4 previous accounted for exactly once** | **PASS** | Physical disk inventory confirms: 4 eligible files in `sample_videos/` (`are you free today.mp4`, `are you free today (2).mp4`, `free (2).MP4`, `MVI_6209.MP4`) and 449 eligible files in `Videos_Sentence_Level/`. $4 + 449 = 453$. Zero overlap. |
| **3** | **Zero duplicate videos included** | **PASS** | Filename collision scan between `sample_videos/` and `Videos_Sentence_Level/` returned **0 collisions** (`len(overlap) == 0`). All 453 eligible filenames are strictly unique. |
| **4** | **Train, validation, and test split assignments preserved** | **PASS** | Catalog dry-run routing confirmed: **319 Train** (Signers 1–5), **66 Validation** (Signer 6), **68 Test** (Signer 7). $319 + 66 + 68 = 453$. Matches the Step 2.4C approved audit invariants identically. |
| **5** | **Signer 7 recordings strictly isolated from train/val** | **PASS** | All 68 Signer 7 videos (67 extracted + `MVI_6209.MP4`) route exclusively to `test/`. Signer 7 splits set = `{'test'}`. Zero Signer 7 takes can enter `train/` or `validation/`. |
| **6** | **Zero quarantined recordings enter benchmark splits** | **PASS** | 232 quarantined recordings remain unextracted in `D:\ISL_CSLRT_Corpus.zip`. The 2 local diagnostic quarantine videos in `sample_videos/` route exclusively to `quarantine/` with `suitability_for_training = False`. Quarantined takes in benchmark splits = **0**. |
| **7** | **Windows case-insensitive path and filename matching** | **PASS** | Discovery uses `.name.lower()` mapping keys (`local_video_map[p.name.lower()] = p`), neutralizing Windows case quirks (`.mp4` vs `.MP4`, `(2).mp4` vs `(2).MP4`). Pathlib resolves paths natively. |
| **8** | **Approved v2 feature schema (Candidate B 3D, 229D)** | **PASS** | Topology strictly set to `candidate_b_3d` (99 pose + 63 left hand + 63 right hand + 4 indicators = 229D). Decoupled indicators `[m_left, m_right, m_pose, m_norm]` and wrist-relative depth preserved. |
| **9** | **Orientation correction active for native signers** | **PASS** | `preprocess_video()` applies 90° clockwise rotation (`cv2.rotate(..., cv2.ROTATE_90_CLOCKWISE)`) iff `signer_verified and signer_id in ("1", "2") and orig_width > orig_height`. Unit tests confirm native signers rotate, volunteers/unverified signers do not. |
| **10** | **Resumption & overwrite protection** | **PASS** | Output files target `data/processed/isl_csltr_tensors_v2/{split}/{clip_id}.npz`. Lines 24–31 check `if npz_out_path.exists() and not overwrite: return {"status": "SKIPPED_EXISTING", ...}`. Unrelated directories and existing tensors are protected. |
| **11** | **Failed / low-quality videos explicitly recorded** | **PASS** | Videos failing QC or discovery are logged with detailed error codes in `manifests/failed_videos.csv` and `manifests/quarantine_manifest.csv`. No clip is silently discarded. |
| **12** | **Test suites green & Git cleanliness** | **PASS** | `pytest backend/tests/test_preprocessing.py`: **44/44 PASSED**. `pytest backend/tests/test_archive_downloader.py`: **10/10 PASSED**. Git working tree clean; zero data files tracked. |

---

## 3. Dataset Accounting & Split Allocation

### 3.1 Total Corpus Accounting

$$\text{Total Published Videos (687)} = \text{Benchmark Eligible (453)} + \text{Quarantined (234)}$$

$$\text{Benchmark Eligible (453)} = \text{Train (319)} + \text{Validation (66)} + \text{Test (68)}$$

### 3.2 Physical Storage Allocation

```text
data/raw/isl-csltr/
├── sample_videos/                         (6 videos total)
│   ├── are you free today.mp4            [Signer 1, Native, Train]
│   ├── are you free today (2).mp4        [Signer 2, Native, Train]
│   ├── free (2).MP4                      [Signer 4, Volunteer, Train]
│   ├── MVI_6209.MP4                      [Signer 7, Volunteer, Test]
│   ├── bring water for me (1).MP4        [Quarantine Diagnostic]
│   └── can you repeat that please (3).MP4 [Quarantine Diagnostic]
│
└── Videos_Sentence_Level/                 (449 videos total)
    ├── are you free today/
    │   ├── free (3).MP4                  [Signer 5, Volunteer, Train]
    │   ├── free (4).MP4                  [Signer 3, Volunteer, Train]
    │   └── ...
    └── ... (101 category folders)
```

- **Benchmark Source Videos Physically Available:** $4 + 449 = \mathbf{453}$
- **Quarantine Diagnostic Videos Physically Available:** $\mathbf{2}$
- **Total Local Videos Mapped:** $\mathbf{455}$

---

## 4. Pipeline Execution Simulation

A dry-run simulation of `run_full_corpus_pipeline()` was executed against the exact physical files on disk:
- **Total Catalog Records Read:** 687
- **Eligible Benchmark Videos Processed:** 453
  - Locally available on disk: **453 / 453 (100%)**
  - Missing from disk: **0**
- **Quarantined Records Processed:** 234
  - Locally available diagnostics: **2** (processed into `quarantine/` with `suitability_for_training = False`)
  - Missing unextracted quarantine: **232** (recorded in `quarantine_manifest.csv` as unextracted)
- **Split Routing Output:**
  - `train/`: 319 videos across 71 categories
  - `validation/`: 66 videos across 66 categories
  - `test/`: 68 videos across 68 categories
  - `quarantine/`: 234 records accounted for (2 processed diagnostics, 232 unextracted)

---

## 5. Potential Operational Risks & Mitigations

| Risk | Likelihood | Impact | Built-in Mitigation |
| :--- | :---: | :---: | :--- |
| **Processing Duration & Interruption** | Moderate | Low | Preprocessing 453 videos through MediaPipe Holistic takes ~15–25 minutes depending on CPU. The pipeline includes **per-clip resumption logic**: if interrupted, re-running skips already generated `.npz` files unless `--overwrite` is explicitly passed. |
| **Short Duration QC Failures** | High | Low | Some short videos may fail the 30-frame threshold (e.g. `are_you_free_today_2.npz` has 25 frames). This is expected behavior; such clips will be stamped with `qc_pass = False` and `suitability_for_training = False`. |
| **Disk Space for Processed Tensors** | Low | Low | 453 compressed `.npz` files require ~150–250 MB total. Drive C: has 65.42 GB free, which easily accommodates the output. |

---

## 6. Exact Execution Command

When approved, execute full-corpus v2 preprocessing from the project root:

```powershell
backend\.venv\Scripts\python.exe scripts/preprocess_isl_csltr.py --mode full_corpus
```

### Optional Flags:
- `--overwrite`: Force re-processing of already generated `.npz` files (default: `False`, skips existing valid files).
- `--limit N`: Limit processing to the first $N$ catalog records for quick smoke testing.

---

## 7. Final Recommendation

### **RECOMMENDATION: FULL-CORPUS PREPROCESSING APPROVED FOR EXECUTION**

All source data, split partitions, test isolation gates, and unit test suites are fully verified. You may proceed with running the full-corpus preprocessing command.
