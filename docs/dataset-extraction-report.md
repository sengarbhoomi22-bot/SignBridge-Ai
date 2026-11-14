# SignBridge AI: Step 2.5B Dataset Extraction Report

**Document ID:** `docs/dataset-extraction-report.md`
**Milestone:** Step 2.5B — Selective ISL-CSLTR Dataset Extraction
**Author:** SignBridge AI ML Engineering
**Date:** November 2026
**Status:** BLOCKER — Archive Not Locally Available

**Primary References:**
- Archive Index: [`data/raw/isl-csltr/metadata/csltr_file_list.json`](../data/raw/isl-csltr/metadata/csltr_file_list.json)
- Corpus Catalog: [`data/raw/isl-csltr/metadata/corpus_dry_run_results.json`](../data/raw/isl-csltr/metadata/corpus_dry_run_results.json)
- Failed Videos Manifest: [`data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv`](../data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv)
- Cross-Reference Results: [`data/raw/isl-csltr/metadata/archive_cross_reference.json`](../data/raw/isl-csltr/metadata/archive_cross_reference.json)
- Dataset Readiness Audit: [`docs/dataset-readiness-audit.md`](dataset-readiness-audit.md)

---

## 1. Executive Summary

This report documents the archive inspection, cross-reference verification, disk-space analysis, and extraction safety assessment for the ISL-CSLTR corpus. All extraction preconditions have been verified against the archive's central-directory index (`csltr_file_list.json`) and the corpus catalog.

**Primary finding:** Every one of the 449 missing benchmark-eligible videos is confirmed present in the Mendeley archive index with no duplicates or naming ambiguities. Disk space is sufficient. However, **the physical ZIP archive (`ISL_CSLRT_Corpus.zip`, 8.29 GB) is not present on the local filesystem** (searched both C: and D: drives exhaustively). The extraction cannot proceed until the archive is downloaded or transferred to local disk.

---

## 2. Archive Inspection (Read-Only)

### 2.1 Archive Index Structure

The archive's central directory was previously read and stored in `data/raw/isl-csltr/metadata/csltr_file_list.json`. This index was produced at an earlier stage of the project and contains the complete file listing of `ISL_CSLRT_Corpus.zip` without requiring the archive to be re-opened.

| Index Metric | Value |
| :--- | :---: |
| Total index entries (files + directories) | 22,073 |
| Directory entries | 999 |
| File entries | 21,074 |
| Video entries in `Videos_Sentence_Level/` (non-empty MP4) | **687** |
| Unique video basenames | **687** (0 duplicates) |
| Total uncompressed content | **8.451 GB** |
| Total compressed content | **8.289 GB** |
| Highest file offset in archive | 8.291 GB |

The archive contains exactly the 687 videos described in the published dataset — a 1-to-1 match with the corpus catalog.

### 2.2 Archive Physical Location Search

An exhaustive filesystem search was performed across both local drives for any file matching `ISL_CSLRT_Corpus.zip` or any file larger than 500 MB on C: and all files on D::

| Drive | Free Space | Search Result |
| :--- | :---: | :--- |
| C: (OS) | 65.42 GB | ❌ No ISL archive found |
| D: (New Volume) | 188.38 GB | ❌ No ISL archive found |

**Conclusion:** The 8.29 GB archive is not present on local disk. The `csltr_file_list.json` index was built from the archive's central directory (a lightweight operation that does not require keeping the archive on disk) and correctly represents the archive contents, but the ZIP itself must be downloaded before selective extraction can proceed.

---

## 3. Archive–Catalog Cross-Reference Verification

### 3.1 Methodology

Cross-referencing was performed by the script `scripts/scratch_inspect_archive_index.py` (read-only, non-destructive) against three sources:

1. **Archive index** (`csltr_file_list.json`): 687 non-empty MP4 entries, all with unique basenames.
2. **Corpus catalog** (`corpus_dry_run_results.json`): 453 `VERIFIED_UNIQUE` benchmark-eligible records.
3. **Failed videos manifest** (`failed_videos.csv`): 449 benchmark-eligible videos with `failure_stage: FILE_DISCOVERY`.

### 3.2 Cross-Reference Results

| Cross-Reference Check | Result |
| :--- | :--- |
| Total catalog videos matching archive entry | **687 / 687** ✅ |
| Duplicate archive basenames | **0** ✅ |
| Ambiguous archive matches (same filename in multiple categories) | **0** ✅ |
| Pending eligible videos found unambiguously in archive | **449 / 449** ✅ |
| Pending eligible videos NOT in archive | **0** ✅ |
| Quarantined videos in archive | **234 / 234** ✅ |

**All 449 missing benchmark-eligible videos are confirmed present in the archive with unambiguous one-to-one filename matches.** No manual disambiguation or guesswork is required.

### 3.3 Split Breakdown of Videos Pending Extraction

| Split | Signers | Videos Pending | Purpose |
| :--- | :--- | :---: | :--- |
| `train/` | Signers 1–5 | **316** | Training data for model development |
| `val/` | Signer 6 | **66** | Validation for hyperparameter tuning |
| `test/` | Signer 7 | **67** | Held-out final evaluation |
| **Total** | | **449** | |

> [!IMPORTANT]
> The 67 pending test videos belong to Signer 7, the held-out test signer. They must be extracted into `data/raw/isl-csltr/Videos_Sentence_Level/` and the preprocessing pipeline will automatically route them to the `test/` split via `KNOWN_CATALOG`. They must never be used for training, validation, or feature selection.

### 3.4 Non-Extraction Confirmation (Quarantined Videos)

The 234 quarantined videos are confirmed present in the archive index. They **must not be extracted** into the benchmark source directory. The extraction plan targets only the 449 `VERIFIED_UNIQUE` benchmark-eligible files.

---

## 4. Disk-Space Analysis

### 4.1 Extraction Size Requirements

| Metric | Value |
| :--- | :---: |
| Videos to extract | 449 |
| Total uncompressed size | **4.650 GB** |
| Total compressed size in archive | **4.622 GB** |
| Recommended free space buffer (1.5×) | ≈ 7.0 GB |

> [!NOTE]
> Since MP4 files are already compressed, ZIP deflation ratios for video content are typically very close to 1:1 (here: 4.650 GB uncompressed vs 4.622 GB compressed, ratio ≈ 0.6% compression saving). The effective disk cost of extraction is essentially the full uncompressed size.

### 4.2 Available Disk Space

| Drive | Total | Free | Needed | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| C: (OS) | 257.82 GB | **65.42 GB** | 4.65 GB | ✅ **Sufficient** (60 GB headroom) |
| D: (New Volume) | 195.80 GB | **188.38 GB** | 4.65 GB | ✅ **Sufficient** (183 GB headroom) |

Either drive has sufficient space. The recommended extraction target is under the existing project data directory at `data/raw/isl-csltr/Videos_Sentence_Level/` on C: (consistent with the pipeline's configured search paths).

---

## 5. Duplicate and Integrity Checks

### 5.1 Archive-Level Duplicate Check

The 687 video basenames in the archive are **all unique** — no two videos share the same filename. This is consistent with the corpus design, where sentence categories are grouped into subdirectories (`Videos_Sentence_Level/<sentence>/`), so a video named `are you free today (2).mp4` appears in exactly one category subfolder.

### 5.2 Existing Local Files (Do Not Overwrite)

The following 4 benchmark-eligible videos already exist locally in `data/raw/isl-csltr/sample_videos/` and **must not be overwritten**:

| Existing Local File | Size | Assigned Split |
| :--- | :---: | :--- |
| `are you free today.mp4` | 620 KB | `train/` (Signer 1) |
| `are you free today (2).mp4` | 56 KB | `train/` (Signer 2) |
| `free (2).MP4` | 10.7 MB | `train/` (Signer 4) |
| `MVI_6209.MP4` | 13.3 MB | `test/` (Signer 7) |

The preprocessing pipeline's `FILE_DISCOVERY` stage searches both `sample_videos/` and `Videos_Sentence_Level/` directories. After extraction, the 4 existing files remain untouched in `sample_videos/` and the new 449 files go into `Videos_Sentence_Level/`.

### 5.3 Signer Identity Verification

Signer assignments are stored in `corpus_dry_run_results.json` and enforced by `KNOWN_CATALOG` in `scripts/preprocess_isl_csltr.py`. The extraction does **not** assign signer identities — it only places raw video files in the filesystem. Signer-to-split routing happens deterministically during preprocessing via the catalog.

---

## 6. Extraction Safety Plan

### 6.1 Target Directory Structure

The videos will be extracted into the pipeline's pre-configured secondary search path, preserving the archive's internal folder structure:

```text
data/raw/isl-csltr/
├── sample_videos/           ← Existing 4 benchmark + 2 quarantine files (DO NOT TOUCH)
└── Videos_Sentence_Level/   ← New extraction target (to be created)
    ├── are you free today/
    │   ├── are you free today (3).MP4   ← Signer 5, train
    │   └── ...
    ├── are you hiding something/
    │   └── ...
    └── ... (101 sentence category subfolders)
```

> [!CAUTION]
> Do not extract quarantined videos (the 234 `AMBIGUOUS`, `UNVERIFIED`, or `NO_FRAME_DIR` entries) into `Videos_Sentence_Level/`. The preprocessing pipeline uses `KNOWN_CATALOG` to route known files, but an unrecognized file in the search path could trigger unexpected behavior. Extract only the 449 `VERIFIED_UNIQUE` files.

### 6.2 Extraction Command (Ready to Execute When Archive Is Available)

Once `ISL_CSLRT_Corpus.zip` is downloaded to a known path (e.g. `D:\ISL_CSLRT_Corpus.zip`), the following selective extraction procedure is safe to run:

```python
# scripts/extract_benchmark_videos.py
# Run as: python scripts/extract_benchmark_videos.py --archive D:\ISL_CSLRT_Corpus.zip
# This script will be authored as a separate Step 2.5B deliverable.
# It will:
#   1. Open the archive in read-only mode.
#   2. Extract only the 449 filenames listed in failed_videos.csv.
#   3. Preserve the archive's internal folder path under Videos_Sentence_Level/.
#   4. Skip any filename already present on disk (no-overwrite).
#   5. Skip any filename not in the VERIFIED_UNIQUE benchmark set.
#   6. Verify extracted file size against archive index uncompressed_size.
#   7. Write an extraction log to data/raw/isl-csltr/metadata/extraction_log.json.
```

> [!IMPORTANT]
> The extraction script must accept the archive path as an explicit CLI argument. It must never attempt to download the archive automatically. It must never extract quarantined videos.

---

## 7. Signer and Split Integrity Confirmation

| Integrity Check | Status |
| :--- | :--- |
| Signer 7 test isolation enforced by `KNOWN_CATALOG` | ✅ Confirmed |
| 234 quarantined videos excluded from extraction target | ✅ Confirmed |
| No filename collision between pending 449 and existing 4 local files | ✅ Confirmed |
| Signer assignments in `corpus_dry_run_results.json` unchanged | ✅ Confirmed (read-only audit) |
| All 449 pending files map 1-to-1 to archive entries | ✅ Confirmed |
| Zero ambiguous or missing files | ✅ Confirmed |

---

## 8. Current Blocker

> [!WARNING]
> **Extraction cannot proceed.** The physical ZIP archive `ISL_CSLRT_Corpus.zip` (8.29 GB) is confirmed absent from both local drives (C: and D:). The `csltr_file_list.json` index is a metadata snapshot of the archive's central directory, not the archive itself.

**To unblock Step 2.5B, one of the following actions is required:**

| Option | Action | Notes |
| :--- | :--- | :--- |
| **Option 1 (Preferred)** | Download `ISL_CSLRT_Corpus.zip` from Mendeley (DOI: `10.17632/5dxqnp93p6.1`) | Requires network access; 8.29 GB download |
| **Option 2** | Transfer the archive from an external drive or USB where it is stored | No download required |
| **Option 3** | Locate a pre-extracted `Videos_Sentence_Level/` folder and copy it to `data/raw/isl-csltr/` | Acceptable if complete and unmodified |

Once the archive or the extracted folder is available at a known path, extraction can proceed using the selective extraction script (Section 6.2 above). Disk space is confirmed sufficient on both drives.

---

## 9. Inventory Verification (Read-Only, Post-Inspection)

### Current State (Before Extraction)

| Location | Contents |
| :--- | :--- |
| `data/raw/isl-csltr/sample_videos/` | 4 benchmark MP4 + 2 quarantine diagnostic MP4 |
| `data/raw/isl-csltr/Videos_Sentence_Level/` | Does not exist (not yet created) |
| `data/processed/isl_csltr_tensors_v2/train/` | 2 tensors (`are_you_free_today.npz`, `are_you_free_today_2.npz`) |
| `data/processed/isl_csltr_tensors_v2/validation/` | 0 tensors |
| `data/processed/isl_csltr_tensors_v2/test/` | 1 tensor (`mvi_6209.npz`) |
| `data/processed/isl_csltr_tensors_v2/quarantine/` | 2 diagnostic tensors |

### Expected State (After Successful Extraction + v2 Preprocessing)

| Location | Expected Contents |
| :--- | :--- |
| `data/raw/isl-csltr/Videos_Sentence_Level/` | 449 MP4 files in 101 sentence category subfolders |
| `data/processed/isl_csltr_tensors_v2/train/` | ≤ 319 tensors (Signers 1–5, 71 classes) |
| `data/processed/isl_csltr_tensors_v2/validation/` | ≤ 66 tensors (Signer 6, 66 classes) |
| `data/processed/isl_csltr_tensors_v2/test/` | ≤ 68 tensors (Signer 7, 68 classes) |

> [!NOTE]
> Counts are ≤ design maximums because some videos may fail per-video QC (minimum duration, landmark detection thresholds). The 3-level QC hierarchy in `preprocess_isl_csltr.py` is authoritative.

---

## 10. Exact Next Steps

1. **Obtain the archive** using one of the three options in Section 8.
2. **Note the archive path** (e.g. `D:\ISL_CSLRT_Corpus.zip` or wherever it is stored/downloaded).
3. **Run selective extraction** (a separate `scripts/extract_benchmark_videos.py` to be authored):
   ```
   python scripts/extract_benchmark_videos.py --archive <path_to_zip>
   ```
4. **Verify extraction**:
   ```
   python scripts/verify_v2_tensors.py --mode inventory
   ```
5. **Run full-corpus v2 preprocessing**:
   ```
   python scripts/preprocess_isl_csltr.py --mode full_corpus
   ```
6. **Validate tensor counts**: Confirm `data/processed/isl_csltr_tensors_v2/` contains ≥ 300 training tensors across ≥ 70 of 71 classes before proceeding to Step 2.5C (PyTorch pipeline smoke test).

---

*No files were modified, staged, committed, or pushed during this audit. The ZIP archive was not downloaded, extracted, or modified.*

