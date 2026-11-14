# SignBridge AI: ISL-CSLTR Dataset Download & Inspection Report

**Document ID:** `docs/isl-csltr-inspection-report.md`  
**Milestone:** Step 2.2A — ISL-CSLTR Dataset Download and Inspection  
**Author:** SignBridge AI ML Engineering  
**Date:** November 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary & Verification Highlights

As part of **Step 2.2A**, the **ISL-CSLTR** (officially designated inside the archive as **ISL-CSLRT**) dataset hosted on Mendeley Data was systematically downloaded, unpacked, and audited using remote byte-range inspection and local extraction of metadata and sample media.

All findings below were verified directly against the author's primary archive.

### Key Audit Findings at a Glance

| Item | Stated in Literature / Web | Actual Verified in Archive | Variance / Status |
| :--- | :--- | :--- | :--- |
| **Archive File** | `kcmpdxky7p-1.zip` | `ISL_CSLRT_Corpus.zip` | Single nested zip file (8.29 GiB / 8,905,822,924 bytes) |
| **Total Videos** | 700 videos | **687 MP4 videos** | **13 videos omitted/missing** from sentence folders |
| **Total Frames** | 18,863 frames | **18,863 JPG frames** | **100% exact match** across 97 sentence categories |
| **Signer Count** | 7 signers | **7 signers (IDs '1'–'7')** | **Verified in frame directories**; website description had a typo |
| **Signer Makeup** | 2 native + 4 students ($=6$) | **2 native + 5 students ($=7$)** | Archive `ISL_CSLRT.txt` clarifies: 5 students, not 4 |
| **Video Resolution** | Controlled standard/HD | **1920 × 1080 (1080p Full HD)** | Canon DSLR camera; some folders also contain 319 × 240 previews |
| **Frame Rate** | ~25–30 fps | **25.00 fps** | Exact 25 fps (`timescale: 25000, duration ticks: 78000–95000`) |
| **Gloss Alignment** | Stated as "annotated" | **Global sentence-to-gloss only** | **❌ No frame-by-frame temporal timestamps** for individual glosses |
| **Authentication Gate** | Mendeley login check | **Public direct S3 download** | No credentials or account required (`allowAnonymous: true`) |

---

## 2. Official Download Source & Access Mechanics

- **Mendeley Data DOI:** `10.17632/kcmpdxky7p.1`
- **Mendeley Landing Page:** [https://data.mendeley.com/datasets/kcmpdxky7p/1](https://data.mendeley.com/datasets/kcmpdxky7p/1)
- **Direct S3 File Endpoint:**  
  `https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/...` (via Mendeley API `public-files/datasets/kcmpdxky7p/files/c14c8953-be53-4c7a-85d5-5bcd7e79d86e/file_downloaded`)
- **Primary Archive Filename:** `ISL_CSLRT_Corpus.zip`
- **File Size:** `8,905,822,924` bytes (8.29 GiB / 8.91 GB)
- **SHA-256 Hash:** `1a849c89c1d53b7e9ce43dc24f6d4f0afd95e707bf256febc26716782b736686`
- **Authentication Required:** **None**. The Mendeley public API permits unauthenticated anonymous direct downloads with byte-range resume support (`Accept-Ranges: bytes`).
- **Local Storage Location:**  
  Extracted metadata and inspection samples reside in:  
  `data/raw/isl-csltr/metadata/` and `data/raw/isl-csltr/sample_videos/`  
  Both are confirmed excluded from Git tracking by `.gitignore` (line 77: `data/`).

---

## 3. Archive Hierarchy & File Structure

The archive contains a total of **22,073 entries** organized under the single root directory `ISL_CSLRT_Corpus/`:

```text
ISL_CSLRT_Corpus/
├── ISL_CSLRT.txt                                    # Primary dataset documentation & contributor manifest
├── corpus_csv_files/
│   ├── ISL Corpus sign glosses.csv                 # 101 English sentences mapped to uppercase ISL glosses
│   ├── ISL_CSLRT_Corpus details.xlsx               # 792 indexed video records and paths
│   ├── ISL_CSLRT_Corpus_frame_details.xlsx         # 18,863 frame path records
│   └── ISL_CSLRT_Corpus_word_details.xlsx          # 18,863 word frame records across 115 word classes
├── Videos_Sentence_Level/                           # 687 MP4 video files across 101 sentence directories
│   ├── are you free today/
│   │   ├── free (2).MP4
│   │   ├── free (3).MP4
│   │   └── ...
│   ├── bring water for me/
│   └── ... (101 sentence subdirectories)
├── Frames_Sentence_Level/                           # 18,863 extracted JPG frames (240x320)
│   ├── are you free today/
│   │   ├── 1/                                       # Signer 1 frame sequence (01.jpg, 02.jpg...)
│   │   ├── 2/                                       # Signer 2 frame sequence
│   │   └── ... (up to 7 signers)
│   └── ... (97 sentence subdirectories)
└── Frames_Word_Level/                               # 1,036 word-level frame segments
    ├── A LOT/
    ├── ABUSE/
    └── ... (115 word subdirectories)
```

### File Extension Distribution in Archive
- `.jpg`: **19,899** (18,863 sentence-level frames + 1,036 word-level frames)
- `.mp4` / `.MP4`: **687** video recordings
- `.xlsx`: **3** Microsoft Excel metadata workbooks
- `.csv`: **1** sign gloss mapping file
- `.txt`: **1** dataset manifest
- `.db`: **483** (Windows `Thumbs.db` system thumbnail caches left by authors)
- Directory entries without extension: **990**

---

## 4. Metadata Schema & Sample Rows

### 4.1. `ISL Corpus sign glosses.csv`
- **Columns:** `['Sentence', 'SIGN GLOSSES']`
- **Total Rows:** 101 mapped sentence pairs (header + 101 data rows)
- **Sample Entries:**
  ```csv
  Sentence,SIGN GLOSSES
  are you free today,YOU FREE TODAY
  are you hiding something,YOU HIDE SOMETHING
  bring water for me,BRING WATER ME
  can i help you,I HELP YOU
  can you repeat that please,YOU REPEAT PLEASE
  he came by train,HE CAME TRAIN
  He is going into the room,HE GO  INTO  ROOM
  ```

### 4.2. `ISL_CSLRT_Corpus details.xlsx`
- **Columns:** `['Sentences', 'File location']`
- **Total Rows:** 792 indexed video records across 102 unique sentence phrases
- **Sample Entries:**
  | Sentences | File location |
  | :--- | :--- |
  | `are you free today` | `ISL_CSLRT_Corpus\Videos_Sentence_Level\are you free today\free (2).MP4` |
  | `are you free today` | `ISL_CSLRT_Corpus\Videos_Sentence_Level\are you free today\free (3).MP4` |
  | `are you free today` | `ISL_CSLRT_Corpus\Videos_Sentence_Level\are you free today\free (4).MP4` |
  | `are you free today` | `ISL_CSLRT_Corpus\Videos_Sentence_Level\are you free today\free.MP4` |

*Note: The workbook lists 792 paths because some sentences list alternate takes or word clips, but only 687 distinct MP4 video files are physically present in `Videos_Sentence_Level/`.*

### 4.3. `ISL_CSLRT_Corpus_frame_details.xlsx`
- **Columns:** `['Sentence', 'Frames path']`
- **Total Rows:** Exactly **18,863** frame path rows
- **Sample Entries:**
  | Sentence | Frames path |
  | :--- | :--- |
  | `are you free today` | `ISL_CSLRT_Corpus\Frames_Sentence_Level\are you free today\1\are you free today 01.jpg` |
  | `are you free today` | `ISL_CSLRT_Corpus\Frames_Sentence_Level\are you free today\1\are you free today 02.jpg` |
  | `are you free today` | `ISL_CSLRT_Corpus\Frames_Sentence_Level\are you free today\2\are you free today 01.jpg` |

---

## 5. Video File Technical Specifications

Extracted video samples (`free (2).MP4`, `MVI_6209.MP4`, `are you free today.mp4`) were analyzed using ISO/IEC 14496-12 box inspection:

- **Video Container:** ISO Base Media File Format (MP4 / QuickTime)
- **Video Compression Codec:** **H.264 / AVC (`avc1`)**
- **Recording Equipment:** **Canon EOS Digital SLR Cameras** (evidenced by atom tags `CAEP` and filename pattern `MVI_xxxx.MP4`)
- **Native Video Resolution:** **$1920 \times 1080$ (1080p Full HD)** at 16:9 aspect ratio
- **Secondary / Preview Resolution:** In certain folders, downscaled **$319 \times 240$** QVGA versions are present alongside the 1080p originals.
- **Pre-Extracted JPEG Frames Resolution:** All 18,863 pre-extracted frames in `Frames_Sentence_Level/` are **$240 \times 320$** pixels.
- **Frame Rate (FPS):** **25.00 fps** (verified via `mvhd` timescale of 25,000 Hz and `stts` sample deltas of 1,000 ticks per frame).
- **Clip Durations:** Between **2.00 seconds and 5.00 seconds** per sentence execution (typically 50 to 125 frames per clip).

---

## 6. Signer Resolution & Discrepancy Clarification

### 6.1. The Website vs. Archive Manifest Contradiction
- **The Web Page Discrepancy:** The Mendeley Data webpage states:  
  *"created with two native signers ... and four student volunteers"* ($2 + 4 = 6$ signers).
- **The Empirical Archive Truth:** The author's readme file inside the root archive (`ISL_CSLRT.txt`) states:  
  *"created with two native signers of Navajeevan Residential School for the Deaf ... on 28-12-2019 and **five student volunteers** from SASTRA Deemed University ... performed by **7 different Signers**."*
- **Resolution:** The web page text contained an arithmetic typo ("four" instead of "five"). The dataset was indeed produced by **7 signers** ($2 \text{ native} + 5 \text{ students} = 7$).

### 6.2. Verified Signer IDs & Sample Distribution
Signer IDs are explicitly designated by the directory structure in `Frames_Sentence_Level/<sentence>/<signer_id>/<frame>.jpg`:

| Signer ID | Frame Count | Percentage | Sentence Coverage (out of 97) |
| :---: | :---: | :---: | :---: |
| **'1'** | 2,203 frames | 11.68% | 96 sentences |
| **'2'** | 1,674 frames | 8.87% | 96 sentences |
| **'3'** | 2,705 frames | 14.34% | 96 sentences |
| **'4'** | 2,656 frames | 14.08% | 90 sentences |
| **'5'** | 3,341 frames | 17.71% | 96 sentences |
| **'6'** | 3,053 frames | 16.19% | 96 sentences |
| **'7'** | 3,231 frames | 17.13% | 93 sentences |
| **Total** | **18,863 frames** | **100.0%** | **97 sentences** |

- **84 sentences** contain performances from all 7 signers.
- **12 sentences** contain performances from 6 signers.
- **1 sentence** contains performances from 3 signers.

---

## 7. Missing, Omitted, or Anomalous Files

During the audit, the following empirical anomalies were identified:

1. **687 MP4 Videos vs. 700 Stated:**  
   The authors cite "700 fully annotated videos", but only **687 MP4 files** are present in `Videos_Sentence_Level/`. Thirteen videos were omitted from the archive.
2. **Video Distribution Imbalance:**  
   While 79 sentence categories have exactly 7 video files:
   - 12 categories have only 6 videos.
   - 2 categories (`No need to worry dont worry`, `i dont agree`) have only 1 video.
   - 1 category (`what have you planned for your career`) has only 3 videos.
   - 6 categories have 8 videos, and 1 category (`serve the food`) has 9 videos due to duplicate take variations.
3. **Inconsistent Filename Conventions:**  
   Files alternate between:
   - Descriptive numbered names: `bring water for me (1).MP4`
   - Raw Canon camera names: `MVI_6207.MP4`, `MVI_6505.MP4`
   - Short word aliases: `free (2).MP4`, `help (3).MP4`, `serve 1.MP4`
4. **Sentence Frame Directory Gap:**  
   While `Videos_Sentence_Level/` has 101 sentence folders, `Frames_Sentence_Level/` has **97 sentence folders** (4 sentences lack pre-extracted frame subdirectories).
5. **Author Artifacts:**  
   483 Windows `Thumbs.db` files are bundled across subdirectories.

---

## 8. Presence/Absence of Gloss Labels and Timestamps

- **Global Sentence-Level Gloss Mapping: `[PRESENT]`**  
  `ISL Corpus sign glosses.csv` provides a valid 1-to-1 mapping from English sentences to sign gloss token strings (e.g., `"can i help you" -> "I HELP YOU"`).
- **Frame-by-Frame Temporal Gloss Alignments: `[ABSENT]`**  
  **No continuous frame timestamps or temporal segment boundaries exist for individual sign glosses.**  
  There is no timestamp stating that frames 1–25 represent `"I"`, frames 26–45 represent `"HELP"`, and frames 46–65 represent `"YOU"`.
- **Engineering Implication:**  
  Connectionist Temporal Classification (CTC) sequence-to-sequence gloss models (Paradigm 2) cannot be trained on this dataset without manual phonetic annotation. The dataset is strictly suited for **Paradigm 1: Closed-Set Sentence Classification**.

---

## 9. Implications for SignBridge AI Engineering

### 9.1. Baseline Modeling Feasibility (Step 2.3)
1. **Target Formulation:** Multi-class sentence classification over $K=100$ validated phrases.
2. **Video Feature Extraction:** Run the MediaPipe Tasks Hand and Pose Landmarker directly on the 1080p MP4 videos (or the pre-extracted $240 \times 320$ frames) to generate normalized $T \times 162$ feature tensors ($T=60$).
3. **Temporal Uniformity:** Resample or pad sequences to a uniform $T=60$ frames, matching the observed natural duration range (50–125 frames at 25 fps).

### 9.2. Recommended Signer-Independent Split Strategy
To evaluate real-world generalization and avoid memorization of individual signer physique or background cues, the following partition is recommended based on verified signer IDs:

```text
┌───────────────────────────────────────────────────────────────┐
│                      ISL-CSLTR Corpus                         │
│                    (18,863 frames / 687 clips)                │
└──────────────────────────────┬────────────────────────────────┘
                               │
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                        ▼
  Train Set                Validation Set           Test Set
  Signers: 1, 2, 3, 4, 5   Signer: 6                Signer: 7
  12,579 frames (66.7%)    3,053 frames (16.2%)     3,231 frames (17.1%)
```

- **Leakage Prevention:** Because signers are partitioned strictly by directory ID, no frames, video takes, or recording sequences from Signers 6 or 7 will appear in the training split.

---

## 10. Reproducibility

To re-run and verify the metadata audit locally at any time, execute:

```bash
python scripts/inspect_isl_csltr.py
```

The script reads the extracted metadata files from `data/raw/isl-csltr/metadata/` and the central directory manifest without requiring external packages.

