# SignBridge AI: Step 3 Static Code Quality & ML Architecture Audit

**Document ID:** `docs/ml-code-quality-audit.md`  
**Milestone:** Step 3 — Machine Learning Model Development & Training Baseline  
**Target Architecture:** Mask-Aware Bidirectional GRU Sequence Classifier  
**Status:** Code Quality & Correctness Audit  
**Auditor:** SignBridge AI ML Engineering  
**Date:** November 2026  

---

## 1. Executive Summary

This document presents a rigorous static code review and architectural correctness audit of the machine learning implementation scaffolding created for Step 3 of SignBridge AI. 

The audit verified mathematical formulation, variable-sequence batch collation, zero-data-fabrication compliance, strict signer-independent partitioning, gradient flow, and regression test suites.

**Audit Verdict:** ✅ **PASS — Code scaffolding is mathematically sound and strictly conforms to all engineering constraints.** Ready for pre-flight dataset validation once full-corpus preprocessing completes.

---

## 2. Files Reviewed

| File Path | Component | Scope / Functionality |
|---|---|---|
| `backend/app/ml/labels.py` | Label Management | Deterministic `LabelEncoder` fitted strictly on training partition with JSON serialization. |
| `backend/app/ml/collate.py` | Batch Collation | `SequenceBatchCollator` with variable-length padding and decoupled sequence padding masks. |
| `backend/app/ml/metrics.py` | Evaluation Metrics | Multi-class accuracy, macro/per-class precision, recall, F1, and confusion matrix. |
| `backend/app/ml/dataset.py` | PyTorch Dataset | `ISLLandmarkDataset` loading `.npz` archives with QC filtering and split routing. |
| `backend/app/ml/models/gru_classifier.py` | Neural Model | Mask-aware bidirectional GRU classifier using PyTorch `PackedSequence`. |
| `backend/app/ml/training/config.py` | Configuration | Reproducible `TrainingConfig` dataclass with full parameter serialization. |
| `backend/app/ml/training/train.py` | Training Pipeline | Epoch loop with AdamW, early stopping, gradient clipping, and validation tracking. |
| `backend/app/ml/training/evaluate.py` | Evaluation & Baseline | Model evaluation and majority-class reference baseline. |
| `backend/app/ml/inference.py` | Inference Interface | `SignBridgeInferenceEngine` with shape gating, NaN rejection, and confidence thresholding. |
| `scripts/validate_ml_dataset.py` | Pre-Flight Validator | Automated 18-point dataset validator for Step 3 Phase A. |
| `scripts/train_gru.py` | Training CLI | Command-line training entrypoint. |
| `scripts/evaluate_gru.py` | Evaluation CLI | Single-shot held-out test evaluation entrypoint. |
| `backend/tests/test_ml_components.py` | Test Suite | 11 unit tests covering encoding, collation, metrics, and inference gating. |
| `notebooks/SignBridge_GRU_Training.ipynb` | Colab Workflow | Self-contained Jupyter notebook for optional cloud GPU training. |
| `backend/requirements-ml.txt` | Dependencies | ML dependencies including `torch>=2.2.0` and `scikit-learn>=1.4.0`. |
| `.gitignore` | Version Control | Model weights (`*.pt`, `*.pth`) and checkpoint directory protections. |

---

## 3. Mathematical & Machine Learning Correctness Verification

### A. Tensor Dimensions
- **Input Dimension:** Exactly 229 features per frame (Candidate B 3D: 99 pose + 63 left hand + 63 right hand + 4 presence/normalization indicators). Verified across `gru_classifier.py`, `collate.py`, `config.py`, `inference.py`, and `validate_ml_dataset.py`.
- **Variable Lengths:** Preserved per sequence as integer length vector $\mathbf{L} = (T_1, \dots, T_B)$. Passed into `pack_padded_sequence(..., enforce_sorted=False)`.
- **Output Dimension:** Dynamically matches the training split vocabulary size ($C = \text{num\_classes}$) without hardcoded constants.

### B. Padding & Mask Decoupling
- **Decoupled Padding Mask:** `collate.py` generates a dedicated binary padding mask $\mathbf{M}_{\text{pad}} \in \{0.0, 1.0\}^{B \times T_{\max}}$. Valid frames have mask $= 1.0$; zero-padded frames have mask $= 0.0$.
- **Modality Mask Preservation:** The four landmark presence indicators $[m_{\text{left}}, m_{\text{right}}, m_{\text{pose}}, m_{\text{norm}}]$ embedded in feature indices 225–228 remain intact and are preserved separately in `presence_masks`.
- **Zero Impact of Padding on GRU:**
  - `pack_padded_sequence` compresses batches so recurrent transitions only execute across actual frames $t \le T_i$.
  - In `gru_classifier.py`, endpoint extraction explicitly indexes `last_indices = lengths - 1`:
    $$\mathbf{e}_i = [\vec{\mathbf{h}}_{i, T_i-1} \,\|\, \overleftarrow{\mathbf{h}}_{i, 0}] \in \mathbb{R}^{2H}$$
    Padded frames at $t > T_i - 1$ are completely excluded from the sequence representation.

### C. Dataset Integrity & Signer Isolation
- **Verified Metadata:** Provenance and signer identity are extracted from verified metadata payloads inside `.npz` files.
- **Strict Partitioning:**
  - Train: Signers 1, 2, 3, 4, 5.
  - Validation: Signer 6 (used exclusively for early stopping and model selection).
  - Test: Signer 7 (held-out volunteer signer, strictly evaluated once).
- **Zero Label Leakage:** `LabelEncoder` is fitted in `train.py` strictly on `train_dataset.get_labels()`. The validation and test sets cannot introduce unseen classes into the model's vocabulary during training.

### D. Training Loop Correctness
- **Raw Logits to CrossEntropyLoss:** `SignBridgeGRUClassifier.forward()` returns unnormalized logits. `CrossEntropyLoss()` directly consumes logits without an intermediate softmax, preserving numerical stability.
- **Validation Monitoring:** Early stopping tracks `epoch_val_f1` (macro-F1) on the validation partition with configurable patience (default 10 epochs).
- **Test Set Protection:** The test partition (Signer 7) is never accessed or referenced during training.
- **Gradient Regularization:** Gradient norm clipping (`clip_grad_norm_ <= 1.0`) prevents exploding gradients across recurrent steps.

### E. Evaluation & Baseline Correctness
- **Macro-F1 Calculation:** Macro-F1 is computed as the unweighted mean of per-class F1 scores. Zero-division cases (e.g. classes with zero support in validation or test) are safely handled with $0.0$ default values.
- **Majority-Class Baseline:** Mode class is computed strictly from training labels using `Counter(train_labels).most_common(1)[0]`. The test set labels are never inspected to derive the majority predictor.

### F. Inference & Calibration Transparency
- **Validation Gating:** `SignBridgeInferenceEngine.validate_input()` checks 2D array shape, feature dimension (229), minimum frame duration ($\ge 10$ frames), and finiteness (no NaN or Inf).
- **Confidence Calibration Disclaimer:** The documentation and docstrings explicitly specify that raw softmax probabilities are heuristic scores and must not be interpreted as calibrated Bayesian posteriors. Predictions below the threshold trigger `LOW_CONFIDENCE` status.

---

## 4. Test Verification & Investigation of Previous Exit Code 1

### A. Independent Test Suite Execution Results

#### 1. ML Component Test Suite (`backend/tests/test_ml_components.py`)
- **Command:** `backend\.venv\Scripts\python.exe -m pytest backend/tests/test_ml_components.py -v`
- **Result:** **11 passed in 0.59s**
- **Exit Code:** `0` (Success)
- **Coverage:** Label encoding, variable-length collation, mask decoupling, classification metrics, configuration serialization, majority baseline, and inference validation.

#### 2. Preprocessing Pipeline Regression Suite (`backend/tests/test_preprocessing.py`)
- **Command:** `backend\.venv\Scripts\python.exe -m pytest backend/tests/test_preprocessing.py -v`
- **Result:** **44 passed in 0.49s**
- **Exit Code:** `0` (Success)
- **Coverage:** Landmark dimensions, coordinate normalization, zero-fabrication, signer routing, orientation correction (Signers 1 & 2), and manifest accounting.

#### 3. Offline Archive Downloader Suite (`backend/tests/test_archive_downloader.py -m "not network"`)
- **Command:** `backend\.venv\Scripts\python.exe -m pytest backend/tests/test_archive_downloader.py -m "not network" -v`
- **Result:** **6 passed, 4 deselected in 0.10s**
- **Exit Code:** `0` (Success)
- **Coverage:** Byte formatting, ZIP integrity validation, corrupted archive handling, and disk space pre-flight checks.

---

### B. Root Cause Analysis of Previous Exit Code 1

In the previous turn, the combined command:
`backend\.venv\Scripts\python.exe -m pytest backend/tests/test_preprocessing.py backend/tests/test_archive_downloader.py -v`
exited with code `1`.

#### Findings:
1. **Network Sandbox Isolation:** 4 test functions in `test_archive_downloader.py` are decorated with `@pytest.mark.network`:
   - `test_fetch_dataset_metadata`
   - `test_resolve_archive_info`
   - `test_live_range_request_and_magic_header`
   - `test_simulated_interrupted_download_and_resume`
2. **Failure Mechanism:** These 4 tests execute live HTTP requests to `https://data.mendeley.com/public-api/datasets/kcmpdxky7p`. Because standard terminal execution runs inside a local secure sandbox without network egress, DNS resolution failed:
   `urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>`
3. **No Code Defects:** When network tests are deselected (`-m "not network"`), all 6 offline downloader tests passed. Furthermore, all 44 preprocessing tests and all 11 ML component tests passed completely.
4. **Resolution:** Offline development and regression testing should standardly run with `-m "not network"`. Network tests are reserved for explicitly authorized network execution environments.

---

## 5. Dependency & Git Integrity Audit

### Dependencies (`backend/requirements-ml.txt`)
- Existing: `mediapipe==0.10.21`, `opencv-python==4.11.0.86`, `opencv-contrib-python==4.11.0.86`, `numpy==1.26.4`.
- Added for Step 3: `torch>=2.2.0`, `scikit-learn>=1.4.0`.
- **Environment Compatibility:** Tested under Python 3.12.10 (`win32`).
- **Graceful Import Fallbacks:** All new ML modules (`gru_classifier.py`, `inference.py`, `dataset.py`, `collate.py`) implement conditional import fallbacks (`try: import torch except ImportError`). They import and execute non-PyTorch tests cleanly without requiring immediate package installation.

### Git Protections (`.gitignore`)
- Datasets: `data/`, `datasets/`, `*.mp4`, `*.zip` are ignored.
- Model Artifacts: `models/weights/`, `models/checkpoints/`, `*.pt`, `*.pth`, `*.onnx` are ignored.
- Notebooks: `.ipynb_checkpoints/` is ignored.
- `git diff --check`: **Clean — 0 whitespace or formatting errors.**

### Python Syntax & Bytecode Compilation Check
- **Command:** `backend\.venv\Scripts\python.exe -m py_compile backend/app/ml/*.py backend/app/ml/models/*.py backend/app/ml/training/*.py scripts/validate_ml_dataset.py scripts/train_gru.py scripts/evaluate_gru.py backend/tests/test_ml_components.py`
- **Result:** **13/13 modules compiled cleanly (Exit Code 0).**

---

## 6. Preprocessing Task Status (`task-2731`)

- **Current Status:** **RUNNING**
- **Progress:** **214 / 455 tensors generated** (~47% complete).
- **Integrity:** Process is running smoothly in the background; no partially written files are being accessed.

---

## 7. Readiness Verdict & Next Actions

| Gate | Status | Finding |
|---|---|---|
| Architecture & Math | ✅ PASS | Input dim 229, packed sequence handling, endpoint extraction verified. |
| Mask Decoupling | ✅ PASS | Sequence padding mask and 4 landmark indicators strictly decoupled. |
| Partitioning & Isolation | ✅ PASS | Signer 7 isolated to test; Signers 1–5 in train; Signer 6 in val. |
| Scaffolding Unit Tests | ✅ PASS | 11/11 ML tests pass. |
| Preprocessing Regressions | ✅ PASS | 44/44 preprocessing tests pass. |
| Ready for Dataset Validation | ✅ APPROVED | Awaiting completion of background task `task-2731`. |

**Next Safe Step:**
Allow `task-2731` to complete all 455 tensors. Upon completion, execute `backend\.venv\Scripts\python.exe scripts/validate_ml_dataset.py` to produce the final dataset validation report.
