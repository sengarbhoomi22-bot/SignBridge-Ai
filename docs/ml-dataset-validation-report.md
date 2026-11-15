# SignBridge AI: Pre-Flight ML Dataset Validation Report

**Overall Status:** `PASS`  
**Date:** November 2026  
**Dataset Directory:** `data\processed\isl_csltr_tensors_v2`  

## 1. Split Partition Summary

| Split | Total Files | Usable for ML | QC Failures | Too Short (<10f) | Signers Present | Categories |
|---|---|---|---|---|---|---|
| **train** | 317 | 298 | 19 | 0 | 1, 2, 3, 4, 5 | 71 |
| **validation** | 66 | 61 | 5 | 0 | 6 | 66 |
| **test** | 68 | 65 | 3 | 0 | 7 | 68 |
| **quarantine** | 2 | 0 | 0 | 0 | None | 2 |

## 2. Integrity & Scientific Verification Checks

- **no_duplicate_clip_ids:** `PASS`
- **signer_7_strictly_isolated:** `PASS`
- **quarantine_excluded_from_train:** `PASS`
- **zero_fabrication_confirmed:** `PASS`
- **no_nan_or_inf:** `PASS`

- **Total Unique Categories Across Splits:** 73
- **Train-Validation Category Overlap:** 66
- **Train-Test Category Overlap:** 68

## 5. Readiness Verdict

✅ **APPROVED FOR MODEL TRAINING:** All splits strictly conform to Candidate B topology and signer isolation rules.
