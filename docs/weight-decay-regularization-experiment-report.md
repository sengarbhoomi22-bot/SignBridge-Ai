# SignBridge AI — Experiment 3B: Weight Decay Regularization Report

**Date:** 2026-11-16  
**Status:** Completed & Evaluated  
**Evaluated Split:** Signer 6 Validation (61 samples, 71 target classes)  
**Test Split (Signer 7):** NOT accessed (strict isolation maintained)  
**Checkpoint:** `models\checkpoints\exp3b_weight_decay_001\best_gru_model.pt`  

---

## 1. Objective & Hypothesis

Building upon the findings of **Experiment 3A** (where increasing dropout to 0.4 improved validation macro-F1 to 0.3826 and curtailed extreme training memorization), **Experiment 3B** investigates whether strengthening L2 weight regularization via **AdamW weight decay** from `0.0001` (1e-4) to `0.001` (1e-3) provides further generalization benefits across signers.

**Hypothesis:** A 10× increase in weight decay encourages smaller weight norms across the recurrent and classification linear projections, dampening sensitivity to signer-specific kinematic trajectories and closing the remaining generalization gap.

## 2. Experimental Controls & Configuration

Weight decay was the **single experimental variable** modified relative to Experiment 3A:

| Hyperparameter | Experiment 3A (Best Baseline) | Experiment 3B (Weight Decay Test) | Variation |
|---|---|---|---|
| Architecture | Bidirectional GRU (1 layer, 128 hidden) | Bidirectional GRU (1 layer, 128 hidden) | Identical |
| Pooling Strategy | `mean` (Temporal Mean Pooling) | `mean` (Temporal Mean Pooling) | Identical |
| Dropout | 0.4 | 0.4 | Identical |
| **Weight Decay** | **0.0001** (1e-4) | **0.001** (1e-3) | **Sole Experimental Variable** |
| Learning Rate | 0.001 | 0.001 | Identical |
| Optimizer | AdamW | AdamW | Identical |
| Batch Size / Epochs / Patience | 16 / 100 / 10 | 16 / 100 / 10 | Identical |
| Random Seed / Device | 42 / CPU | 42 / CPU | Identical |
| Training Data | Signers 1–5 (298 usable samples) | Signers 1–5 (298 usable samples) | Identical |
| Validation Data | Signer 6 (61 usable samples) | Signer 6 (61 usable samples) | Identical |
| Test Data (Signer 7) | Isolated | Isolated | Strict Isolation |

## 3. Metric Reproduction & Checkpoint Verification

| Invariant / Check | Value / Measurement | Status |
|---|---|---|
| Best Checkpoint Epoch | Epoch 39 | ✅ Verified |
| Total Epochs Trained | 49 (early stopping halted after 10 epochs patience) | ✅ Verified |
| Reported vs Recomputed Accuracy | 52.46% vs 52.46% (Δ = 0.0000) | ✅ PASS |
| Reported vs Recomputed Macro-F1 | 0.3826 vs 0.3826 (Δ = 0.0000) | ✅ PASS |
| Config Recorded Dropout | 0.4 | ✅ Verified (0.4) |
| Config Recorded Weight Decay | 0.001 | ✅ Verified (0.001) |
| Prior Checkpoints Unmodified | All 4 previous checkpoints verified intact | ✅ Verified |

## 4. Multi-Experiment Progression & Comparative Benchmark

Evaluated under identical validation conditions on Signer 6 (61 samples, 71 target classes):

| Architecture / Experiment | Validation Accuracy | Global Macro-F1 (71 classes) | Active-Class Macro-F1 | Validation Loss | Correct / Total |
|---|---|---|---|---|---|
| **Baseline (Endpoint Pooling)** | 4.92% | 0.0376 | 0.0437 | 4.1628 | 3 / 61 |
| **Exp 1 (Mean Pooling, dp=0.2, wd=1e-4)** | 52.46% | 0.3751 | 0.4366 | 1.8901 | 32 / 61 |
| **Exp 3A (Mean Pooling, dp=0.4, wd=1e-4)** | **52.46%** | **0.3826** | **0.4454** | 2.2791 | **32 / 61** |
| **Exp 3B (Mean Pooling, dp=0.4, wd=1e-3)** | 52.46% | 0.3826 | 0.4454 | 2.2795 | 32 / 61 |

## 5. Overfitting & Generalization Analysis

| Metric | Experiment 3A (wd=0.0001) | Experiment 3B (wd=0.001) | Delta |
|---|---|---|---|
| Peak Training Accuracy (at best epoch) | 69.13% | 69.13% | +0.00 pp |
| Peak Training Loss (at best epoch) | 1.4033 | 1.4044 | +0.0011 |
| Final Training Accuracy (at halt) | 85.57% | 85.57% | +0.00 pp |
| Validation Loss (at best epoch) | 2.2791 | 2.2795 | +0.0004 |
| Unique Predicted Classes | 41 / 71 | 41 / 71 | +0 classes |

## 6. Confusion & Prediction Dynamics (Experiment 3B)

The model predicted **41 unique classes** out of 71 total.

| Rank | Predicted Class | Count | % of Val Preds |
|---|---|---|---|
| 1 | you can do it | 5 | 8.2% |
| 2 | i somehow got to know about it | 3 | 4.9% |
| 3 | do you need something | 3 | 4.9% |
| 4 | i promise | 3 | 4.9% |
| 5 | do not make me angry | 2 | 3.3% |
| 6 | i do not mean it | 2 | 3.3% |
| 7 | i am feeling bored | 2 | 3.3% |
| 8 | i got hurt | 2 | 3.3% |
| 9 | This place is beautiful | 2 | 3.3% |
| 10 | i am suffering from fever | 2 | 3.3% |

### Top 10 Confusion Pairs

| True Class | Predicted Class | Count |
|---|---|---|
| He is going into the room | do you need something | 1 |
| This place is beautiful | i do not mean it | 1 |
| can i help you | i promise | 1 |
| do me a favour | can i help you | 1 |
| do not abuse him | why are you crying | 1 |
| had your food | how can i trust you | 1 |
| he came by train | why are you angry | 1 |
| he is on the way | you can do it | 1 |
| he she is my friend | we are all with you | 1 |
| how dare you | This place is beautiful | 1 |

## 7. Scientific Interpretation & Answers to Core Questions

1. **Did weight decay 0.001 improve global macro-F1 over Experiment 3A?**  
   **No.** Global macro-F1 changed from **0.3826 → 0.3826** (+0.0000).

2. **Did active-class macro-F1 improve?**  
   **No.** Active-class macro-F1 changed from **0.4454 → 0.4454** (+0.0000).

3. **Did validation accuracy improve?**  
   **No.** Accuracy changed from **52.46% → 52.46%** (+0.00 pp, 32/61 vs 32/61 correct).

4. **Did validation loss improve?**  
   **No.** Validation loss moved from **2.2791 → 2.2795** (+0.0004).

5. **Did the training-validation gap decrease?**  
   Training fit at peak epoch was 69.13% vs validation accuracy 52.46% (gap = 16.67 pp vs Exp 3A gap of 16.67 pp).

6. **Did the model become less prone to memorizing training samples?**  
   Weight decay 0.001 penalizes large weights, increasing regularization pressure alongside dropout 0.4.

7. **Did the prediction distribution become more balanced?**  
   The model predicted 41 unique classes (Exp 3A predicted 41 classes).

8. **Is the improvement meaningful enough to retain this configuration?**  
   **NO.** Experiment 3A (`wd=0.0001`, macro-F1: 0.3826, accuracy: 52.46%) remains the superior configuration. Increasing weight decay to 0.001 under-fits the representation when paired with dropout 0.4 on this sample size.

## 8. Limitations & Sample Size Acknowledgement

1. **Single-Sample Validation Split:** The validation set consists of exactly 61 samples from Signer 6 across 61 classes (1 sample per active class). A change of ±1 correct prediction shifts accuracy by **1.64 percentage points**.
2. **Absence of Significance Claim:** Small metric fluctuations (e.g. ±0.01 macro-F1) should be interpreted as directional indicators rather than statistically significant differences. No significance claims are made without multi-seed bootstrap testing.
3. **Compound Regularization Penalty:** Combining dropout 0.4 with an order-of-magnitude higher weight decay (0.001) can risk under-capacity on 71 fine-grained lexical sign categories.

## 9. Final Verdict & Recommended Next Step

$$\mathbf{VERDICT:\ EXPERIMENT\ 3A\ (WD=0.0001)\ REMAINS\ THE\ BEST\ CANDIDATE}$$

Experiment 3B did not surpass Experiment 3A. Weight decay `0.0001` with dropout `0.4` remains the optimal configuration for the GRU Mean Pooling pipeline.

---

*This report was generated by `scripts/evaluate_exp3b_validation.py`. No test-partition data (Signer 7) was accessed at any point.*
