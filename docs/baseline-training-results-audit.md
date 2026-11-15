# SignBridge AI — Step 3.5B: GRU Baseline Training Results Audit

**Date:** November 2026  
**Auditor:** Automated Engineering Audit (SignBridge ML Architecture)  
**Run Target:** SignBridge GRU Baseline v1 (`models/checkpoints/gru_baseline_v1/`)  
**Hardware:** Intel Core Ultra 5 225H CPU (`torch.device("cpu")`)  
**Verdict (see §11):** ⚠️ **BASELINE TRAINING NEEDS IMPROVEMENT**

---

## 1. Executive Summary

The first full baseline training run of the SignBridge Bi-GRU sequence classifier (`models/checkpoints/gru_baseline_v1/`) was audited.

The training pipeline completed **26 epochs** on the CPU before the early stopping mechanism intervened. Early stopping halted training after 10 consecutive non-improving validation epochs following the best epoch (**Epoch 16**).

All numerical routines functioned cleanly without numerical instability:
- Zero NaN or Infinite values were observed across all losses, logits, and parameter updates.
- The best checkpoint (`best_gru_model.pt`, 3.28 MB) was successfully serialized and verified by loading into `SignBridgeInferenceEngine`.
- All 71 training classes were handled consistently across the model output layer, loss function, and `LabelEncoder`.

However, the generalization gap reveals **substantial overfitting** to the small training partition ($N=298$, $\approx 4.2$ samples/class), with training accuracy climbing to **68.46%** while validation accuracy on the unseen volunteer signer (Signer 6) reached a peak of **4.92%** (Macro-F1: **0.0376**).

---

## 2. Key Metrics Summary

| Metric | Measured Value | Notes |
|---|---|---|
| **Total Epochs Run** | **26 epochs** | Early stopping halted at epoch 26 (patience = 10) |
| **Best Checkpoint Epoch** | **Epoch 16** | Saved based on peak validation macro-F1 |
| **Initial Training Loss (Epoch 1)** | `4.4074` | Initial training accuracy: `0.0067` (0.67%) |
| **Final Training Loss (Epoch 26)** | `1.6459` | Final training accuracy: `0.6846` (68.46%) |
| **Minimum Validation Loss** | `4.0877` | Observed at Epoch 18 (Val Loss at best Epoch 16: `4.1628`) |
| **Best Validation Accuracy** | `0.0492` (**4.92%**) | 3 / 61 validation clips correctly classified |
| **Best Validation Macro-F1** | `0.0376` (**3.76%**) | Global 71-class macro-average |
| **Random Chance Benchmark** | `0.0141` (**1.41%**) | $1 / 71$ uniform random probability |
| **Numerical Sanity** | **100% Finite** | Zero NaN or Inf values in losses, logits, or weights |
| **Checkpoint Integrity** | **Verified** | Successfully reloaded into `SignBridgeInferenceEngine` |

---

## 3. Epoch-by-Epoch Training & Validation Progression

The full 26-epoch history extracted from `models/checkpoints/gru_baseline_v1/training_history.json`:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro-F1 | Status / Notes |
|---|---|---|---|---|---|---|
| **1** | 4.4074 | 0.67% | 4.2672 | 1.64% | 0.0005 | Initial epoch |
| **2** | 4.3111 | 0.34% | 4.2597 | 1.64% | 0.0005 | — |
| **3** | 4.2616 | 2.01% | 4.2454 | 1.64% | 0.0005 | — |
| **4** | 4.2467 | 1.68% | 4.2536 | 1.64% | 0.0006 | Marginal F1 gain |
| **5** | 4.2193 | 2.01% | 4.2768 | 0.00% | 0.0000 | Zero correct |
| **6** | 4.1854 | 3.69% | 4.2725 | 3.28% | 0.0106 | 2 correct |
| **7** | 4.1053 | 4.03% | 4.2761 | 3.28% | 0.0028 | — |
| **8** | 4.0087 | 5.70% | 4.3020 | 0.00% | 0.0000 | Val loss spike |
| **9** | 3.8691 | 8.72% | 4.2782 | 3.28% | 0.0109 | — |
| **10** | 3.7494 | 12.75% | 4.2870 | 3.28% | 0.0059 | — |
| **11** | 3.6234 | 9.73% | 4.1950 | 3.28% | 0.0087 | Val loss dropping |
| **12** | 3.5001 | 16.11% | 4.2102 | 1.64% | 0.0035 | — |
| **13** | 3.3363 | 22.48% | 4.1350 | **4.92%** | 0.0169 | 3 correct |
| **14** | 3.2427 | 23.15% | 4.1040 | 1.64% | 0.0070 | — |
| **15** | 3.0725 | 27.85% | 4.0910 | 1.64% | 0.0047 | — |
| **16** | **2.9535** | **27.18%** | **4.1628** | **4.92%** | **0.0376** | 🏆 **BEST CHECKPOINT (Patience Reset)** |
| **17** | 2.8020 | 36.91% | 4.0985 | 4.92% | 0.0228 | Patience: 1/10 |
| **18** | 2.7060 | 35.57% | **4.0877** | 4.92% | 0.0235 | Patience: 2/10 (Lowest Val Loss) |
| **19** | 2.5857 | 37.92% | 4.1509 | 3.28% | 0.0235 | Patience: 3/10 |
| **20** | 2.4588 | 44.63% | 4.1426 | 1.64% | 0.0141 | Patience: 4/10 |
| **21** | 2.2980 | 48.99% | 4.1909 | 4.92% | 0.0313 | Patience: 5/10 |
| **22** | 2.1486 | 54.36% | 4.2366 | 3.28% | 0.0181 | Patience: 6/10 |
| **23** | 1.9863 | 62.08% | 4.2007 | 4.92% | 0.0270 | Patience: 7/10 |
| **24** | 1.9025 | 59.40% | 4.1909 | 3.28% | 0.0188 | Patience: 8/10 |
| **25** | 1.7554 | 63.09% | 4.1825 | 3.28% | 0.0141 | Patience: 9/10 |
| **26** | 1.6459 | 68.46% | 4.1903 | 1.64% | 0.0040 | 🛑 **EARLY STOPPING HALT (Patience: 10/10)** |

---

## 4. Underfitting vs. Overfitting Diagnosis

The metrics provide unequivocal evidence of **high overfitting**:

1. **Divergence of Losses:**
   - Training loss fell steadily from `4.4074` to `1.6459` ($\Delta = -2.7615$, a $62.7\%$ decrease).
   - Validation loss initially dropped from `4.2672` to `4.0877` (Epoch 18), but then plateaued and began oscillating around `4.18 - 4.24`.
2. **Divergence of Accuracy:**
   - Training accuracy increased from `0.67%` to `68.46%` ($\approx 100\times$ increase).
   - Validation accuracy peaked at `4.92%` (Epochs 13, 16, 17, 18, 21, 23) and never exceeded 3 correct predictions out of 61.
3. **Generalization Gap:**
   - At Epoch 26, the gap between training accuracy ($68.46\%$) and validation accuracy ($1.64\%$) reached **$66.82\%$**.
   - The model has memorized the idiosyncratic signing dynamics of Signers 1–5 in the training set rather than learning invariant semantic sign representations that transfer cleanly to volunteer Signer 6.

---

## 5. Per-Class Performance on Validation Set (Best Model, Epoch 16)

Out of 61 evaluated validation sentences (Signer 6), exactly **3 sentences** were correctly classified by the Epoch 16 checkpoint:

| Category | True Positives | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| `go and sleep` | 1 | 1 | 1.0000 | 1.0000 | **1.0000** |
| `i am crying` | 1 | 1 | 1.0000 | 1.0000 | **1.0000** |
| `do not take it to the heart` | 1 | 1 | 0.5000 | 1.0000 | **0.6667** |
| *All other 58 validation categories* | 0 | 1 | 0.0000 | 0.0000 | 0.0000 |

### False Positive Concentration
Certain categories acted as "attractors" for false positives:
- `do not make me angry` received **7 false positives** (predicted 7 times when it was not the true class).
- `can i help you` received **4 false positives**.
- `do not be stubborn` received **3 false positives**.
- `are you free today` received **2 false positives**.

---

## 6. Numerical Integrity and Error Analysis

- **NaN / Infinite Values:** Zero detected in any epoch for train loss, train accuracy, val loss, val accuracy, or val macro-F1.
- **Gradient Stability:** Gradient clipping (`grad_clip_norm = 1.0`) prevented gradient explosions.
- **Loss Progression:** CrossEntropyLoss on raw logits transitioned smoothly from near the theoretical random initialization loss ($-\ln(1/71) \approx 4.2626$) down to $1.6459$.
- **Corrupted Samples / Crashes:** Zero pipeline crashes or corrupted archive errors encountered.

---

## 7. Checkpoint Verification

The saved checkpoint was loaded and tested:
- **Path:** `models/checkpoints/gru_baseline_v1/best_gru_model.pt`
- **File Size:** `3,283,565 bytes` ($\approx 3.13 \text{ MB}$)
- **Internal Payload Keys:** `['epoch', 'model_state_dict', 'optimizer_state_dict', 'config', 'model_config', 'best_val_metrics', 'num_classes', 'classes']`
- **Associated Metadata Files:**
  - `label_mapping.json` ($6,774 \text{ bytes}$, maps all 71 classes deterministically)
  - `training_config.json` ($557 \text{ bytes}$)
  - `training_history.json` ($4,362 \text{ bytes}$, 26 epochs)
- **Inference Smoke Check:** Successfully instantiated via `SignBridgeInferenceEngine.load_from_checkpoint()`. Executed forward pass on test tensor and correctly returned `status: LOW_CONFIDENCE` (confidence: $0.0489$) with all 71 vocabulary classes recognized.

---

## 8. Root Cause Analysis of Low Validation Performance

The low validation accuracy ($4.92\%$) and macro-F1 ($0.0376$) on Signer 6 are expected consequences of four fundamental factors:

1. **Extreme Class Scarcity ($N=298$ across 71 categories):**
   - With an average of only **4.2 training samples per class**, the model has seen each sentence signed by at most 4 or 5 different individuals.
   - Deep recurrent networks require sufficient variance across multiple signers to separate lexical sign features from signer-specific biomechanical artifacts (speed, hand size, resting pose).
2. **Cross-Signer Domain Shift:**
   - Training contains Signers 1–5 (native and volunteer), while Validation is strictly Signer 6 (volunteer).
   - Without signer-adaptive normalization or data augmentation, the Bi-GRU overfits to signer-specific motion trajectories.
3. **Absence of Pretraining or Temporal Augmentation:**
   - The model was trained from scratch with randomly initialized weights.
   - No temporal jittering, speed perturbation, or spatial scaling augmentations were applied to the landmark sequences.
4. **Vocabulary Zero-Support Ceiling:**
   - As established in Step 3.3, 10 training categories do not appear in the validation set. Their F1 contribution is permanently 0.0, capping validation Macro-F1 at $85.92\%$.

---

## 9. Comparison Against Baselines

| Baseline / Model | Validation Accuracy | Validation Macro-F1 | Notes |
|---|---|---|---|
| **Uniform Random Guess** | `1.41%` ($1/71$) | `0.0141` | Theoretical expectation |
| **Majority Class Reference** | `1.64%` ($1/61$) | `0.0005` | Predicting mode training class for all samples |
| **SignBridge Bi-GRU (Epoch 16)** | **`4.92%`** ($3/61$) | **`0.0376`** | **$> 3\times$ higher than random guess** |

The Bi-GRU demonstrates genuine signal learning (performing $> 3\times$ better than random chance on an unseen signer for a 71-class sentence problem), but remains far below acceptable interactive recognition thresholds.

---

## 10. Recommended Remediation Strategies for Future Iterations

1. **Data Augmentation (Immediate Priority):**
   - Temporal stretching / compression ($0.8\times \dots 1.2\times$ speed variations).
   - Coordinate jittering ($\mathcal{N}(0, \sigma^2)$ Gaussian noise on landmarks).
   - Spatial scaling and rotation perturbations around shoulder anchors.
2. **Architecture Enhancements:**
   - Spatial-Temporal Graph Convolutional Networks (ST-GCN) or Transformer encoders with multi-head self-attention.
   - Pre-training on isolated sign glosses before sentence sequence modeling.
3. **Regularization:**
   - Increase dropout rate to $0.4 - 0.5$.
   - Apply Label Smoothing ($\alpha = 0.1$) to soften target distributions and discourage overconfident attractor predictions.
   - Introduce weight decay of $10^{-3}$ to penalize overparameterized recurrent transition matrices.

---

## 11. Final Verdict

```
======================================================================
FINAL VERDICT: BASELINE TRAINING NEEDS IMPROVEMENT
======================================================================
```

**Justification:**
1. **Pipeline Execution: SUCCESSFUL.** The end-to-end training harness, autograd engine, early stopping mechanism, metric computation, and isolated checkpointing functioned flawlessly on CPU without any numerical or memory issues.
2. **Model Performance: NEEDS IMPROVEMENT.** Validation accuracy of $4.92\%$ and Macro-F1 of $0.0376$ across 71 categories confirm severe cross-signer overfitting. While outperforming random chance ($1.41\%$), substantial architectural, regularization, and data augmentation improvements are required before production deployment.
