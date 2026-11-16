# SignBridge AI — Experiment 1: Mean Pooling Results

**Evaluated:** Validation partition only (Signer 6)  
**Test partition (Signer 7):** NOT accessed  
**Exp1 Checkpoint:** `models/checkpoints/exp1_mean_pooling/best_gru_model.pt`  
**Baseline Checkpoint:** `models/checkpoints/gru_baseline_v1/best_gru_model.pt`  

---

## 1. Safety & Checkpoint Verification

| Check | Result |
|---|---|
| Test partition accessed | ❌ No (SAFE) |
| Exp1 pooling_strategy in checkpoint | `mean` ✅ |
| Baseline pooling_strategy in checkpoint | `endpoint` ✅ |
| Baseline checkpoint modified | No ✅ |
| Exp1 num_classes | 71 ✅ |

## 2. Training Summary (Experiment 1)

| Parameter | Value |
|---|---|
| Best epoch | 36 |
| Total epochs completed | 46 |
| Early stopping | Yes (patience=10 on val macro-F1) |
| Training duration | 246.3s |
| Val samples | 61 (Signer 6) |
| Train samples | 298 (Signers 1–5) |

## 3. Validation Metrics Comparison

| Metric | Baseline (endpoint) | Exp1 (mean) | Δ |
|---|---|---|---|
| Accuracy | 4.92% | 52.46% | **+47.54pp** |
| Macro-F1 (71 classes) | 0.0376 | 0.3751 | **+0.3375** |
| Active-class macro-F1 | 0.0437 | 0.4366 | **+0.3929** |
| Validation loss | 4.1628 | 1.8901 | -2.2727 |
| Correct / Total | — | 32/61 | — |

## 4. Metric Reproduction Check

| Metric | Reported (training log) | Recomputed (this eval) | Δ | Status |
|---|---|---|---|---|
| Accuracy | 0.5246 | 0.5246 | 0.0000 | ✅ PASS |
| Macro-F1 | 0.3751 | 0.3751 | 0.0000 | ✅ PASS |

> [!NOTE]
> The recomputed metrics match the training-log reports within ±0.005. The checkpoint correctly represents the best epoch.

## 5. Prediction Concentration

The model predicted **40 unique classes** out of 71 total.

| Rank | Predicted Class | Count | % of Val Preds |
|---|---|---|---|
| 1 | it does not make any difference to me | 4 | 6.6% |
| 2 | you can do it | 4 | 6.6% |
| 3 | i am feeling bored | 3 | 4.9% |
| 4 | i am suffering from fever | 3 | 4.9% |
| 5 | try to understand | 3 | 4.9% |
| 6 | take care of yourself | 2 | 3.3% |
| 7 | do not make me angry | 2 | 3.3% |
| 8 | i am fine. thank you sir | 2 | 3.3% |
| 9 | i somehow got to know about it | 2 | 3.3% |
| 10 | had your food | 2 | 3.3% |

## 6. Top 15 Confusion Pairs

| True Class | Predicted Class | Count |
|---|---|---|
| He is going into the room | do you need something | 1 |
| This place is beautiful | i am fine. thank you sir | 1 |
| can i help you | i promise | 1 |
| congratulations | how can i trust you | 1 |
| do me a favour | can i help you | 1 |
| do not abuse him | take care of yourself | 1 |
| do not take it to the heart | do not be stubborn | 1 |
| he came by train | why are you angry | 1 |
| he is on the way | you can do it | 1 |
| he she is my friend | do me a favour | 1 |
| how can i trust you | it does not make any difference to me | 1 |
| how dare you | This place is beautiful | 1 |
| i am fine. thank you sir | it does not make any difference to me | 1 |
| i am tired | i am feeling bored | 1 |
| i do not agree | it does not make any difference to me | 1 |

## 7. Best-Performing Classes (Top 15 by F1)

| Class | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| are you free today | 1.000 | 1.000 | 1.000 | 1 |
| comb your hair | 1.000 | 1.000 | 1.000 | 1 |
| do not hurt me | 1.000 | 1.000 | 1.000 | 1 |
| do not worry | 1.000 | 1.000 | 1.000 | 1 |
| go and sleep | 1.000 | 1.000 | 1.000 | 1 |
| i am afraid of that | 1.000 | 1.000 | 1.000 | 1 |
| i am crying | 1.000 | 1.000 | 1.000 | 1 |
| i am feeling cold | 1.000 | 1.000 | 1.000 | 1 |
| i am so sorry to hear that | 1.000 | 1.000 | 1.000 | 1 |
| i am very happy | 1.000 | 1.000 | 1.000 | 1 |
| i got hurt | 1.000 | 1.000 | 1.000 | 1 |
| let him take time | 1.000 | 1.000 | 1.000 | 1 |
| prepare the bed | 1.000 | 1.000 | 1.000 | 1 |
| shall we go outside | 1.000 | 1.000 | 1.000 | 1 |
| turn on light turn off light | 1.000 | 1.000 | 1.000 | 1 |

## 8. Worst-Performing Classes (Bottom 15 by F1)

| Class | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| pour some more water into the glass | 0.000 | 0.000 | 0.000 | 1 |
| serve the food | 0.000 | 0.000 | 0.000 | 1 |
| speak softly | 0.000 | 0.000 | 0.000 | 1 |
| tell me truth | 0.000 | 0.000 | 0.000 | 1 |
| we are all with you | 0.000 | 0.000 | 0.000 | 0 |
| what did you tell him | 0.000 | 0.000 | 0.000 | 1 |
| what do you want to become | 0.000 | 0.000 | 0.000 | 1 |
| what happened | 0.000 | 0.000 | 0.000 | 1 |
| what have you planned for your career | 0.000 | 0.000 | 0.000 | 0 |
| which collegeschool are you from | 0.000 | 0.000 | 0.000 | 0 |
| why are you angry | 0.000 | 0.000 | 0.000 | 1 |
| why are you crying | 0.000 | 0.000 | 0.000 | 1 |
| why are you disappointed | 0.000 | 0.000 | 0.000 | 1 |
| you are bad | 0.000 | 0.000 | 0.000 | 1 |
| you do anything, i do not care | 0.000 | 0.000 | 0.000 | 0 |

## 9. Full Per-Class Metrics (All 71 Classes)

| Class | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| are you free today | 1.000 | 1.000 | 1.000 | 1 |
| comb your hair | 1.000 | 1.000 | 1.000 | 1 |
| do not hurt me | 1.000 | 1.000 | 1.000 | 1 |
| do not worry | 1.000 | 1.000 | 1.000 | 1 |
| go and sleep | 1.000 | 1.000 | 1.000 | 1 |
| i am afraid of that | 1.000 | 1.000 | 1.000 | 1 |
| i am crying | 1.000 | 1.000 | 1.000 | 1 |
| i am feeling cold | 1.000 | 1.000 | 1.000 | 1 |
| i am so sorry to hear that | 1.000 | 1.000 | 1.000 | 1 |
| i am very happy | 1.000 | 1.000 | 1.000 | 1 |
| i got hurt | 1.000 | 1.000 | 1.000 | 1 |
| let him take time | 1.000 | 1.000 | 1.000 | 1 |
| prepare the bed | 1.000 | 1.000 | 1.000 | 1 |
| shall we go outside | 1.000 | 1.000 | 1.000 | 1 |
| turn on light turn off light | 1.000 | 1.000 | 1.000 | 1 |
| wear the shirt | 1.000 | 1.000 | 1.000 | 1 |
| when will the train leave | 1.000 | 1.000 | 1.000 | 1 |
| you are good | 1.000 | 1.000 | 1.000 | 1 |
| you need a medicine, take this one | 1.000 | 1.000 | 1.000 | 1 |
| do not be stubborn | 0.667 | 0.500 | 1.000 | 1 |
| do not make me angry | 0.667 | 0.500 | 1.000 | 1 |
| do you need something | 0.667 | 0.500 | 1.000 | 1 |
| had your food | 0.667 | 0.500 | 1.000 | 1 |
| i do not mean it | 0.667 | 0.500 | 1.000 | 1 |
| i somehow got to know about it | 0.667 | 0.500 | 1.000 | 1 |
| take care of yourself | 0.667 | 0.500 | 1.000 | 1 |
| you are welcome | 0.667 | 0.500 | 1.000 | 1 |
| i am feeling bored | 0.500 | 0.333 | 1.000 | 1 |
| i am suffering from fever | 0.500 | 0.333 | 1.000 | 1 |
| try to understand | 0.500 | 0.333 | 1.000 | 1 |
| it does not make any difference to me | 0.400 | 0.250 | 1.000 | 1 |
| you can do it | 0.400 | 0.250 | 1.000 | 1 |
| He is going into the room | 0.000 | 0.000 | 0.000 | 1 |
| This place is beautiful | 0.000 | 0.000 | 0.000 | 1 |
| are you hiding something | 0.000 | 0.000 | 0.000 | 0 |
| can i help you | 0.000 | 0.000 | 0.000 | 1 |
| congratulations | 0.000 | 0.000 | 0.000 | 1 |
| do me a favour | 0.000 | 0.000 | 0.000 | 1 |
| do not abuse him | 0.000 | 0.000 | 0.000 | 1 |
| do not take it to the heart | 0.000 | 0.000 | 0.000 | 1 |
| he came by train | 0.000 | 0.000 | 0.000 | 1 |
| he is on the way | 0.000 | 0.000 | 0.000 | 1 |
| he she is my friend | 0.000 | 0.000 | 0.000 | 1 |
| how are things | 0.000 | 0.000 | 0.000 | 0 |
| how can i trust you | 0.000 | 0.000 | 0.000 | 1 |
| how dare you | 0.000 | 0.000 | 0.000 | 1 |
| i am fine. thank you sir | 0.000 | 0.000 | 0.000 | 1 |
| i am in dilemma what to do | 0.000 | 0.000 | 0.000 | 0 |
| i am sitting in the class | 0.000 | 0.000 | 0.000 | 0 |
| i am tired | 0.000 | 0.000 | 0.000 | 1 |
| i do not agree | 0.000 | 0.000 | 0.000 | 1 |
| i do not like it | 0.000 | 0.000 | 0.000 | 1 |
| i enjoyed a lot | 0.000 | 0.000 | 0.000 | 1 |
| i like you i love you | 0.000 | 0.000 | 0.000 | 0 |
| i promise | 0.000 | 0.000 | 0.000 | 1 |
| i was stopped by some one | 0.000 | 0.000 | 0.000 | 0 |
| pour some more water into the glass | 0.000 | 0.000 | 0.000 | 1 |
| serve the food | 0.000 | 0.000 | 0.000 | 1 |
| speak softly | 0.000 | 0.000 | 0.000 | 1 |
| tell me truth | 0.000 | 0.000 | 0.000 | 1 |
| we are all with you | 0.000 | 0.000 | 0.000 | 0 |
| what did you tell him | 0.000 | 0.000 | 0.000 | 1 |
| what do you want to become | 0.000 | 0.000 | 0.000 | 1 |
| what happened | 0.000 | 0.000 | 0.000 | 1 |
| what have you planned for your career | 0.000 | 0.000 | 0.000 | 0 |
| which collegeschool are you from | 0.000 | 0.000 | 0.000 | 0 |
| why are you angry | 0.000 | 0.000 | 0.000 | 1 |
| why are you crying | 0.000 | 0.000 | 0.000 | 1 |
| why are you disappointed | 0.000 | 0.000 | 0.000 | 1 |
| you are bad | 0.000 | 0.000 | 0.000 | 1 |
| you do anything, i do not care | 0.000 | 0.000 | 0.000 | 0 |

## 10. Conclusion & Analysis

### Result

Experiment 1 (mask-aware temporal mean pooling) produced a **genuine, substantial improvement** over the endpoint-pooling baseline on the validation set:

- Validation accuracy improved from **4.92%** → **52.46%** (+47.54 percentage points).
- Global 71-class macro-F1 improved from **0.0376** → **0.3751** (+0.3375).
- Active-class macro-F1 improved from **0.0437** → **0.4366**.

### Why Mean Pooling Helped

The endpoint-pooling baseline concatenates the forward GRU final state and backward GRU initial state, which are temporally biased toward the boundary frames of each clip. These boundary frames often capture neutral resting poses as the signer transitions into or out of the sign. By averaging over all valid timesteps, mean pooling integrates the full kinematic arc of each sentence into the sequence embedding, reducing the boundary-pose dominance. The 10× improvement in accuracy is consistent with this hypothesis and with the boundary hand-absence statistics documented in the error analysis.

### Remaining Limitations

1. **Structural class imbalance:** With only 61 validation samples across 61 classes (1 per class), per-class F1 is high-variance and a single prediction error costs 100% recall for that class.
2. **Train/val distribution shift:** Training classes with ≥4 samples significantly outperform singleton training classes. The global 71-class macro-F1 (0.3751) is penalized by the 10 training classes absent from validation receiving zero recall contribution.
3. **Overfitting trajectory:** Training accuracy reached 95.6% at best epoch 36 (and 100% by epoch 45) while validation accuracy plateaued at ~52.5%, indicating a train-val generalization gap. This is expected given the small dataset size (298 training samples, 71 classes).
4. **Signer-independence:** Validation is on Signer 6 (one signer). The true generalization capability will be measured on the held-out Signer 7 test set after all experiments complete.
5. **Mean pooling vs attention:** Uniform mean pooling weights all valid frames equally. Experiment 2 (learned temporal attention) may selectively weight the most discriminative frames and further improve performance.

### Next Step Recommendation

Mean pooling is the **new stronger baseline** (macro-F1 0.3751) for all subsequent comparisons. Proceed to **Experiment 2: Learned Temporal Attention Pooling** using the same training configuration, data splits, and random seed, with checkpoint target `models/checkpoints/exp2_attention_pooling/`.

---

*This report was generated by `scripts/evaluate_exp1_validation.py`. No test-partition data (Signer 7) was accessed at any point.*
