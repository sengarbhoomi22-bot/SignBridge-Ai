# SignBridge AI — Step 3.5C: GRU Baseline Diagnostic Error Analysis

**Date:** November 2026  
**Auditor:** Automated Engineering Audit (SignBridge ML Architecture)  
**Analyzed Run:** SignBridge GRU Baseline v1 (`models/checkpoints/gru_baseline_v1/`)  
**Analyzed Checkpoint:** Best Validation Checkpoint — Epoch 16 (`best_gru_model.pt`)  
**Data Evaluated:** Training Partition ($N=298$, Signers 1–5), Validation Partition ($N=61$, Signer 6)  
**Deliverable Paths:**
- Detailed Analysis: `docs/gru-baseline-error-analysis.md`
- Machine-Readable Diagnostics: `reports/gru_baseline_error_analysis.json`

---

## 1. Executive Summary

This diagnostic investigation provides a root-cause breakdown of why the baseline Bi-GRU achieved **$68.46\%$ training accuracy** but only **$4.92\%$ validation accuracy** ($0.0376$ macro-F1) on the held-out volunteer signer (Signer 6).

### Primary Findings

1. **Prediction Collapse into 8 "Attractor" Classes:**
   More than half ($55.7\%$, 34 of 61) of all validation predictions collapsed into just 8 classes, led by `do not make me angry` (7 predictions, 0 correct) and `i do not mean it` (6 predictions, 0 correct).
2. **Substantial Cross-Signer Biomechanical Domain Shift:**
   Signer 6 (volunteer) signs with a radically different spatial envelope than native Signers 1–5:
   - Left-hand coordinate mean is **$+105\%$ higher** ($0.4100$ vs $0.1993$) with **$+59\%$ greater spatial variance** ($0.8992$ vs $0.5656$).
   - Signer 6 is significantly more right-hand dominant ($64.9\%$ right hand presence vs $54.9\%$ in training).
   - Signer 6 signs on average **$\approx 9$ frames slower** ($98.25$ vs $89.85$ frames).
3. **Endpoint Pooling Architectural Limitation:**
   The Bi-GRU model aggregates temporal features by concatenating the forward hidden state at frame $T-1$ with the backward hidden state at frame $0$. Empirical analysis shows that **hands are absent in $75\% - 90\%$ of frame 0 clips** and **$55\% - 69\%$ of frame $T-1$ clips**. The recurrent states at endpoints are dominated by neutral resting poses rather than dynamic sign gestures.
4. **Metric Implementation Sanity:**
   The reported global 71-class Macro-F1 ($0.0376$) mathematically reconciles with the active 61-class Macro-F1 ($0.0437$): $0.0437 \times \frac{61}{71} = 0.0376$. The 10 inactive classes contribute 0.0 to the global average as expected in closed-set classification.

---

## 2. Class Support and Validation Performance Breakdown

### 2.1 Highest-Performing Validation Classes

Only 3 out of 61 validation sentences were correctly classified by the Epoch 16 checkpoint:

| Sentence Category | Train Support | Val Support | True Positives | False Positives | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| `go and sleep` | 5 | 1 | 1 | 0 | 1.0000 | 1.0000 | **1.0000** |
| `i am crying` | 5 | 1 | 1 | 0 | 1.0000 | 1.0000 | **1.0000** |
| `do not take it to the heart` | 5 | 1 | 1 | 1 | 0.5000 | 1.0000 | **0.6667** |

*Notice: All 3 correctly predicted categories had maximum training support (5 samples across 5 signers).*

### 2.2 Lowest-Performing Validation Classes

The remaining **58 active validation categories** received an F1-score of **$0.0000$** ($\text{Recall} = 0.0$).

### 2.3 Training Sample Support Distribution vs Success Rate

| Training Support | Total Classes in Train | Classes Present in Val | Val Correct | Val Accuracy |
|---|---|---|---|---|
| **5 samples** | 41 classes | 39 classes | 3 | **7.69%** |
| **4 samples** | 13 classes | 11 classes | 0 | **0.00%** |
| **3 samples** | 9 classes | 6 classes | 0 | **0.00%** |
| **2 samples** | 6 classes | 4 classes | 0 | **0.00%** |
| **1 sample** | 2 classes | 1 class | 0 | **0.00%** |
| **Total** | **71 classes** | **61 classes** | **3** | **4.92%** |

No category with fewer than 5 training samples achieved a single correct prediction on Signer 6.

---

## 3. Confusion Matrix and Attractor Analysis

Because Signer 6 signs each sentence exactly once, every true class appears with $\text{support} = 1$. The errors manifest as prediction concentration into specific "attractor" classes.

### 3.1 Top Attractor Classes

| Attractor Class | Total Predicted | True Positives | False Positives | False Positive Precision |
|---|---|---|---|---|
| `do not make me angry` | **7** | 0 | 7 | 0.0% |
| `i do not mean it` | **6** | 0 | 6 | 0.0% |
| `why are you disappointed` | **4** | 0 | 4 | 0.0% |
| `try to understand` | **4** | 0 | 4 | 0.0% |
| `can i help you` | **4** | 0 | 4 | 0.0% |
| `wear the shirt` | **3** | 0 | 3 | 0.0% |
| `you are welcome` | **3** | 0 | 3 | 0.0% |
| `do not be stubborn` | **3** | 0 | 3 | 0.0% |
| **Total Across Top 8** | **34** | **0** | **34** | **0.0%** |

**Finding:** 8 classes account for **$55.7\%$** of all validation predictions.

### 3.2 Semantic / Biomechanical Clustering into Attractors

Inspecting which true labels mapped into the attractors reveals distinct motion clustering:

1. **`do not make me angry` (7 FPs):**
   - Attracts sentences of distress, warning, and physical ailment:
     `i am afraid of that`, `i am suffering from fever`, `i got hurt`, `do not hurt me`, `do not be stubborn`, `take care of yourself`, `try to understand`.
2. **`can i help you` (4 FPs):**
   - Attracts polite social interactions and offers:
     `do me a favour`, `shall we go outside`, `i am very happy`, `you are welcome`.
3. **`i do not mean it` (6 FPs):**
   - Attracts hand-forward conversational gestures:
     `what do you want to become`, `i am feeling bored`, `you are good`, `pour some more water into the glass`, `i promise`, `what did you tell him`.

The model has learned coarse gesture clusters (e.g., chest-centered defensive hands vs. forward-reaching social hands) but lacks the fine-grained discriminative power to distinguish specific sentences within each cluster across different signers.

---

## 4. Biomechanical & Preprocessing Domain Shift (Signers 1–5 vs Signer 6)

A detailed statistical comparison between the training partition (Signers 1–5) and validation partition (Signer 6) exposes major distribution shifts:

| Feature Dimension | Train Split (Signers 1–5) | Validation Split (Signer 6) | Disparity / Domain Shift |
|---|---|---|---|
| **Sample Count** | 298 | 61 | — |
| **Sequence Length Mean** | $89.85 \pm 28.07$ frames | $98.25 \pm 18.66$ frames | **$+9.3\%$ slower signing** ($+336\text{ ms}$) |
| **Sequence Length Median** | 89.0 frames | 98.0 frames | Signer 6 is consistently slower |
| **Left Hand Presence Ratio** | $45.82\%$ | $37.37\%$ | **$-8.45\%$ lower left-hand activity** |
| **Right Hand Presence Ratio**| $54.93\%$ | $64.86\%$ | **$+9.93\%$ higher right-hand activity** |
| **Both Hands Active Ratio** | $26.04\%$ | $27.74\%$ | Comparable |
| **Neither Hand Active Ratio** | $25.28\%$ | $25.52\%$ | Comparable |
| **Left Hand X/Y/Z Mean** | $0.1993$ | $0.4100$ | **$+105.7\%$ spatial offset** |
| **Left Hand X/Y/Z Std** | $0.5656$ | $0.8992$ | **$+58.9\%$ greater spatial variance** |
| **Right Hand X/Y/Z Mean** | $-0.0991$ | $-0.1100$ | Small shift |
| **Right Hand X/Y/Z Std** | $0.4576$ | $0.5966$ | **$+30.4\%$ greater spatial variance** |
| **Pose Coordinate Mean** | $-0.1774$ | $-0.0649$ | $+0.1125$ vertical/depth shift |
| **Normalization Validity** | $100.0\%$ | $100.0\%$ | 0% failure in both |
| **Applied Preprocess Rotation**| $40.3\%$ rotated $90^\circ$ (Signers 1, 2) | $0.0\%$ rotated | Video aspect ratio & sensor variation |

### Key Takeaway:
Signer 6 is significantly more right-hand dominant than the native training signers and executes gestures in a much broader spatial volume (especially for the left hand: mean coordinate offset $+105\%$, standard deviation $+59\%$). Because the model was trained without spatial jittering or scaling, it treats these physical differences as out-of-distribution feature values.

---

## 5. Architectural Flaw: Recency Bias of Endpoint Pooling

In `SignBridgeGRUClassifier`:
```python
final_forward = forward_out[batch_indices, lengths - 1, :]
initial_backward = backward_out[batch_indices, 0, :]
seq_embedding = torch.cat([final_forward, initial_backward], dim=-1)
```

### Empirical Endpoint Analysis:

| Metric | Training Split (Signers 1–5) | Validation Split (Signer 6) |
|---|---|---|
| **Frame 0: Left Hand Present** | $25.2\%$ | **$11.5\%$** (Hands absent in $88.5\%$) |
| **Frame 0: Right Hand Present** | $23.5\%$ | **$6.6\%$** (Hands absent in $93.4\%$) |
| **Frame $T-1$: Left Hand Present** | $45.6\%$ | $31.1\%$ |
| **Frame $T-1$: Right Hand Present** | $34.9\%$ | $39.3\%$ |

### Diagnosis:
At Frame 0 and Frame $T-1$, the signers are predominantly in neutral resting poses (hands lowered or transitioning).
While the recurrent hidden state theoretically encodes the entire sequence history, a single-layer vanilla GRU running over $T=90 \dots 150$ frames suffers from recency bias. The forward state at $T-1$ is heavily weighted toward the concluding resting movement, and the backward state at $0$ is heavily weighted toward the initial onset movement. The active lexical signing occurring in frames $20 \dots 80$ is attenuated.

---

## 6. Preprocessing Integrity & Data Consistency

| Check | Result | Evidence |
|---|---|---|
| **Duplicate Clip IDs** | ✅ None | All 298 train and 61 val clips have distinct IDs |
| **Casing / Whitespace Mismatch** | ✅ None | `train_labels` and `val_labels` match exactly across all string checks |
| **Missing Value Leakage** | ✅ Zero | All tensors contain 100% finite values; zero-fabrication holds |
| **Normalization Stability** | ✅ 100% | Shoulder normalization ratio is 1.0 across all samples |

There are no data contamination, labeling errors, or preprocessing defects. The problem is fundamentally **model representation and sample distribution shift**.

---

## 7. Metric Implementation Analysis: Active vs. Global Macro-F1

- **Global 71-Class Macro-F1 (in `metrics.py`):** **`0.0376`**
  - Averages all 71 training classes.
  - The 10 classes with 0 support in validation contribute `f1 = 0.0`.
  - Theoretical maximum is $61 / 71 \approx 0.8592$.
- **Active 61-Class Macro-F1:** **`0.0437`**
  - Averages only the 61 classes that actually appear in the validation set.
  - Mathematical relationship: $0.0437 \times \frac{61}{71} = 0.03754 \approx \mathbf{0.0376}$.

**Conclusion:** The metric implementation is mathematically sound and consistent with standard closed-set multi-class evaluation. However, Active Macro-F1 ($0.0437$) provides a clearer view of performance on evaluated classes and should be reported alongside Global Macro-F1.

---

## 8. Summary of Confirmed Problems vs. Hypotheses

### Confirmed Problems (Supported by Direct Empirical Evidence)

1. **Extreme Overfitting to Small Sample Base:**
   $N=298$ training samples across 71 categories ($\approx 4.2$ clips/class) is insufficient for unaugmented sequence learning. The model achieved $68.46\%$ training accuracy while validation accuracy remained at $4.92\%$.
2. **Cross-Signer Biomechanical Shift:**
   Signer 6 has a $+105\%$ left-hand spatial coordinate offset and signs $9.3\%$ slower than training signers. The unaugmented model has no mechanism to normalize or generalize over these signer variations.
3. **Endpoint Pooling Dilution:**
   In $88\% - 93\%$ of validation clips, hands are completely absent at Frame 0. Concatenating endpoints disproportionately samples resting/boundary frames rather than core gesture trajectory frames.
4. **Attractor Collapse:**
   8 classes absorb $55.7\%$ of all validation predictions, reflecting a collapse of the decision space into coarse spatial clusters.

### Hypotheses (To Be Validated in Future Iterations)

1. *Hypothesis:* Applying Gaussian landmark jittering and spatial scaling during training will reduce the left-hand coordinate shift penalty.
2. *Hypothesis:* Temporal attention pooling or mask-aware mean pooling over active frames will eliminate the endpoint boundary bias.
3. *Hypothesis:* Label smoothing ($\alpha=0.1$) will suppress the formation of dominant attractor classes.

---

## 9. Top 3 Evidence-Supported Improvements (Ranked by Priority)

### Priority 1: Replace Endpoint Pooling with Temporal Attention or Mask-Aware Pooling
- **Reason:** Confirmed empirical evidence shows Frame 0 and Frame $T-1$ are resting frames (hands absent $75\% - 93\%$).
- **Action:** Replace `torch.cat([forward[T-1], backward[0]])` with **Temporal Masked Average Pooling** or **Attention Pooling** over active gesture frames:
  $$\mathbf{h}_{\text{seq}} = \frac{1}{\sum_{t=1}^{T} m_t} \sum_{t=1}^{T} m_t \mathbf{h}_t$$
  where $m_t = 1$ if at least one hand is detected, and $0$ otherwise.

### Priority 2: Coordinate & Spatial Data Augmentation
- **Reason:** Confirmed empirical evidence shows Signer 6's left-hand coordinate mean is shifted by $+105\%$ ($0.41$ vs $0.20$) and standard deviation is $+59\%$ wider.
- **Action:** Introduce on-the-fly landmark augmentation in the training DataLoader:
  - Random spatial scaling: $x' = x \times s$, where $s \sim \mathcal{U}(0.85, 1.15)$
  - Random landmark jittering: $x' = x + \epsilon$, where $\epsilon \sim \mathcal{N}(0, 0.02^2)$
  - Random horizontal translation relative to the shoulder midpoint.

### Priority 3: Regularization & Loss Calibration
- **Reason:** Confirmed empirical evidence shows training accuracy reached $68.46\%$ while validation plateaued at $4.92\%$, with predictions collapsing into 8 attractor classes.
- **Action:**
  - Increase Dropout from $0.2$ to $0.4$.
  - Apply **Label Smoothing** ($\alpha = 0.1$) in `nn.CrossEntropyLoss(label_smoothing=0.1)` to penalize overconfident attractor probabilities.
  - Increase weight decay in AdamW to $10^{-3}$.
