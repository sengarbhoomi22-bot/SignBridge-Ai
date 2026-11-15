# SignBridge AI — Step 3.6: GRU Architecture & Improvement Design Review (Revised)

**Date:** November 2026  
**Auditor:** Automated Engineering Review (SignBridge ML Architecture)  
**Document Purpose:** Architectural and mathematical review of improvement proposals before any code modification or experimental execution.  
**Scope:** Sequence aggregation pooling, landmark data augmentation, temporal resampling, loss regularization, training/inference compatibility, and experimental protocol.

---

## 1. Bidirectional GRU Hidden State Extraction: Technical Explanation

### 1.1 Existing Mechanism in `backend/app/ml/models/gru_classifier.py`

In `SignBridgeGRUClassifier.forward()`:
```python
# 1. Linear input projection
proj_features = self.input_proj(features)  # (B, T_max, 128)

# 2. Packed sequence collation
packed_in = pack_padded_sequence(proj_features, lengths_cpu, batch_first=True, enforce_sorted=False)

# 3. Recurrent encoding
packed_out, _ = self.gru(packed_in)

# 4. Unpack sequence
unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_max, 256)

# 5. Extract endpoints
if self.bidirectional:
    forward_out = unpacked_out[:, :, :self.hidden_dim]       # (B, T_max, 128)
    backward_out = unpacked_out[:, :, self.hidden_dim:]      # (B, T_max, 128)

    batch_indices = torch.arange(batch_size, device=features.device)
    last_indices = lengths - 1

    final_forward = forward_out[batch_indices, last_indices, :]  # (B, 128)
    initial_backward = backward_out[batch_indices, 0, :]         # (B, 128)

    seq_embedding = torch.cat([final_forward, initial_backward], dim=-1)  # (B, 256)
```

### 1.2 Mathematical Formulation and Sequence Coverage
- **Forward GRU:** Processes the sequence from $t=0$ to $t=T_i-1$:
  $$\vec{h}_t = \text{GRU}_{\text{fwd}}(\vec{h}_{t-1}, x_t)$$
  The extracted state $\vec{h}_{T_i-1}$ has processed the **entire valid sequence** from frame 0 through frame $T_i-1$.
- **Backward GRU:** Processes the sequence in reverse order from $t=T_i-1$ down to $t=0$:
  $$\overleftarrow{h}_t = \text{GRU}_{\text{bwd}}(\overleftarrow{h}_{t+1}, x_t)$$
  The extracted state $\overleftarrow{h}_0$ has likewise processed the **entire valid sequence** in reverse, from frame $T_i-1$ back through frame 0.
- **Combined Representation:** $\mathbf{h}_{\text{seq}} = [\vec{h}_{T_i-1} \,\|\, \overleftarrow{h}_0] \in \mathbb{R}^{256}$.
- **Important Technical Clarification:** It is incorrect to claim that the bidirectional GRU ignores the middle frames. By definition of recurrent state transitions, both the forward and backward terminal states incorporate inputs from every valid time step $t \in [0, T_i-1]$.

---

## 2. Technical Evaluation of the Endpoint Sensitivity Hypothesis

### 2.1 The Endpoint Sensitivity Hypothesis
While both terminal states theoretically encode complete sequence information, recurrent transitions in finite-capacity networks can exhibit recency bias or boundary sensitivity over longer sequences ($T=90 \dots 150$ frames).

The hypothesis is that because the final forward state $\vec{h}_{T_i-1}$ immediately follows the final frames, and the backward final state $\overleftarrow{h}_0$ immediately follows the initial frames, the representation may be disproportionately sensitive to boundary dynamics compared to alternative pooling mechanisms that explicitly sample all intermediate hidden states.

### 2.2 Empirical Observations (Supporting Evidence vs. Assumptions)

| Aspect | Direct Empirical Evidence | Working Hypothesis (To Be Tested) |
|---|---|---|
| **Boundary Hand Detection** | Hands are absent in **$88.5\%$ (left) and $93.4\%$ (right) of validation clips at Frame 0**, and absent in **$60.7\% - 68.9\%$ at Frame $T-1$**. | Boundary absence may cause the terminal hidden states to capture resting pose dynamics rather than active gesture transitions. |
| **Prediction Concentration** | 8 classes accounted for $55.7\%$ of validation predictions on Signer 6. | Predictions may cluster around resting or transition trajectories shared across multiple signs. |
| **Recurrent Fading Memory** | Recurrent gates update continuously; in small-sample regimes ($N=298$), network weights may struggle to maintain long-range dependencies over 100+ frames. | Explicit temporal pooling over all steps could provide a more direct gradient path to intermediate frames. |

**Scientific Conclusion:**
Endpoint sensitivity is a **well-motivated hypothesis supported by boundary hand-absence statistics**, but it is **not conclusively established as the sole or primary cause of low validation accuracy**. Cross-signer spatial shift (e.g., $+105\%$ coordinate shift in Signer 6's left hand) is an equally plausible primary bottleneck. Therefore, sequence pooling strategies must be evaluated as independent, controlled experiments.

---

## 3. Comparison of Sequence Aggregation Approaches

```
+--------------------------------------------------------------------------------------------------+
| Approach 0: Existing Endpoint Pooling (Baseline)                                                 |
|                                                                                                  |
|   h_0 ------> h_1 ------> ... ------> h_{T-2} ------> [ h_{T-1} (Forward final) ]                |
|   [ h_0 (Backward final) ] <------ ... <------ h_{T-2} <------ h_{T-1}                           |
|   Concatenate: [ h_{T-1}^{fwd} || h_0^{bwd} ]  (Processes full sequence; evaluated at endpoints) |
+--------------------------------------------------------------------------------------------------+
| Approach 1: Mask-Aware Temporal Average Pooling                                                  |
|                                                                                                  |
|   h_0 ------> h_1 ------> ... ------> h_{T-1}                                                    |
|    |           |                       |                                                         |
|    +-----------+-----------+-----------+  ---> Average over valid frames: (1/T) * sum(h_t)        |
|                                                (0 added parameters, uniform frame weighting)     |
+--------------------------------------------------------------------------------------------------+
| Approach 2: Learned Temporal Attention Pooling                                                   |
|                                                                                                  |
|   h_0 ------> h_1 ------> ... ------> h_{T-1}                                                    |
|    |           |                       |                                                         |
|   [a_0]       [a_1]                   [a_{T-1}] ---> Softmax Attention weights                   |
|    |           |                       |                                                         |
|    +-----------+-----------+-----------+  ---> Weighted sum: sum(a_t * h_t)                      |
|                                                (Learned query weights salient gesture frames)    |
+--------------------------------------------------------------------------------------------------+
```

### Detailed Comparison Table

| Property | Experiment 0: Endpoint Pooling (Baseline) | Experiment 1: Mask-Aware Average Pooling | Experiment 2: Learned Temporal Attention Pooling |
|---|---|---|---|
| **Mathematical Formulation** | $\mathbf{h}_{\text{seq}} = [\vec{h}_{T_i-1} \,\|\, \overleftarrow{h}_0]$ | $\mathbf{h}_{\text{seq}} = \frac{1}{T_i} \sum_{t=0}^{T_i-1} \mathbf{h}_t$ | $\mathbf{h}_{\text{seq}} = \sum_{t=0}^{T_i-1} \alpha_t \mathbf{h}_t$, where $\alpha_t = \frac{\exp(e_t)}{\sum_\tau \exp(e_\tau)}$, $e_t = \mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_t)$ |
| **Additional Parameters** | `0` (Baseline) | `0` (Parameter-free) | $\approx 4,112$ parameters ($W \in \mathbb{R}^{16 \times 256}, v \in \mathbb{R}^{16}$) |
| **Direct Gradient Path to Middle Frames** | Via recurrent BPTT chain | Direct skip-connection gradient to all valid time steps | Direct attention-weighted gradient to all valid time steps |
| **Weighting of Resting Frames** | Dependent on recurrent transition state at endpoints | Uniform across all valid frames ($1/T_i$) | Learnable; can downweight frames with lower task relevance |
| **Risk of Overfitting on Small Data ($N=298$)** | Baseline risk | **Lowest** (adds zero trainable parameters) | **Low to Moderate** (adds 4k weights; requires attention dropout) |
| **Inference Latency on CPU** | $\approx 0.05 \text{ ms}$ | $\approx 0.08 \text{ ms}$ | $\approx 0.15 \text{ ms}$ |
| **Selection Criterion** | Baseline reference | Compared independently against Exp 0 | Compared independently against Exp 0 |

---

## 4. Preserving Hand-Presence Masks and Non-Manual Information

### 4.1 Risk of Naive Hand-Presence Thresholding
A naive proposal might attempt to restrict pooling exclusively to frames where hands are detected ($m_{\text{left}} > 0$ or $m_{\text{right}} > 0$).
- **Technical Risk:** In Indian Sign Language, non-manual markers (head orientation, shoulder posture, torso shifts, facial inflections) provide essential grammatical and lexical context. Dropping frames where hands are absent would truncate valid visual signs during transition phases or non-manual expressions.

### 4.2 Recommended Mechanism
1. **Never drop frames from the recurrent sequence:** All $T_i$ physical frames are passed to the GRU.
2. **Modality indicators remain embedded in features:** Indices `225:229` ($m_{\text{left}}, m_{\text{right}}, m_{\text{pose}}, m_{\text{norm}}$) inform the input projection layer about detection validity at each frame.
3. **Use sequence padding masks strictly for batch zero-padding:**
   $$\mathbf{M}_{\text{pad}}[i, t] = \begin{cases} 1.0 & \text{if } t < T_i \\ 0.0 & \text{if } t \ge T_i \end{cases}$$
   Average pooling computes:
   $$\mathbf{h}_{\text{seq}}[i] = \frac{\sum_{t=0}^{T_{\text{max}}-1} \mathbf{M}_{\text{pad}}[i, t] \cdot \mathbf{h}[i, t]}{\sum_{t=0}^{T_{\text{max}}-1} \mathbf{M}_{\text{pad}}[i, t]}$$
   This ensures zero-padded frames are strictly excluded while all physical frames (both manual and non-manual) are fully preserved.

---

## 5. Safe Spatial Landmark Augmentation Design

### 5.1 Non-Negotiable Safety Invariants
1. **Missing-Landmark Coordinate Invariant:**
   If a landmark is undetected (indicated by $m_{\text{left}} == 0.0$ or $m_{\text{right}} == 0.0$), its coordinate slice (`[0:63]` or `[63:126]`) **MUST remain strictly $0.0$**. Adding noise or scaling to undetected landmarks fabricates artificial coordinates and violates the zero-fabrication contract.
2. **Binary Indicator Invariant:**
   Indices `225:229` ($m_{\text{left}}, m_{\text{right}}, m_{\text{pose}}, m_{\text{norm}}$) are discrete detection flags. They **must never be altered, scaled, or jittered**.
3. **Anatomical Scale Consistency:**
   Spatial transformations must apply uniform scaling across connected joints to preserve physical bone lengths (e.g., wrist-to-elbow, shoulder-to-elbow distances).

### 5.2 Mathematical Formulation of Safe Spatial Augmentation
Applied on-the-fly during training batches, exclusively on active landmarks ($m > 0$):

1. **Uniform Spatial Scale:**
   $$s \sim \mathcal{U}(0.90, 1.10)$$
   $$x_{t, j} \gets x_{t, j} \times s \quad (\forall j \text{ where } m_{t, j} > 0)$$
2. **Horizontal Translation (Signer Placement Variance):**
   $$\Delta x \sim \mathcal{U}(-0.03, 0.03)$$
   $$x_{t, j} \gets x_{t, j} + \Delta x \quad (\forall j \text{ where } m_{t, j} > 0)$$
3. **Sensor Gaussian Jitter:**
   $$\epsilon \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.010$$
   $$x_{t, j} \gets x_{t, j} + \epsilon \quad (\text{only where } m_{t, j} == 1.0)$$
4. **Hand-Specific Scaling (Treated as a Separate Optional Experiment):**
   Hand-specific scaling (scaling fingers relative to wrist) introduces non-rigid anatomical transformations. To maintain experimental clarity, hand-specific scaling will **NOT** be included in the default spatial augmentation stage; it will be evaluated only as an independent subsequent exploration if global spatial augmentation demonstrates positive validation gains.

---

## 6. Temporal Resampling Augmentation for Variable-Length Sequences

### 6.1 Motivation
- Training signers mean length: $89.85 \pm 28.07$ frames.
- Validation Signer 6 mean length: $98.25 \pm 18.66$ frames ($\approx 9$ frames slower).

### 6.2 Safe Implementation Principles
1. **Speed Perturbation Factor:**
   Sample $\rho \sim \mathcal{U}(0.85, 1.15)$. Target length: $T' = \max(10, \text{round}(T \times \rho))$.
2. **Continuous Landmark Coordinates ($0 \dots 224$):**
   Resample using 1D linear interpolation along the time axis:
   $$x'(t') = (1 - \lambda) x_{\lfloor t \rfloor} + \lambda x_{\lceil t \rceil}$$
3. **Discrete Indicators ($225 \dots 228$):**
   To prevent intermediate fractional values (e.g. $0.3$ or $0.7$), indicators must be interpolated using nearest-neighbor lookup, or thresholded strictly at $0.5$:
   $$m'(t') = \begin{cases} 1.0 & \text{if } m_{\text{interp}}(t') \ge 0.5 \\ 0.0 & \text{otherwise} \end{cases}$$
4. **Post-Interpolation Zero-Fabrication Enforcement:**
   After interpolation, verify: if $m'_{\text{left}}(t') == 0.0$, strictly set $x'_{t', [0:63]} = 0.0$; if $m'_{\text{right}}(t') == 0.0$, strictly set $x'_{t', [63:126]} = 0.0$. This guarantees no artificial coordinates are created during transition boundaries.

---

## 7. Regularization & Loss Calibration Analysis

### 7.1 Dropout
- **Baseline:** $0.20$.
- **Proposal:** Test $0.35$ on the classifier head ($256 \to 128 \to 71$) to mitigate co-adaptation on the 298 training samples.

### 7.2 Weight Decay (AdamW)
- **Baseline:** $10^{-4}$ (`0.0001`).
- **Proposal:** Test $10^{-3}$ (`0.001`) to provide stronger $L_2$ shrinkage on the recurrent transition matrices ($W_{hh}, W_{ih}$).

### 7.3 Label Smoothing
- **Baseline:** $\alpha = 0.0$ (standard hard one-hot CrossEntropyLoss).
- **Proposal:** Apply label smoothing ($\alpha = 0.10$):
  $$y'_c = (1 - \alpha) y_c + \frac{\alpha}{K} = 0.90 y_c + \frac{0.10}{71} \approx 0.90 y_c + 0.0014$$
  This directly bounds maximum logit divergence, penalizing overconfident attractor predictions.

---

## 8. Training and Inference Architecture Compatibility

To guarantee full backward and forward compatibility with `SignBridgeInferenceEngine`:

1. **Model Parameterization:**
   Add `pooling_strategy: str = "endpoint"` (`"endpoint"`, `"mean"`, `"attention"`) to `SignBridgeGRUClassifier`:
   ```python
   def __init__(
       self,
       input_dim: int = 229,
       hidden_dim: int = 128,
       num_layers: int = 1,
       num_classes: int = 71,
       bidirectional: bool = True,
       dropout: float = 0.2,
       pooling_strategy: str = "endpoint",  # 'endpoint', 'mean', 'attention'
   )
   ```
2. **Transparent Checkpoint Reload:**
   `get_config()` includes `"pooling_strategy"`. `from_config()` defaults to `"endpoint"` when loading legacy checkpoints.
3. **Inference Interface Unaltered:**
   `SignBridgeInferenceEngine.load_from_checkpoint()` seamlessly instantiates the model via `from_config()` without modifying `inference.py`.

---

## 9. Staged Experimental Protocol

Each experiment tests **one variable at a time** under identical conditions:
- **Fixed Random Seed:** `42`
- **Fixed Partitions:** Train (Signers 1–5, $N=298$), Validation (Signer 6, $N=61$), Test (Signer 7, **strictly held out**)
- **Evaluation Metrics:** Primary selection metric is **Validation Macro-F1**, supported by Overall Accuracy, Active Macro-F1, and per-class confusion metrics.
- **Isolated Checkpoints:** Every experiment writes to its own separate directory.

### 9.1 Experimental Stages

```
[ Stage 1: Pooling Comparison ]
  ├── Exp 0: Endpoint Pooling Baseline (models/checkpoints/exp0_baseline/)
  ├── Exp 1: Mean Pooling              (models/checkpoints/exp1_mean_pooling/)
  └── Exp 2: Attention Pooling         (models/checkpoints/exp2_attention_pooling/)
              │
              ▼
   Select Best Pooling Architecture (based on Val Macro-F1)
              │
              ▼
[ Stage 2: Regularization Tuning ]
  └── Exp 3: Best Pooling + Label Smoothing (0.10) + Dropout (0.35)
              │
              ▼
[ Stage 3: Spatial Augmentation ]
  └── Exp 4: Best Regularized Model + Safe Spatial Scaling & Jitter
              │
              ▼
[ Stage 4: Temporal Augmentation ]
  └── Exp 5: Best Spatially Augmented Model + Temporal Resampling (0.85x - 1.15x)
```

### Detailed Experimental Matrix

| Stage | Experiment ID | Checkpoint Directory | Change Evaluated | Comparison Target |
|---|---|---|---|---|
| **Stage 1A** | `exp0_baseline` | `models/checkpoints/exp0_baseline/` | Existing Endpoint Pooling (reproduction) | Baseline reference |
| **Stage 1B** | `exp1_mean_pooling` | `models/checkpoints/exp1_mean_pooling/` | Mask-Aware Temporal Average Pooling | Independently compared vs. Exp 0 |
| **Stage 1C** | `exp2_attention_pooling` | `models/checkpoints/exp2_attention_pooling/` | Learned Temporal Attention Pooling | Independently compared vs. Exp 0 |
| **Selection Gate** | — | — | **Select best pooling architecture** based on Val Macro-F1 | Best of (Exp 0, 1, 2) |
| **Stage 2** | `exp3_regularization` | `models/checkpoints/exp3_regularization/` | Add Label Smoothing ($\alpha=0.10$) + Dropout ($0.35$) | Evaluated on selected pooling model |
| **Stage 3** | `exp4_spatial_aug` | `models/checkpoints/exp4_spatial_aug/` | Add safe spatial scaling & landmark jittering | Evaluated on best Stage 2 model |
| **Stage 4** | `exp5_temporal_aug` | `models/checkpoints/exp5_temporal_aug/` | Add temporal speed resampling ($0.85\times - 1.15\times$) | Evaluated on best Stage 3 model |

---

## 10. Explicit Acceptance Criteria

Before any candidate model from an experimental stage is accepted as the new baseline:

1. **Automated Test Suite Integrity:** All existing tests in `backend/tests/` (ML components, preprocessing, downloader) must pass with exit code 0.
2. **Checkpoint Reload Verification:** The saved checkpoint must reload cleanly into `SignBridgeInferenceEngine` on CPU, executing forward inference on a test array without shape or numerical errors.
3. **Configuration Compatibility:** `get_config()` and `from_config()` must correctly round-trip all hyperparameters.
4. **Signer 7 Isolation:** Signer 7 test data must **never be loaded or accessed** during training, validation, or checkpoint selection.
5. **Demonstrated Validation Improvement:** The candidate model must demonstrate measurable improvement in Validation Macro-F1 on Signer 6 validation data over the baseline ($> 0.0376$). No accuracy or generalization gains may be assumed without empirical test records.

---

## 11. Recommended Implementation Order and Justification

The experiments should be executed strictly in the order: **Stage 1 (Pooling Comparison) $\to$ Stage 2 (Regularization) $\to$ Stage 3 (Spatial Augmentation) $\to$ Stage 4 (Temporal Augmentation)**.

### Justification

1. **Why Pooling First (Stage 1):**
   Determining the sequence aggregation architecture establishes the core representation mechanism. Comparing Endpoint, Mean, and Attention pooling independently against the baseline with all other variables fixed ensures that aggregation differences are isolated cleanly.
2. **Why Regularization Second (Stage 2):**
   Once the superior sequence aggregation mechanism is selected, regularization (label smoothing and dropout) directly addresses the overconfidence and attractor collapse observed in baseline training, stabilizing the loss landscape before data augmentation.
3. **Why Spatial Augmentation Third (Stage 3):**
   Spatial scaling and landmark jittering directly target the confirmed $+105\%$ coordinate shift between Signers 1–5 and Signer 6. Evaluating it after architecture and regularization ensures that improvements are attributable to spatial invariance.
4. **Why Temporal Augmentation Fourth (Stage 4):**
   Speed resampling adds interpolation complexity. Evaluating it last isolates the specific benefit of signing speed invariance on top of an already robust, well-regularized model.
