# SignBridge AI — Experiment 2: Learned Temporal Attention Pooling Implementation Report

**Date:** 2026-11-16  
**Status:** Complete & Verified  
**Target Checkpoint Directory:** `models/checkpoints/exp2_attention_pooling/`  
**Verdict:** **READY FOR EXPERIMENT 2 TRAINING**

---

## 1. Executive Summary

In accordance with the SignBridge AI sequence classification architecture roadmap, **Experiment 2** implements **Learned Temporal Attention Pooling** for the bidirectional GRU sequence classifier (`SignBridgeGRUClassifier`).

This implementation introduces an additive temporal attention mechanism that calculates a dynamic scalar relevance score for every frame in a signing sequence, applies strict padding masking prior to softmax normalization, and aggregates valid bidirectional GRU states as an attention-weighted sum.

Crucially:
- **Zero-padded frames** receive exactly zero attention weight ($-\infty$ masking before softmax).
- **Frames lacking hand detections** remain fully eligible for attention; body posture, face orientation, and presence indicators continue to supply valid linguistic context.
- **Strict backward compatibility** is preserved: `attention_scorer` is conditionally instantiated only when `pooling_strategy == "attention"`, guaranteeing that legacy endpoint baseline (`gru_baseline_v1`) and Experiment 1 mean pooling (`exp1_mean_pooling`) checkpoints load cleanly without missing parameter key errors.
- **Classifier interface** is invariant: sequence representation dimension remains $2 \times D_{\text{hidden}} = 256$, matching the input dimension expected by the classification head.

A dedicated 12-test suite (`backend/tests/test_attention_pooling.py`) was created, and all 78 unit/regression tests in the repository passed with zero regressions.

---

## 2. Modified & Created Files

| File | Change Type | Summary of Modifications |
| :--- | :--- | :--- |
| [`backend/app/ml/models/gru_classifier.py`](file:///c:/Users/Bhoomi%20Sengar/SignBridge-Ai/backend/app/ml/models/gru_classifier.py) | Modified | Added `"attention"` to accepted pooling strategies. Added conditional instantiation of `self.attention_scorer = nn.Linear(gru_output_dim, 1, bias=True)` when `pooling_strategy == "attention"`. Implemented masked additive temporal attention in `forward()`. Added optional `return_attention: bool = False` flag and `get_attention_weights(...)` inspection helper. |
| [`backend/app/ml/training/config.py`](file:///c:/Users/Bhoomi%20Sengar/SignBridge-Ai/backend/app/ml/training/config.py) | Modified | Updated type documentation and comments for `pooling_strategy` to support `'endpoint'`, `'mean'`, and `'attention'`. |
| [`scripts/train_gru.py`](file:///c:/Users/Bhoomi%20Sengar/SignBridge-Ai/scripts/train_gru.py) | Modified | Updated `--pooling-strategy` argument parser choices to `["endpoint", "mean", "attention"]`. |
| [`backend/tests/test_attention_pooling.py`](file:///c:/Users/Bhoomi%20Sengar/SignBridge-Ai/backend/tests/test_attention_pooling.py) | New | 12 comprehensive unit tests validating shape, weight normalization, padding masking, variable-length batches, hand-absence inclusion, full 229 feature propagation, finite gradients, serialization, backward compatibility with endpoint and mean checkpoints, and inference engine execution. |
| [`docs/attention-pooling-implementation-report.md`](file:///c:/Users/Bhoomi%20Sengar/SignBridge-Ai/docs/attention-pooling-implementation-report.md) | New | Architectural documentation, mathematical formulation, verification report, and training protocol for Experiment 2. |

---

## 3. Mathematical Formulation & Architecture

### 3.1 Bidirectional GRU Hidden Representations

Given an input batch of landmark sequences $\mathbf{X} \in \mathbb{R}^{B \times T_{\max} \times D_{\text{in}}}$ with lengths $T_b \in [1, T_{\max}]$, packed sequence encoding through a 2-layer bidirectional GRU ($D_{\text{hidden}} = 128$) produces:

$$\mathbf{h}_t^{(b)} = \left[ \vec{h}_t^{(b)} \,\|\, \overleftarrow{h}_t^{(b)} \right] \in \mathbb{R}^{256}, \quad \forall t \in [1, T_b]$$

### 3.2 Additive Temporal Attention Mechanism

For each frame $t \in [1, T_{\max}]$, a trainable linear projection scores frame importance:

$$e_t^{(b)} = \mathbf{w}_a^T \mathbf{h}_t^{(b)} + b_a \in \mathbb{R}$$

where $\mathbf{w}_a \in \mathbb{R}^{256}$ and $b_a \in \mathbb{R}$ constitute `self.attention_scorer`.

### 3.3 Dynamic Padding Masking & Softmax Normalization

To ensure padded timesteps ($t > T_b$) exert no influence on the sequence representation:

$$\tilde{e}_t^{(b)} = \begin{cases} e_t^{(b)}, & \text{if } t \le T_b \\ -\infty, & \text{if } t > T_b \end{cases}$$

Applying softmax along the temporal dimension normalizes energy scores into probability weights:

$$\alpha_t^{(b)} = \frac{\exp(\tilde{e}_t^{(b)})}{\sum_{\tau=1}^{T_{\max}} \exp(\tilde{e}_\tau^{(b)})} = \frac{\exp(e_t^{(b)})}{\sum_{\tau=1}^{T_b} \exp(e_\tau^{(b)})}, \quad \forall t \le T_b$$

$$\alpha_t^{(b)} = 0.0, \quad \forall t > T_b$$

$$\sum_{t=1}^{T_b} \alpha_t^{(b)} = 1.0$$

### 3.4 Attention-Weighted Sequence Embedding

The sequence representation $\mathbf{z}^{(b)} \in \mathbb{R}^{256}$ is the attention-weighted sum of GRU outputs:

$$\mathbf{z}^{(b)} = \sum_{t=1}^{T_b} \alpha_t^{(b)} \mathbf{h}_t^{(b)} \in \mathbb{R}^{256}$$

The classification head (`Linear(256, 128) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(128, num_classes)`) receives $\mathbf{z}^{(b)}$ without any dimensionality modification.

---

## 4. Attention Masking Behavior & Data Invariant Safety

1. **Zero-Padded Timestep Rejection:**
   In `test_padded_frames_receive_zero_attention`, a batch containing variable lengths ($T_0=12$, $T_1=35$) was tested. For sample 0, all 23 padded frames ($t \in [12, 34]$) exhibited exactly $\alpha_t = 0.0000$, while valid frames summed to $1.0000 \pm 10^{-5}$.
2. **Missing-Hand Frame Inclusion:**
   Frames where MediaPipe fails to detect hands are normalized with zero-fabrication safety (coordinates $= 0.0$, hand presence indicators $= 0.0$, pose presence $= 1.0$). In `test_hand_absent_frames_remain_eligible_for_attention`, synthetic samples with absent hands over frames $0..10$ received finite, positive attention weights ($\alpha_t > 0$), confirming that hand-absent frames remain eligible to supply grammatical or contextual transition cues.
3. **Full 229 Feature Propagation:**
   All 229 input features (66 pose, 60 face, 42 left hand, 42 right hand, 4 presence indicators, 15 trajectory dynamics) pass through `input_proj` and recurrent encoding.

---

## 5. Checkpoint Backward & Forward Compatibility

### 5.1 Conditional Scorer Instantiation

In PyTorch, registering an unused submodule (such as `nn.Linear`) adds parameters (`weight` and `bias`) to the module's `state_dict`. When loading a pre-existing checkpoint trained without attention (such as `gru_baseline_v1` or `exp1_mean_pooling`), PyTorch's default strict loading raises:
```text
RuntimeError: Missing key(s) in state_dict: "attention_scorer.weight", "attention_scorer.bias"
```

To prevent this breaking change:
```python
if self.pooling_strategy == "attention":
    self.attention_scorer = nn.Linear(gru_output_dim, 1, bias=True)
else:
    self.attention_scorer = None
```
- **Endpoint Checkpoints:** `attention_scorer` is `None`; state dictionary matches legacy parameters exactly. Verified on disk with `models/checkpoints/gru_baseline_v1/best_gru_model.pt`.
- **Mean Pooling Checkpoints:** `attention_scorer` is `None`; state dictionary matches Experiment 1 parameters exactly. Verified on disk with `models/checkpoints/exp1_mean_pooling/best_gru_model.pt`.
- **Attention Checkpoints:** `attention_scorer` is instantiated and serialized; loaded seamlessly by `SignBridgeInferenceEngine`.

---

## 6. Test Suite Execution & Verification

### 6.1 Attention Pooling Unit Tests (`backend/tests/test_attention_pooling.py`)

Command: `backend\.venv\Scripts\python.exe -m pytest backend/tests/test_attention_pooling.py -v`

```text
backend/tests/test_attention_pooling.py::test_attention_pooling_output_shape PASSED [  8%]
backend/tests/test_attention_pooling.py::test_attention_weights_sum_to_one_across_valid_frames PASSED [ 16%]
backend/tests/test_attention_pooling.py::test_padded_frames_receive_zero_attention PASSED [ 25%]
backend/tests/test_attention_pooling.py::test_variable_length_batches PASSED [ 33%]
backend/tests/test_attention_pooling.py::test_hand_absent_frames_remain_eligible_for_attention PASSED [ 41%]
backend/tests/test_attention_pooling.py::test_all_229_input_features_supported PASSED [ 50%]
backend/tests/test_attention_pooling.py::test_forward_backward_finite_values_and_gradients PASSED [ 58%]
backend/tests/test_attention_pooling.py::test_configuration_serialization_and_roundtrip PASSED [ 66%]
backend/tests/test_attention_pooling.py::test_legacy_endpoint_checkpoint_compatibility PASSED [ 75%]
backend/tests/test_attention_pooling.py::test_exp1_mean_checkpoint_compatibility PASSED [ 83%]
backend/tests/test_attention_pooling.py::test_inference_engine_loads_attention_checkpoint PASSED [ 91%]
backend/tests/test_attention_pooling.py::test_no_accidental_modification_of_existing_pooling_behavior PASSED [100%]

============================= 12 passed in 6.44s ==============================
```

### 6.2 Full Regression Suite

Command: `backend\.venv\Scripts\python.exe -m pytest backend/tests/test_attention_pooling.py backend/tests/test_pooling_strategies.py backend/tests/test_ml_components.py backend/tests/test_preprocessing.py -q`

```text
.............................................................................. [100%]
78 passed in 8.01s
```

### 6.3 Static Checks
- `py_compile`: Passed without syntax or import errors.
- `git diff --check`: Passed with zero whitespace or line-ending conflicts.
- `git status data/processed/isl_csltr_tensors_v2/test`: Clean (Signer 7 test partition untouched).
- Existing checkpoints on disk: `gru_baseline_v1` and `exp1_mean_pooling` verified intact.

---

## 7. Experimental Controls & Protocol for Experiment 2 Training

When training is launched, strict experimental controls must ensure that the pooling strategy is the **sole experimental variable**:

1. **Target Checkpoint Directory:** `models/checkpoints/exp2_attention_pooling/`
2. **Execution Command:**
   ```powershell
   backend\.venv\Scripts\python.exe scripts/train_gru.py --checkpoint-dir models/checkpoints/exp2_attention_pooling --pooling-strategy attention
   ```
3. **Identical Hyperparameters:**
   - Architecture: Bidirectional GRU (num_layers=1, hidden_dim=128, dropout=0.2)
   - Input dimension: 229 features
   - Optimizer: AdamW (lr=0.001, weight_decay=1e-4, grad_clip_norm=1.0)
   - Loss: CrossEntropyLoss (unweighted, label_smoothing=0.0)
   - Batch size: 16
   - Early stopping patience: 10 epochs monitoring validation macro-F1
   - Seed: 42
4. **Data Partitions:**
   - Training: Signers 1–5 (298 samples, 71 classes).
   - Validation: Signer 6 (61 samples, 61 classes).
   - Held-Out Test: Signer 7 (strictly isolated and unaccessed).

---

## 8. Limitations & Open Questions

1. **Scoring Model Complexity:** A single linear layer ($256 \to 1$) evaluates each frame's representation independently without cross-frame query-key interactions (as in multi-head self-attention). While appropriate for this small dataset to prevent severe overfitting, its capacity is focused on temporal saliency rather than relational alignment.
2. **Softmax Entropy:** On short sequences or sequences where all frames are informative, attention weights may approach a near-uniform distribution, behaving similarly to mean pooling. Training metrics will reveal whether the learned weights achieve meaningful temporal selectivity.

---

## 9. Final Verdict

$$\mathbf{READY\ FOR\ EXPERIMENT\ 2\ TRAINING}$$
