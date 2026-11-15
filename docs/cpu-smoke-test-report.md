# SignBridge AI — Step 3.4: CPU Smoke Test Report

**Execution Date:** 2026-11-16 00:52:13  
**Hardware Target:** CPU (cpu)  
**PyTorch Version:** `2.15.0+cpu`  
**Final Verdict:** `CPU SMOKE TEST PASSED`  

## 1. Executive Summary

A 10-step isolated CPU smoke test was conducted on real preprocessed ISL-CSLTR tensors. The test verified dataset loading, variable-length collation, forward pass, loss calculation, backpropagation, gradient finiteness, optimizer step, validation evaluation, and checkpoint persistence.

## 2. Test Configuration & Samples

- **Training Samples:** 16 clips across 8 categories (2 clips/class)
- **Validation Samples:** 8 clips across the same 8 categories (1 clip/class)
- **Signers in Training:** Signers 1, 2, 3, 4, 5 (Native & Volunteer)
- **Signers in Validation:** Signer 6 (Volunteer)
- **Test Partition / Signer 7:** **NEVER ACCESSED (0 samples)**
- **Batch Size:** 8 (2 training batches, 1 validation batch)

### Selected Vocabulary (8 classes)

- `He is going into the room`
- `This place is beautiful`
- `are you free today`
- `can i help you`
- `comb your hair`
- `congratulations`
- `do me a favour`
- `do not abuse him`

## 3. Step-by-Step Verification Results

| Step | Description | Result |
|---|---|---|
| **step_1_dataset_selection** | Step 1 Dataset Selection | `PASS` |
| **step_2_input_shapes** | Step 2 Input Shapes | `PASS` |
| **step_3_collation** | Step 3 Collation | `PASS` |
| **step_4_forward_pass** | Step 4 Forward Pass | `PASS` |
| **step_5_loss_computation** | Step 5 Loss Computation | `PASS` |
| **step_6_backpropagation** | Step 6 Backpropagation | `PASS` |
| **step_7_gradient_check** | Step 7 Gradient Check | `PASS` |
| **step_8_optimizer_update** | Step 8 Optimizer Update | `PASS` |
| **step_9_validation_pass** | Step 9 Validation Pass | `PASS` |
| **step_10_checkpoint_save_and_reload** | Step 10 Checkpoint Save And Reload | `PASS` |

## 4. Numerical and Gradient Sanity

- **Trainable Parameters Checked:** 18
- **All Gradients Finite:** `True`
- **Max Gradient Norm:** `4.0259`
- **Parameters Updated by Optimizer:** `True`
- **Max Parameter Delta:** `0.221568`

## 5. Validation Evaluation

- **Validation Loss:** `2.1755`
- **Validation Accuracy:** `0.25`
- **Validation Macro-F1:** `0.1042`
- **Samples Evaluated:** `8`

## 6. Checkpoint Persistence Verification

- **Checkpoint Path:** `models\checkpoints\smoke_test\smoke_test_gru.pt`
- **File Size:** `3167981` bytes
- **Reload Verification:** `True`

## 7. Final Verdict

### ✅ CPU SMOKE TEST PASSED

The GRU architecture, loss function, backpropagation pipeline, and validation harness are fully operational on real preprocessed ISL-CSLTR sequence tensors.
