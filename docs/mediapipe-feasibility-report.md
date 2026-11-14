# MediaPipe Landmark Extraction Feasibility Report
## SignBridge AI — Step 2.2B

**Date:** 2026-11-14
**Interpreter:** `backend/.venv` — Python 3.12.10
**MediaPipe version:** 0.10.21
**OpenCV version:** 4.11.0
**NumPy version:** 1.26.4

---

## 1. Environment

| Component | Version / Detail |
|---|---|
| MediaPipe | 0.10.21 (Tasks API — `mediapipe.tasks.python.vision`) |
| OpenCV | 4.11.0 (via `opencv-python==4.11.0.86`) |
| NumPy | 1.26.4 |
| Python | 3.12.10 (venv: `backend/.venv`) |
| OS | Windows 11 (Smart App Control enabled) |
| Hand model | `hand_landmarker.task` (7.8 MB, MediaPipe official) |
| Pose model | `pose_landmarker_lite.task` (5.8 MB, MediaPipe official) |
| Face model | `face_landmarker.task` (3.8 MB, MediaPipe official) |

**MediaPipe architecture note:**
MediaPipe 0.10.21 uses the Tasks API (`mediapipe.tasks.python.vision`). The older `mp.solutions.holistic` / `mp.solutions.hands` API no longer exists in this version. The Tasks API loads a native shared library (`libmediapipe.dll` on Windows), which is blocked by Windows Smart App Control (SAC) when installed via MediaPipe 1.2.3. After downgrading to 0.10.21, SAC allowed the library to load successfully.

**Script:** `scripts/test_mediapipe_extraction.py`
**Raw metrics:** `docs/mediapipe-feasibility-report-raw-data.json`

---

## 2. Videos Tested

Five ISL-CSLTR sentence-level videos were tested, covering 4 distinct signers and 4 distinct sentences.

| # | File | Signer | Sentence | Resolution | FPS | Frames |
|---|---|---|---|---|---|---|
| 1 | `free (2).MP4` | Signer 2 | "are you free today" | 1920×1080 | 25 | 78 |
| 2 | `MVI_6209.MP4` | Signer 1 | "are you hiding something" | 1920×1080 | 25 | 95 |
| 3 | `bring water for me (1).MP4` | Signer 1 | "bring water for me" | 1920×1080 | 25 | 116 |
| 4 | `can you repeat that please (3).MP4` | Signer 3 | "can you repeat that please" | 1920×1080 | 25 | 109 |
| 5 | `are you free today.mp4` | Signer 7 | "are you free today" | 320×240 | 25 | 50 |

Videos 1–4 are full-HD recordings from the main dataset. Video 5 is a low-resolution preview copy included for signer diversity.

---

## 3. Per-Video Detection Results

### Video 1 — `free (2).MP4` (Signer 2, 1920×1080)

| Metric | Value |
|---|---|
| Frames read | 78 / 78 |
| Left hand detected | 32 frames (41.0%) |
| Right hand detected | 43 frames (55.1%) |
| Both hands detected | 29 frames (37.2%) |
| Pose detected | **78 frames (100.0%)** |
| Face detected | 2 frames (2.6%) |
| Frames missing hands | 32 (41.0%) |
| Processing time | 12.3 s |

**Observation:** Pose detected in every frame. Hand detection is partial — only one hand is visible in some frames (signer moves hands in and out of frame). Face detection is very low; the signer's face is likely angled away from camera or cropped by the portrait framing.

---

### Video 2 — `MVI_6209.MP4` (Signer 1, 1920×1080)

| Metric | Value |
|---|---|
| Frames read | 95 / 95 |
| Left hand detected | 72 frames (75.8%) |
| Right hand detected | 42 frames (44.2%) |
| Both hands detected | 27 frames (28.4%) |
| Pose detected | **95 frames (100.0%)** |
| Face detected | 0 frames (0.0%) |
| Frames missing hands | 8 (8.4%) |
| Processing time | 14.1 s |

**Observation:** Strong hand detection; left hand dominant (75.8%). Pose at 100%. Face at 0% — signer's face is likely outside the detection region or obscured. Only 8.4% of frames have no hands detected at all — the best hand coverage of the five videos.

---

### Video 3 — `bring water for me (1).MP4` (Signer 1, 1920×1080)

| Metric | Value |
|---|---|
| Frames read | 116 / 116 |
| Left hand detected | 3 frames (2.6%) |
| Right hand detected | 47 frames (40.5%) |
| Both hands detected | 0 frames (0.0%) |
| Pose detected | **116 frames (100.0%)** |
| Face detected | 116 frames (100.0%) |
| Frames missing hands | 66 (56.9%) |
| Processing time | 16.7 s |

**Observation:** Only one hand detected in any frame; both hands never detected simultaneously. This may reflect a signing style that is predominantly one-handed for this phrase, or that one hand stays near the body and falls below the hand detection threshold. Face at 100% — signer faces the camera directly. The gap between face/pose coverage (100%) and hand coverage (40.5% max) is the largest of the five videos, suggesting the hands spend significant time in low-confidence positions.

---

### Video 4 — `can you repeat that please (3).MP4` (Signer 3, 1920×1080)

| Metric | Value |
|---|---|
| Frames read | 109 / 109 |
| Left hand detected | 1 frame (0.9%) |
| Right hand detected | 51 frames (46.8%) |
| Both hands detected | 0 frames (0.0%) |
| Pose detected | **109 frames (100.0%)** |
| Face detected | 109 frames (100.0%) |
| Frames missing hands | 57 (52.3%) |
| Processing time | 17.1 s |

**Observation:** Nearly identical pattern to Video 3. The left hand is essentially undetected (1 frame). Both-hand simultaneous detection is zero. Face and pose at 100%. The pattern across videos 3 and 4 suggests that for some signing styles or sentences, one hand is dominant and the other remains stationary or at a low position not reliably detected by the lite model.

---

### Video 5 — `are you free today.mp4` (Signer 7, 320×240)

| Metric | Value |
|---|---|
| Frames read | 50 / 50 |
| Left hand detected | 42 frames (84.0%) |
| Right hand detected | 28 frames (56.0%) |
| Both hands detected | 23 frames (46.0%) |
| Pose detected | **50 frames (100.0%)** |
| Face detected | 50 frames (100.0%) |
| Frames missing hands | 3 (6.0%) |
| Processing time | 5.1 s |

**Observation:** The best hand and face detection of all five videos, despite the low resolution (320×240). The signer is centered in the frame with both hands and face visible throughout. This suggests that **framing and centering matter more than resolution** for MediaPipe's detection quality. The lite models perform reliably even at 320×240.

---

## 4. Overall Detection Summary

| Metric | Video 1 | Video 2 | Video 3 | Video 4 | Video 5 | Average |
|---|---|---|---|---|---|---|
| Left hand ratio | 41.0% | 75.8% | 2.6% | 0.9% | 84.0% | **40.9%** |
| Right hand ratio | 55.1% | 44.2% | 40.5% | 46.8% | 56.0% | **48.5%** |
| Both hands ratio | 37.2% | 28.4% | 0.0% | 0.0% | 46.0% | **22.3%** |
| Pose ratio | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| Face ratio | 2.6% | 0.0% | 100.0% | 100.0% | 100.0% | **60.5%** |
| Missing hands | 41.0% | 8.4% | 56.9% | 52.3% | 6.0% | **32.9%** |
| Processing time | 12.3 s | 14.1 s | 16.7 s | 17.1 s | 5.1 s | **13.1 s** |

**Key findings:**
- **Pose detection: 100% across all videos.** This is the most reliable modality and must be the foundation of any feature representation.
- **Right hand average: 48.5%.** The right hand is detected in roughly half of all frames across the dataset.
- **Left hand average: 40.9%.** Slightly lower than right hand; more variable across signers.
- **Both hands simultaneously: 22.3% average.** Drops to 0% for two out of five videos — the most common case is a single visible hand.
- **Face: highly variable (0–100%).** Strongly dependent on signer framing. Cannot be relied upon as a stable feature for all ISL-CSLTR videos.
- **Missing hands (no hand at all): 32.9% average.** One in three frames has zero hand landmarks — this directly affects the T×162 representation.
- **Processing throughput:** ~6–9 frames/sec on CPU for 1920×1080. For a 687-video dataset totalling ~2,000+ seconds of footage, full extraction will take several hours on the same CPU.

---

## 5. Actual Landmark Tensor Shapes

The MediaPipe Tasks API returns the following for each detected instance:

| Modality | Landmarks | Coords | Dims per instance | Note |
|---|---|---|---|---|
| Hand (per hand) | 21 | 3 (x, y, z) | **63D** | z is depth relative to wrist, normalised; not raw metric depth |
| Pose (body) | 33 | 3 (x, y, z) | **99D** | World landmarks (metric, in metres) also available separately |
| Face | 478 | 3 (x, y, z) | **1,434D** | z is depth relative to face centre, normalised |

**Combined maximum per frame** (both hands + pose + face):
63 + 63 + 99 + 1,434 = **1,659D**

---

## 6. Assessment of the T×162 Design Assumption

The implementation plan proposed a `T × 162` feature representation per video:
- 42D : left hand (21 landmarks × 2: x, y only)
- 42D : right hand (21 landmarks × 2: x, y only)
- 78D : upper-body pose (first 26 landmarks × 3: x, y, z)

### Verdict: Achievable, with explicit design decisions required

The 162D structure is **reachable** from MediaPipe's actual outputs, but only by making the following explicit choices — none of which the MediaPipe API enforces automatically:

| Choice | Detail |
|---|---|
| Drop z from hands | MediaPipe returns 21×3=63D per hand; retaining only x,y gives 21×2=42D ← matches proposal |
| Truncate pose to 26 landmarks | MediaPipe returns 33 landmarks; using only the first 26 gives 26×3=78D ← matches proposal, but "first 26" is not a named MediaPipe subset — must be verified against the [PoseLandmarker landmark index map](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| Exclude face | FaceLandmarker returns 478×3=1,434D — must be entirely excluded to maintain 162D |
| Zero-pad absent hands | When a hand is not detected in a frame, 42D must be filled with zeros (or masked). Missing both hands ≈ 33% of frames on average |

### Caveats

**(a) Dropping z-depth from hands** discards articulation depth — the third dimension that distinguishes some handshapes that look identical from a 2D projection (e.g. a flat palm vs a cupped palm at the same x/y position). This may reduce model accuracy.

**(b) "Upper-body first 26" is an unverified assumption.** The PoseLandmarker lite model landmark index map must be inspected to confirm that landmarks 0–25 correspond exclusively to upper-body joints (shoulders, elbows, wrists, hips) and do not include leg or foot keypoints. If the first 26 landmarks include lower-body joints, a custom landmark selection is needed.

**(c) Missing hand zero-padding.** On average, 32.9% of frames have no hand landmarks. For videos 3 and 4, the left hand is essentially absent (0–2.6%). Zero-padding is a reasonable strategy but must be paired with a **binary detection mask** (1 = present, 0 = absent) fed alongside the pose features, so the model can distinguish genuine zero-position hands from absent hands.

**(d) Simultaneous two-hand detection is rare.** Both hands detected simultaneously averaged only 22.3% across the five videos. Models relying on both-hand features will encounter substantial sparsity without masking.

### Alternative 225D representation worth considering

| Component | Landmarks | Coords | Dims |
|---|---|---|---|
| Left hand | 21 | 3 (x, y, z) | 63D |
| Right hand | 21 | 3 (x, y, z) | 63D |
| Pose (all 33) | 33 | 3 (x, y, z) | 99D |
| **Total** | | | **225D** |

This retains full 3D depth for hand articulation, avoids arbitrary pose truncation, and excludes the unreliable face modality. The input width increases by only 39% over 162D, which is negligible for LSTM/Transformer architectures. This is the **recommended baseline** for the pipeline.

---

## 7. Failure Analysis

### 7.1 Low and inconsistent hand detection

Across four of the five videos, one or both hands are absent in a majority of frames. The primary causes observed:

- **Camera distance and signer framing:** The ISL-CSLTR videos are recorded at upper-body camera distance (1920×1080, signer fills the frame from waist to above head). At this framing, hands in motion sometimes enter the edge of the frame or move behind the body.
- **Signing style variation:** Some ISL sentences appear to be predominantly one-handed (videos 3, 4). The left hand may function as a base/passive hand and remain relatively stationary near the body — often below the MediaPipe confidence threshold for detection.
- **Lite model limitations:** The `pose_landmarker_lite.task` model is optimised for speed over accuracy. The `pose_landmarker_full.task` model may produce higher detection rates at higher computational cost.
- **Hand occlusion:** Signers occasionally bring one hand in front of the other, reducing the visibility of the rear hand.

### 7.2 Face detection variability

Face detection ranged from 0% (videos 1–2) to 100% (videos 3–5). This is primarily explained by signer-camera angle: some signers face directly at the camera; others face slightly sideways. The face landmark detector requires a minimum frontal face confidence. Face landmarks should not be relied upon as a primary ISL feature without controlling for recording angle.

### 7.3 Timestamp bug in the original script (corrected)

The first extraction run shared landmarker instances across all five videos. The MediaPipe VIDEO-mode API enforces monotonically increasing timestamps within a single landmarker session. When the second video started at frame 0 (timestamp = 0 ms), it was lower than the first video's final timestamp (~3,080 ms), causing all frames of videos 2–5 to raise `"Input timestamp must be monotonically increasing"` and produce zero detections.

**Fix applied:** Fresh landmarker instances are now created for each video, giving each its own monotonic timestamp clock starting at 0 ms. This is the **required pattern** for multi-video batch extraction pipelines.

### 7.4 Processing speed

At 6–9 frames/sec throughput on CPU for 1920×1080 videos, the full ISL-CSLTR dataset (687 videos, estimated 2,000+ seconds of footage at 25 fps ≈ 50,000+ frames) will require approximately **1.5–2.5 hours** of CPU time for complete extraction. GPU acceleration is not available through the MediaPipe Tasks API (CPU inference only via XNNPACK delegate). This is acceptable for a one-time offline preprocessing step.

### 7.5 Non-critical warnings observed

| Warning | Source | Significance |
|---|---|---|
| `inference_feedback_manager.cc: single signature inference` | TFLite feedback tensor registration | Benign — model format limitation, no effect on output |
| `landmark_projection_calculator.cc: Using NORM_RECT without IMAGE_DIMENSIONS` | PoseLandmarker (lite) internals | Benign — coordinates are still returned normalised |
| `face_landmarker_graph.cc: Sets FaceBlendshapesGraph acceleration to xnnpack` | FaceLandmarker loading | Informational — XNNPACK CPU acceleration applied |

None of these warnings indicate incorrect behaviour or corrupt landmark data.

---

## 8. Temporal Consistency Observations

Frame-by-frame detection was recorded per video. Based on the detection ratios:

- **Pose** is temporally stable — 100% detection across all 448 frames tested. There are no isolated drop-out frames.
- **Hands** are temporally inconsistent — detection drops appear to cluster (e.g. when a hand leaves the frame mid-signing motion) rather than being randomly scattered. This suggests hand tracking fails during rapid motion or at frame edges.
- **Face** switches sharply between detected and undetected based on head angle — this is binary in practice (consistent either on or off for a given recording session).

The temporal inconsistency of hands means that a **temporal interpolation strategy** will be needed for absent frames during the full preprocessing pipeline (or alternatively, a masking-based approach as described in Section 6).

---

## 9. Recommendations for Preprocessing

### Immediate next step (Step 2.3)

Implement a full batch preprocessing pipeline with the following design:

1. **Input:** All 687 MP4 videos from ISL-CSLTR (after full dataset download).
2. **Landmarker config:**
   - Create fresh landmarker instances per video (required for timestamp monotonicity).
   - HandLandmarker: `num_hands=2`, `VIDEO` mode, confidence thresholds 0.5.
   - PoseLandmarker: `pose_landmarker_lite.task`, `VIDEO` mode.
   - FaceLandmarker: optional secondary extraction.
3. **Feature vector design:** Use the **225D representation** (63D left hand + 63D right hand + 99D pose) rather than the original 162D. This avoids z-depth truncation and arbitrary pose truncation.
4. **Missing landmark handling:**
   - Record a binary detection mask per frame: `hand_left_present` (0/1), `hand_right_present` (0/1), `pose_present` (0/1).
   - Zero-pad absent hand slots.
   - Do NOT interpolate or fill missing values silently.
5. **Output format:** Per-video NumPy arrays of shape `(T, 225)` and a corresponding mask array `(T, 3)`. Save as `.npz` files alongside metadata CSV.
6. **Verify pose landmark subset:** Before using "first 26" pose landmarks for the 78D slot, inspect the MediaPipe PoseLandmarker index map to confirm which joints are included.
7. **Batch throughput:** Run extraction on all 687 videos and report per-video completion, errors, and detection statistics.

### Secondary recommendation

Consider running the extraction on a GPU-enabled cloud environment (Google Colab) for speed. MediaPipe's CPU-only XNNPACK inference at ~6–9 fps/video is acceptable but will take 1.5–2.5 hours for the full dataset on this machine.

---

## 10. Summary Table

| Question | Answer |
|---|---|
| Does MediaPipe 0.10.21 load on this machine? | ✅ Yes (after downgrade from 1.2.3 blocked by Windows Smart App Control) |
| Does the Tasks API work for VIDEO-mode extraction? | ✅ Yes — with per-video fresh instances |
| Is pose reliably detected? | ✅ Yes — 100% detection across all 5 videos |
| Are hands reliably detected? | ⚠️ Partial — avg 40.9% left, 48.5% right, 22.3% both; 32.9% frames have no hands |
| Is face reliably detected? | ⚠️ Variable — 0–100% depending on signer angle; not reliable across all signers |
| Is the T×162 design achievable? | ✅ Achievable with explicit design choices (drop hand z, truncate pose to 26, zero-pad absences) |
| Is T×162 recommended? | ⚠️ Acceptable but T×225 (full 3D hands + all 33 pose landmarks) is preferable |
| Is batch extraction feasible? | ✅ Yes — estimated 1.5–2.5 hours on CPU for full 687-video dataset |
| Biggest risk for the full pipeline? | High hand-absence rate (~33%) — must be handled with masking, not imputation |

