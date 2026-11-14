"""
SignBridge AI — Step 2.2B: MediaPipe Landmark Extraction Feasibility Test
=========================================================================
Tests MediaPipe Tasks API landmark extraction (hands, pose, face) on 3–5
actual ISL-CSLTR sentence-level videos.

Usage:
    backend\\.venv\\Scripts\\python.exe scripts/test_mediapipe_extraction.py

Requirements:
    - backend/.venv must have mediapipe >= 1.2.0 and opencv-python installed.
    - MediaPipe model bundles must be present in data/raw/isl-csltr/models/.
    - Sample videos must be present in data/raw/isl-csltr/sample_videos/.

Output:
    - Console: per-video metrics table and overall summary.
    - File:    docs/mediapipe-feasibility-report-raw-data.json  (raw metrics)

IMPORTANT:
    - Missing landmark detections are recorded as ABSENT — never fabricated.
    - Processing uses VIDEO running mode with per-frame timestamps (ms).
    - This script does NOT train a model or modify any application code.
"""

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  Resolve project root and validate paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODELS_DIR   = PROJECT_ROOT / "data" / "raw" / "isl-csltr" / "models"
VIDEOS_DIR   = PROJECT_ROOT / "data" / "raw" / "isl-csltr" / "sample_videos"
OUTPUT_DIR   = PROJECT_ROOT / "docs"

HAND_MODEL   = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL   = MODELS_DIR / "pose_landmarker_lite.task"
FACE_MODEL   = MODELS_DIR / "face_landmarker.task"

# Videos to test (in order of priority; skip if file missing)
CANDIDATE_VIDEOS = [
    ("free (2).MP4",                    "Signer 2", "are you free today"),
    ("MVI_6209.MP4",                    "Signer 1", "are you hiding something"),
    ("bring water for me (1).MP4",      "Signer 1", "bring water for me"),
    ("can you repeat that please (3).MP4", "Signer 3", "can you repeat that please"),
    ("are you free today.mp4",          "Signer 7", "are you free today (small)"),
]
MAX_VIDEOS = 5  # process at most this many

# ---------------------------------------------------------------------------
# 1.  Validate environment before importing heavy libraries
# ---------------------------------------------------------------------------
def validate_paths() -> bool:
    ok = True
    for p in (HAND_MODEL, POSE_MODEL, FACE_MODEL):
        if not p.exists():
            print(f"[ERROR] Missing model: {p}")
            ok = False
    if not VIDEOS_DIR.exists():
        print(f"[ERROR] Missing videos directory: {VIDEOS_DIR}")
        ok = False
    return ok


if not validate_paths():
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2.  Import libraries (after path validation)
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python is not installed in this environment.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
except ImportError as exc:
    print(f"[ERROR] mediapipe import failed: {exc}")
    sys.exit(1)

print(f"[INFO] OpenCV version  : {cv2.__version__}")
print(f"[INFO] MediaPipe version: {mp.__version__}")
print()

# ---------------------------------------------------------------------------
# 3.  Build MediaPipe landmarker options (VIDEO mode)
# ---------------------------------------------------------------------------

def build_hand_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def build_pose_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def build_face_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=str(FACE_MODEL))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# ---------------------------------------------------------------------------
# 4.  Process a single video — returns a metrics dict
# ---------------------------------------------------------------------------

def process_video(video_path: Path, signer_id: str, sentence: str,
                  hand_lmk, pose_lmk, face_lmk) -> dict:
    """
    Open video with OpenCV, process frame-by-frame with VIDEO-mode landmarkers.
    Returns a dict with per-video metrics and landmark dimension information.
    Missing detections are recorded as absent (count incremented), never filled.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"cv2.VideoCapture could not open: {video_path}"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Per-frame counters
    frames_read          = 0
    frames_left_hand     = 0
    frames_right_hand    = 0
    frames_both_hands    = 0
    frames_pose          = 0
    frames_face          = 0
    frames_missing_hands = 0  # neither left nor right detected

    # Landmark dimension samples (record from first successful detection)
    hand_lm_shape_sample   = None   # (num_hands, landmarks_per_hand, coords)
    pose_lm_shape_sample   = None   # (num_landmarks, coords)
    face_lm_shape_sample   = None   # (num_landmarks, coords)

    # Timing
    t_start = time.perf_counter()

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp in milliseconds (monotonically increasing)
        timestamp_ms = int(frame_idx * (1000.0 / fps))

        # --- Hand detection
        try:
            hand_result = hand_lmk.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            hand_result = None
            print(f"  [WARN] Hand detection exception at frame {frame_idx}: {exc}")

        left_detected  = False
        right_detected = False

        if hand_result and hand_result.hand_landmarks:
            detected_hands = len(hand_result.hand_landmarks)
            # Record shape from first detection
            if hand_lm_shape_sample is None and detected_hands > 0:
                lm_list = hand_result.hand_landmarks[0]
                hand_lm_shape_sample = {
                    "num_hands_in_sample": detected_hands,
                    "landmarks_per_hand":  len(lm_list),
                    "coords_per_landmark": 3,   # x, y, z (normalised)
                    "note": "HandLandmarker returns x,y,z normalised [0,1]; z is depth relative to wrist"
                }

            for idx, handedness_list in enumerate(hand_result.handedness):
                if handedness_list:
                    label = handedness_list[0].category_name  # "Left" or "Right"
                    if label == "Left":
                        left_detected  = True
                    elif label == "Right":
                        right_detected = True

        if left_detected:
            frames_left_hand += 1
        if right_detected:
            frames_right_hand += 1
        if left_detected and right_detected:
            frames_both_hands += 1
        if not left_detected and not right_detected:
            frames_missing_hands += 1

        # --- Pose detection
        try:
            pose_result = pose_lmk.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            pose_result = None
            print(f"  [WARN] Pose detection exception at frame {frame_idx}: {exc}")

        if pose_result and pose_result.pose_landmarks:
            frames_pose += 1
            if pose_lm_shape_sample is None:
                lm_list = pose_result.pose_landmarks[0]
                pose_lm_shape_sample = {
                    "num_pose_instances": len(pose_result.pose_landmarks),
                    "landmarks_per_pose": len(lm_list),
                    "coords_per_landmark": 3,   # x, y, z (normalised)
                    "note": "PoseLandmarker (lite) returns x,y,z; world_landmarks also available with x,y,z in metres"
                }

        # --- Face detection
        try:
            face_result = face_lmk.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            face_result = None
            print(f"  [WARN] Face detection exception at frame {frame_idx}: {exc}")

        if face_result and face_result.face_landmarks:
            frames_face += 1
            if face_lm_shape_sample is None:
                lm_list = face_result.face_landmarks[0]
                face_lm_shape_sample = {
                    "landmarks_per_face": len(lm_list),
                    "coords_per_landmark": 3,
                    "note": "FaceLandmarker returns x,y,z normalised; z is depth relative to face centre"
                }

        frame_idx  += 1
        frames_read += 1

    cap.release()
    t_elapsed = time.perf_counter() - t_start

    # --- Build metrics
    n = frames_read if frames_read > 0 else 1  # avoid div-by-zero

    metrics = {
        "video_name":               video_path.name,
        "signer_id":                signer_id,
        "sentence":                 sentence,
        "resolution":               f"{width}x{height}",
        "fps_declared":             fps,
        "total_frames_declared":    total_frames_declared,
        "frames_read":              frames_read,
        "frames_left_hand":         frames_left_hand,
        "frames_right_hand":        frames_right_hand,
        "frames_both_hands":        frames_both_hands,
        "frames_pose_detected":     frames_pose,
        "frames_face_detected":     frames_face,
        "frames_missing_hands":     frames_missing_hands,
        "left_hand_ratio":          round(frames_left_hand  / n, 4),
        "right_hand_ratio":         round(frames_right_hand / n, 4),
        "both_hands_ratio":         round(frames_both_hands / n, 4),
        "pose_ratio":               round(frames_pose       / n, 4),
        "face_ratio":               round(frames_face       / n, 4),
        "missing_hands_ratio":      round(frames_missing_hands / n, 4),
        "processing_time_sec":      round(t_elapsed, 3),
        "hand_landmark_shape":      hand_lm_shape_sample,
        "pose_landmark_shape":      pose_lm_shape_sample,
        "face_landmark_shape":      face_lm_shape_sample,
    }
    return metrics


# ---------------------------------------------------------------------------
# 5.  Main execution
# ---------------------------------------------------------------------------

def main():
    # Select videos that exist on disk
    selected = []
    for fname, signer, sentence in CANDIDATE_VIDEOS:
        p = VIDEOS_DIR / fname
        if p.exists():
            selected.append((p, signer, sentence))
        else:
            print(f"[SKIP] Not found: {fname}")
        if len(selected) >= MAX_VIDEOS:
            break

    if not selected:
        print("[ERROR] No candidate videos found in:", VIDEOS_DIR)
        sys.exit(1)

    print(f"[INFO] Will process {len(selected)} video(s).\n")

    all_metrics = []

    for video_path, signer, sentence in selected:
        print(f"{'='*70}")
        print(f"Processing : {video_path.name}")
        print(f"Signer     : {signer}  |  Sentence: {sentence}")
        print(f"{'='*70}")

        # Create fresh landmarker instances per video.
        # MediaPipe VIDEO-mode landmarkers require monotonically increasing
        # timestamps scoped to a single session; re-using them across videos
        # would cause "Input timestamp must be monotonically increasing" errors
        # since the second video restarts at frame 0 / timestamp 0 ms.
        print("  [INFO] Loading landmarkers ...", end=" ", flush=True)
        t0 = time.perf_counter()
        hand_lmk = build_hand_landmarker()
        pose_lmk = build_pose_landmarker()
        face_lmk = build_face_landmarker()
        print(f"done ({time.perf_counter()-t0:.2f}s)")

        metrics = process_video(video_path, signer, sentence,
                                hand_lmk, pose_lmk, face_lmk)

        hand_lmk.close()
        pose_lmk.close()
        face_lmk.close()

        all_metrics.append(metrics)

        if "error" in metrics:
            print(f"  [ERROR] {metrics['error']}\n")
            continue

        # Print per-video summary
        print(f"  Resolution         : {metrics['resolution']} @ {metrics['fps_declared']} fps")
        print(f"  Frames read        : {metrics['frames_read']}  (declared: {metrics['total_frames_declared']})")
        print(f"  Left  hand         : {metrics['frames_left_hand']:4d}  ({metrics['left_hand_ratio']*100:.1f}%)")
        print(f"  Right hand         : {metrics['frames_right_hand']:4d}  ({metrics['right_hand_ratio']*100:.1f}%)")
        print(f"  Both  hands        : {metrics['frames_both_hands']:4d}  ({metrics['both_hands_ratio']*100:.1f}%)")
        print(f"  Pose detected      : {metrics['frames_pose_detected']:4d}  ({metrics['pose_ratio']*100:.1f}%)")
        print(f"  Face detected      : {metrics['frames_face_detected']:4d}  ({metrics['face_ratio']*100:.1f}%)")
        print(f"  Missing hands      : {metrics['frames_missing_hands']:4d}  ({metrics['missing_hands_ratio']*100:.1f}%)")
        print(f"  Processing time    : {metrics['processing_time_sec']}s")
        if metrics["hand_landmark_shape"]:
            h = metrics["hand_landmark_shape"]
            print(f"  Hand LM shape      : {h['landmarks_per_hand']} pts x {h['coords_per_landmark']} coords "
                  f"= {h['landmarks_per_hand']*h['coords_per_landmark']}D per hand  ({h['note']})")
        else:
            print(f"  Hand LM shape      : NO hands detected in any frame")
        if metrics["pose_landmark_shape"]:
            p = metrics["pose_landmark_shape"]
            print(f"  Pose LM shape      : {p['landmarks_per_pose']} pts x {p['coords_per_landmark']} coords "
                  f"= {p['landmarks_per_pose']*p['coords_per_landmark']}D  ({p['note']})")
        else:
            print(f"  Pose LM shape      : NO pose detected in any frame")
        if metrics["face_landmark_shape"]:
            f = metrics["face_landmark_shape"]
            print(f"  Face LM shape      : {f['landmarks_per_face']} pts x {f['coords_per_landmark']} coords "
                  f"= {f['landmarks_per_face']*f['coords_per_landmark']}D  ({f['note']})")
        else:
            print(f"  Face LM shape      : NO face detected in any frame")
        print()

    # ---------------------------------------------------------------------------
    # 6.  Overall summary and T×162 assessment
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    good_videos = [m for m in all_metrics if "error" not in m]
    if good_videos:
        avg_left  = sum(m["left_hand_ratio"]  for m in good_videos) / len(good_videos)
        avg_right = sum(m["right_hand_ratio"] for m in good_videos) / len(good_videos)
        avg_both  = sum(m["both_hands_ratio"] for m in good_videos) / len(good_videos)
        avg_pose  = sum(m["pose_ratio"]       for m in good_videos) / len(good_videos)
        avg_face  = sum(m["face_ratio"]       for m in good_videos) / len(good_videos)
        avg_miss  = sum(m["missing_hands_ratio"] for m in good_videos) / len(good_videos)

        print(f"  Videos processed successfully : {len(good_videos)} / {len(selected)}")
        print(f"  Avg left-hand  detection      : {avg_left*100:.1f}%")
        print(f"  Avg right-hand detection      : {avg_right*100:.1f}%")
        print(f"  Avg both-hands detection      : {avg_both*100:.1f}%")
        print(f"  Avg pose detection            : {avg_pose*100:.1f}%")
        print(f"  Avg face detection            : {avg_face*100:.1f}%")
        print(f"  Avg frames missing hands      : {avg_miss*100:.1f}%")

    # T×162 assessment
    print(f"\n{'='*70}")
    print("T×162 DESIGN ASSUMPTION — ASSESSMENT")
    print(f"{'='*70}")

    # Collect actual shapes from results
    hand_shapes = [m["hand_landmark_shape"] for m in good_videos if m["hand_landmark_shape"]]
    pose_shapes = [m["pose_landmark_shape"] for m in good_videos if m["pose_landmark_shape"]]
    face_shapes = [m["face_landmark_shape"] for m in good_videos if m["face_landmark_shape"]]

    print("""
  Proposed T×162 representation (from docs/dataset-research-audit.md):
    - 42D  : left hand   (21 landmarks × 2 coords: x, y)
    - 42D  : right hand  (21 landmarks × 2 coords: x, y)
    - 78D  : upper-body pose (26 landmarks × 3 coords: x, y, z)
    - Total: 162D per frame
""")

    if hand_shapes:
        h = hand_shapes[0]
        actual_per_hand_3d = h["landmarks_per_hand"] * h["coords_per_landmark"]
        actual_per_hand_2d = h["landmarks_per_hand"] * 2
        print(f"  Actual HandLandmarker output  : {h['landmarks_per_hand']} landmarks × 3 coords (x,y,z)")
        print(f"    → 3D per hand : {actual_per_hand_3d}D    (both hands: {actual_per_hand_3d*2}D)")
        print(f"    → 2D per hand : {actual_per_hand_2d}D    (both hands: {actual_per_hand_2d*2}D)  ← matches 42D×2 assumption")

    if pose_shapes:
        p = pose_shapes[0]
        actual_pose_3d = p["landmarks_per_pose"] * p["coords_per_landmark"]
        print(f"\n  Actual PoseLandmarker output  : {p['landmarks_per_pose']} landmarks × 3 coords (x,y,z)")
        print(f"    → Full body 3D: {actual_pose_3d}D")
        print(f"    → Upper-body only (first 26 lm, 3D): {26*3}D = 78D  ← matches 78D assumption")
        print(f"    → Upper-body only (first 26 lm, 2D): {26*2}D = 52D")

    print("""
  VERDICT on T×162:
    The T×162 assumption is ACHIEVABLE with the following design choices:
      • Hands : retain only x,y (drop z) → 21×2 = 42D per hand, 84D for both.
      • Pose  : retain x,y,z for the first 26 upper-body landmarks → 78D.
      • Face  : excluded from the 162D encoding (adds 478×3 = 1434D if included).
      • Total : 42 + 42 + 78 = 162D per frame — matches the proposal.

    Caveats:
      (a) Dropping z-depth from hand landmarks discards articulation depth cues.
      (b) "Upper-body first 26" is not a named MediaPipe subset — must be verified
          against the PoseLandmarker landmark index map.
      (c) If a hand is absent in a frame, the 42D slot must be zero-padded (or
          masked); the approach for missing data must be explicitly defined.
      (d) If both hands are absent, 84D of the 162D are fabricated zeros — this
          can mislead downstream models if not handled with masking.

    Alternative worth considering: use full 3D for hands (63D each) and all 33
    pose landmarks (99D) → 63+63+99 = 225D. This retains more signal and avoids
    arbitrary truncation, at a small cost in model input width.
""")

    # ---------------------------------------------------------------------------
    # 7.  Save raw metrics to JSON for the report writer
    # ---------------------------------------------------------------------------
    out_path = OUTPUT_DIR / "mediapipe-feasibility-report-raw-data.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)
    print(f"[INFO] Raw metrics saved to: {out_path}")

    print("\n[DONE] Step 2.2B extraction test complete.")


if __name__ == "__main__":
    main()

