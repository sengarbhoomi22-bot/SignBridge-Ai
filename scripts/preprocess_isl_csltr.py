"""
SignBridge AI — Step 2.3B: Preprocessing Prototype Pipeline
============================================================
Implements a reproducible, modular preprocessing prototype for ISL-CSLTR videos:
- Discovers source videos and reconciles sentence and signer metadata.
- Ingests at native 25.0 FPS with duration bounding (30 <= T <= 160).
- Extracts MediaPipe HandLandmarker and PoseLandmarker features.
- Implements decoupled 2D torso-centering normalization with wrist-relative depth.
- Enforces normalization validity tracking with configurable maximum stale gap (default: 5 frames).
- Supports four candidate feature topologies:
    Candidate A: Pure 2D        (154D: 42 L + 42 R + 66 Pose + 4 Indicators)
    Candidate B: Full 3D        (229D: 63 L + 63 R + 99 Pose + 4 Indicators)
    Candidate C: 3D Upper Pose  (166D: 63 L + 63 R + 36 Pose + 4 Indicators)
    Candidate D: 2.5D Hybrid    (196D: 63 L + 63 R + 66 Pose + 4 Indicators)
- Enforces zero-fabrication contract: undetected articulators are set strictly to 0.0.
- Decouples 4 presence/validity indicators: [m_left, m_right, m_pose, m_norm].
- Enforces 7 automated quality checks (QC-1 through QC-7).
- Serializes per-clip compressed NumPy archives (.npz) and master manifest.csv.

Usage:
    python scripts/preprocess_isl_csltr.py [--topology candidate_b_3d] [--limit 5] [--overwrite]

Design reference:
    docs/preprocessing-design.md
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "isl-csltr"
MODELS_DIR = RAW_DIR / "models"
SAMPLE_VIDEOS_DIR = RAW_DIR / "sample_videos"
METADATA_DIR = RAW_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed" / "isl_csltr_tensors_v1"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
GLOSS_CSV_PATH = METADATA_DIR / "ISL Corpus sign glosses.csv"
CORPUS_DETAILS_PATH = METADATA_DIR / "ISL_CSLRT_Corpus details.xlsx"

# ---------------------------------------------------------------------------
# Preprocessing Parameters (Provisional Constants)
# ---------------------------------------------------------------------------
FPS_EXPECTED = 25.0
MIN_FRAMES_BOUND = 30     # 1.2s at 25 FPS
MAX_FRAMES_BOUND = 160    # 6.4s at 25 FPS
MIN_POSE_RATIO = 0.80     # QC Gate 3 threshold
MIN_HAND_ACTIVITY = 0.05  # QC Warning threshold for passive hands
MAX_STALE_NORMALIZATION_GAP = 5  # Provisional max gap (frames) to re-use previous anchor
HIP_SCALING_HEURISTIC = 1.25     # Provisional anthropometric ratio L_shoulder / L_hip

# Pose landmark indices for Candidate C (12 key upper-body landmarks)
# 0: Nose, 7: L Ear, 8: R Ear, 11: L Shoulder, 12: R Shoulder, 13: L Elbow,
# 14: R Elbow, 15: L Wrist, 16: R Wrist, 23: L Hip, 24: R Hip, 0/Neck Proxy
UPPER_POSE_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 0]

TOPOLOGY_DIMS = {
    "candidate_a_2d": 154,        # 42 + 42 + 66 + 4
    "candidate_b_3d": 229,        # 63 + 63 + 99 + 4
    "candidate_c_upper_3d": 166,  # 63 + 63 + 36 + 4
    "candidate_d_hybrid": 196,    # 63 + 63 + 66 + 4
}


# ---------------------------------------------------------------------------
# Metadata & Video Discovery
# ---------------------------------------------------------------------------
def load_gloss_mapping(csv_path: Path) -> Dict[str, str]:
    """Load sentence-to-gloss mapping from CSV."""
    mapping = {}
    if not csv_path.exists():
        return mapping
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                key = row[0].strip().lower()
                gloss = row[1].strip()
                mapping[key] = gloss
    return mapping


def discover_sample_videos(sample_dir: Path, gloss_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Discover sample videos in data/raw/isl-csltr/sample_videos.
    Associates each video with verified sentence and signer metadata where available.
    """
    samples = []
    # Known empirical mappings from Steps 2.2A, 2.3A, and 2.4B audits:
    KNOWN_CATALOG = {
        "free (2).mp4": {
            "sentence_text": "are you free today",
            "signer_id": "4",           # Frame stem match: Frames_Sentence_Level/are you free today/4/free (2) 01.jpg
            "signer_type": "volunteer",
            "signer_verified": True,
            "split": "train",           # Signer 4 is in Training (Signers 1-5)
        },
        "are you free today.mp4": {
            "sentence_text": "are you free today",
            "signer_id": "1",           # Frame stem match: Frames_Sentence_Level/are you free today/1/are you free today 01.jpg
            "signer_type": "native",
            "signer_verified": True,
            "split": "train",           # Signer 1 is in Training (Signers 1-5)
        },
        "are you free today (2).mp4": {
            "sentence_text": "are you free today",
            "signer_id": "2",           # Empirically verified in Step 2.4B:
                                        #   Pixel MAE = 0.42% vs Frames_Sentence_Level/are you free today/2/*.jpg
                                        #   Rotated shoulder angle: +0.3° (horizontal after 90° CW correction)
            "signer_type": "native",
            "signer_verified": True,
            "split": "train",           # Signer 2 is in Training (Signers 1-5)
        },
        "mvi_6209.mp4": {
            "sentence_text": "are you hiding something",
            "signer_id": "7",           # Frame stem match: Frames_Sentence_Level/are you hiding something/7/MVI_6209 01.jpg
            "signer_type": "volunteer",
            "signer_verified": True,
            "split": "test",            # Signer 7 is held out as final Test
        },
        "bring water for me (1).mp4": {
            "sentence_text": "bring water for me",
            "signer_id": "unverified",
            "signer_type": "unknown",
            "signer_verified": False,   # Frame names in catalog are generic (1).jpg; no video stem match
            "split": "quarantine",      # Strictly quarantined to prevent train/val/test leakage
        },
        "can you repeat that please (3).mp4": {
            "sentence_text": "can you repeat that please",
            "signer_id": "unverified",
            "signer_type": "unknown",
            "signer_verified": False,   # Frame names in catalog are generic (1).jpg; no video stem match
            "split": "quarantine",      # Strictly quarantined to prevent train/val/test leakage
        },
    }

    if not sample_dir.exists():
        return samples

    for p in sorted(sample_dir.glob("*.mp4")) + sorted(sample_dir.glob("*.MP4")):
        fname_lower = p.name.lower()
        info = KNOWN_CATALOG.get(fname_lower, {
            "sentence_text": p.stem.lower(),
            "signer_id": "unverified",
            "signer_type": "unknown",
            "signer_verified": False,
            "split": "quarantine",
        })

        # Safeguard: Any unverified signer MUST be routed to quarantine
        assigned_split = info["split"] if info["signer_verified"] else "quarantine"

        sent_key = info["sentence_text"].strip().lower()
        gloss = gloss_map.get(sent_key, "UNKNOWN_GLOSS")

        samples.append({
            "path": p,
            "filename": p.name,
            "clip_id": p.stem.replace(" ", "_").replace("(", "").replace(")", "").lower(),
            "sentence_text": info["sentence_text"],
            "gloss_text": gloss,
            "signer_id": info["signer_id"],
            "signer_type": info["signer_type"],
            "signer_identity_verified": info["signer_verified"],
            "split": assigned_split,
        })

    return samples


# ---------------------------------------------------------------------------
# Coordinate Normalization Logic
# ---------------------------------------------------------------------------
def compute_normalization_anchor(
    pose_landmarks: Optional[List[Any]],
    prev_anchor: Optional[Tuple[Tuple[float, float], float]],
    stale_gap_count: int,
    max_gap: int = MAX_STALE_NORMALIZATION_GAP,
    hip_scale_heuristic: float = HIP_SCALING_HEURISTIC,
) -> Tuple[Tuple[float, float], float, float, int, int]:
    """
    Computes 2D torso anchor and scale factor with explicit fallback safeguards.
    Returns:
        (center_2d, scale_2d, norm_valid_flag, ref_code, updated_stale_gap)

    ref_codes:
        0: Shoulders (landmarks 11 & 12) valid
        1: Hips fallback (landmarks 23 & 24)
        2: Temporal reference fallback (gap <= max_gap)
        3: Frame-center fallback / expired (gap > max_gap or anchors unavailable)
    """
    # 1. Try Shoulders (11 & 12)
    if pose_landmarks and len(pose_landmarks) > 12:
        p11 = pose_landmarks[11]
        p12 = pose_landmarks[12]
        # Check presence and minimum physical distance
        dist = math.sqrt((p11.x - p12.x) ** 2 + (p11.y - p12.y) ** 2)
        if dist >= 0.05:
            cx = (p11.x + p12.x) / 2.0
            cy = (p11.y + p12.y) / 2.0
            return (cx, cy), dist, 1.0, 0, 0

    # 2. Try Hips (23 & 24) fallback
    if pose_landmarks and len(pose_landmarks) > 24:
        p23 = pose_landmarks[23]
        p24 = pose_landmarks[24]
        hip_dist = math.sqrt((p23.x - p24.x) ** 2 + (p23.y - p24.y) ** 2)
        if hip_dist >= 0.04:
            cx = (p23.x + p24.x) / 2.0
            cy = (p23.y + p24.y) / 2.0
            # Apply unvalidated heuristic ratio
            estimated_shoulder_scale = hip_dist * hip_scale_heuristic
            return (cx, cy), estimated_shoulder_scale, 0.0, 1, 0

    # 3. Try Temporal Reference Fallback (bounded by max_gap)
    if prev_anchor is not None and stale_gap_count < max_gap:
        prev_center, prev_scale = prev_anchor
        return prev_center, prev_scale, 0.0, 2, stale_gap_count + 1

    # 4. Fallback to Frame-Center (invalid / expired)
    return (0.5, 0.5), 0.20, 0.0, 3, stale_gap_count + 1


def normalize_coordinates(
    pose_lms: Optional[List[Any]],
    left_hand_lms: Optional[List[Any]],
    right_hand_lms: Optional[List[Any]],
    center_2d: Tuple[float, float],
    scale_2d: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies decoupled 2D torso-centering and scale invariance:
    - Pose: (x - cx)/s, (y - cy)/s, z/s
    - Hands: (x - cx)/s, (y - cy)/s, z/s (wrist-relative z is scaled, not translated)
    - Missing landmarks are set strictly to 0.0.
    """
    cx, cy = center_2d
    s = max(scale_2d, 1e-4)

    # Pose: 33 x 3
    pose_arr = np.zeros((33, 3), dtype=np.float32)
    if pose_lms is not None:
        for i, lm in enumerate(pose_lms[:33]):
            pose_arr[i, 0] = (lm.x - cx) / s
            pose_arr[i, 1] = (lm.y - cy) / s
            pose_arr[i, 2] = lm.z / s

    # Left Hand: 21 x 3
    left_arr = np.zeros((21, 3), dtype=np.float32)
    if left_hand_lms is not None:
        for i, lm in enumerate(left_hand_lms[:21]):
            left_arr[i, 0] = (lm.x - cx) / s
            left_arr[i, 1] = (lm.y - cy) / s
            left_arr[i, 2] = lm.z / s  # wrist-relative z scaled

    # Right Hand: 21 x 3
    right_arr = np.zeros((21, 3), dtype=np.float32)
    if right_hand_lms is not None:
        for i, lm in enumerate(right_hand_lms[:21]):
            right_arr[i, 0] = (lm.x - cx) / s
            right_arr[i, 1] = (lm.y - cy) / s
            right_arr[i, 2] = lm.z / s  # wrist-relative z scaled

    return pose_arr, left_arr, right_arr


# ---------------------------------------------------------------------------
# Topology Assembly
# ---------------------------------------------------------------------------
def assemble_topology_vector(
    pose_arr: np.ndarray,          # shape: (33, 3)
    left_arr: np.ndarray,          # shape: (21, 3)
    right_arr: np.ndarray,         # shape: (21, 3)
    indicators: np.ndarray,        # shape: (4,) [m_left, m_right, m_pose, m_norm]
    topology: str = "candidate_b_3d",
) -> np.ndarray:
    """
    Assembles a 1D frame vector for the chosen topology.
    """
    if topology == "candidate_a_2d":
        # Left 2D (42) + Right 2D (42) + Pose 2D (66) + Indicators (4) = 154
        l_2d = left_arr[:, :2].flatten()
        r_2d = right_arr[:, :2].flatten()
        p_2d = pose_arr[:, :2].flatten()
        return np.concatenate([l_2d, r_2d, p_2d, indicators]).astype(np.float32)

    elif topology == "candidate_b_3d":
        # Left 3D (63) + Right 3D (63) + Pose 3D (99) + Indicators (4) = 229
        l_3d = left_arr.flatten()
        r_3d = right_arr.flatten()
        p_3d = pose_arr.flatten()
        return np.concatenate([l_3d, r_3d, p_3d, indicators]).astype(np.float32)

    elif topology == "candidate_c_upper_3d":
        # Left 3D (63) + Right 3D (63) + Upper Pose 3D (12 * 3 = 36) + Indicators (4) = 166
        l_3d = left_arr.flatten()
        r_3d = right_arr.flatten()
        p_upper = pose_arr[UPPER_POSE_INDICES, :].flatten()
        return np.concatenate([l_3d, r_3d, p_upper, indicators]).astype(np.float32)

    elif topology == "candidate_d_hybrid":
        # Left 3D (63) + Right 3D (63) + Pose 2D (66) + Indicators (4) = 196
        l_3d = left_arr.flatten()
        r_3d = right_arr.flatten()
        p_2d = pose_arr[:, :2].flatten()
        return np.concatenate([l_3d, r_3d, p_2d, indicators]).astype(np.float32)

    else:
        raise ValueError(f"Unknown topology: {topology}. Choose from {list(TOPOLOGY_DIMS.keys())}")


# ---------------------------------------------------------------------------
# MediaPipe Model Loader
# ---------------------------------------------------------------------------
def create_landmarkers(hand_model: Path, pose_model: Path):
    """
    Instantiates fresh HandLandmarker and PoseLandmarker sessions.
    Must be called per video clip to maintain monotonic timestamps.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    # Hand Landmarker
    h_base = mp_python.BaseOptions(model_asset_path=str(hand_model))
    h_opts = mp_vision.HandLandmarkerOptions(
        base_options=h_base,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    hand_lmk = mp_vision.HandLandmarker.create_from_options(h_opts)

    # Pose Landmarker (Lite)
    p_base = mp_python.BaseOptions(model_asset_path=str(pose_model))
    p_opts = mp_vision.PoseLandmarkerOptions(
        base_options=p_base,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.50,
        min_pose_presence_confidence=0.50,
        min_tracking_confidence=0.50,
        output_segmentation_masks=False,
    )
    pose_lmk = mp_vision.PoseLandmarker.create_from_options(p_opts)

    return hand_lmk, pose_lmk


# ---------------------------------------------------------------------------
# Single Video Preprocessing Routine
# ---------------------------------------------------------------------------
def preprocess_video(
    video_item: Dict[str, Any],
    hand_model: Path,
    pose_model: Path,
    output_dir: Path,
    topology: str = "candidate_b_3d",
    max_gap: int = MAX_STALE_NORMALIZATION_GAP,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Processes a single video file through extraction, normalization, and packaging.
    """
    import mediapipe as mp

    clip_id = video_item["clip_id"]
    split = video_item["split"]
    target_split_dir = output_dir / split
    target_split_dir.mkdir(parents=True, exist_ok=True)
    npz_out_path = target_split_dir / f"{clip_id}.npz"

    # Check resumption
    if npz_out_path.exists() and not overwrite:
        print(f"  [SKIP] Output already exists: {npz_out_path.name}")
        return {
            "status": "SKIPPED_EXISTING",
            "clip_id": clip_id,
            "npz_path": str(npz_out_path),
            "qc_pass": True,
        }

    vpath = video_item["path"]
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return {
            "status": "ERROR_CORRUPT_VIDEO",
            "clip_id": clip_id,
            "qc_pass": False,
            "qc_reason": "Could not open video with OpenCV",
        }

    fps_declared = cap.get(cv2.CAP_PROP_FPS) or FPS_EXPECTED
    frames_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------------------------------------------------------
    # Orientation Correction (Step 2.4B)
    # Only applied when ALL three conditions are met:
    #   1. signer_identity_verified == True
    #   2. signer_id in {"1", "2"}  (empirically confirmed native sideways recorders)
    #   3. orig_width > orig_height  (video is in landscape / sideways orientation)
    # Rotation is 90° clockwise, applied exactly once per frame before MediaPipe.
    # Unverified and quarantined clips are NEVER rotated.
    # ---------------------------------------------------------------------------
    signer_id_str = str(video_item.get("signer_id", ""))
    signer_verified = bool(video_item.get("signer_identity_verified", False))

    apply_rotation = False
    applied_rotation_deg = 0
    orientation_warning = None

    if signer_verified and signer_id_str in ("1", "2"):
        if orig_width > orig_height:
            # Landscape native-signer video: rotate 90° CW to restore upright orientation
            apply_rotation = True
            applied_rotation_deg = 90
            proc_width = orig_height
            proc_height = orig_width
        else:
            # Native-signer video is already portrait — unexpected; skip rotation and warn
            orientation_warning = (
                f"Native signer {signer_id_str} video is not landscape "
                f"({orig_width}x{orig_height}); expected width > height. "
                f"Rotation skipped — verify camera setup manually."
            )
            proc_width = orig_width
            proc_height = orig_height
    else:
        # Volunteers (Signers 3–7), unverified, and quarantined clips: no rotation
        proc_width = orig_width
        proc_height = orig_height

    # Initialize fresh MediaPipe landmarker sessions
    t_extract_start = time.perf_counter()
    hand_lmk, pose_lmk = create_landmarkers(hand_model, pose_model)

    feature_frames = []
    presence_masks = []
    ref_codes = []

    frames_read = 0
    left_hand_detected_count = 0
    right_hand_detected_count = 0
    pose_detected_count = 0
    norm_valid_count = 0

    prev_anchor = None
    stale_gap_count = 0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Apply orientation correction exactly once per frame (before any other processing)
        if apply_rotation:
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        timestamp_ms = int(frame_idx * (1000.0 / fps_declared))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 1. Pose detection
        try:
            pose_res = pose_lmk.detect_for_video(mp_img, timestamp_ms)
        except Exception as e:
            pose_res = None

        pose_lms = None
        m_pose = 0.0
        if pose_res and pose_res.pose_landmarks:
            pose_lms = pose_res.pose_landmarks[0]
            m_pose = 1.0
            pose_detected_count += 1

        # 2. Hand detection
        try:
            hand_res = hand_lmk.detect_for_video(mp_img, timestamp_ms)
        except Exception as e:
            hand_res = None

        left_lms = None
        right_lms = None
        m_left = 0.0
        m_right = 0.0

        if hand_res and hand_res.hand_landmarks:
            for lms, handedness_list in zip(hand_res.hand_landmarks, hand_res.handedness):
                if handedness_list:
                    label = handedness_list[0].category_name
                    if label == "Left":
                        left_lms = lms
                        m_left = 1.0
                    elif label == "Right":
                        right_lms = lms
                        m_right = 1.0

        if m_left > 0:
            left_hand_detected_count += 1
        if m_right > 0:
            right_hand_detected_count += 1

        # 3. Normalization anchor calculation (with safeguards)
        center_2d, scale_2d, m_norm, r_code, stale_gap_count = compute_normalization_anchor(
            pose_lms, prev_anchor, stale_gap_count, max_gap
        )
        if m_norm > 0:
            norm_valid_count += 1
            prev_anchor = (center_2d, scale_2d)

        ref_codes.append(r_code)

        # 4. Decoupled coordinate normalization
        norm_pose, norm_left, norm_right = normalize_coordinates(
            pose_lms, left_lms, right_lms, center_2d, scale_2d
        )

        # 5. Assemble indicator block: [m_left, m_right, m_pose, m_norm]
        indicators = np.array([m_left, m_right, m_pose, m_norm], dtype=np.float32)

        # 6. Assemble candidate topology vector
        feat_vec = assemble_topology_vector(
            norm_pose, norm_left, norm_right, indicators, topology=topology
        )

        feature_frames.append(feat_vec)
        presence_masks.append(indicators)

        frame_idx += 1
        frames_read += 1

    cap.release()
    hand_lmk.close()
    pose_lmk.close()
    t_extract_elapsed = time.perf_counter() - t_extract_start

    # Quality Checks
    qc_pass = True
    qc_failures = []
    qc_warnings = []

    # QC-1: File readability & frames (Technical Validity)
    if frames_read == 0:
        qc_pass = False
        qc_failures.append("QC-1: 0 frames read from video")

    # QC-2: Duration bounds (Coverage / Boundary Validity)
    if frames_read < MIN_FRAMES_BOUND or frames_read > MAX_FRAMES_BOUND:
        qc_pass = False
        qc_failures.append(f"QC-2: Duration out of bounds ({frames_read} frames)")

    # QC-3: Pose ratio (Landmark Coverage)
    pose_ratio = pose_detected_count / max(frames_read, 1)
    if pose_ratio < MIN_POSE_RATIO:
        qc_pass = False
        qc_failures.append(f"QC-3: Pose ratio {pose_ratio:.2f} < {MIN_POSE_RATIO}")

    # Convert to NumPy
    features_arr = np.array(feature_frames, dtype=np.float32)
    masks_arr = np.array(presence_masks, dtype=np.float32)

    # QC-4: Non-finite / NaN check (Technical Validity)
    if not np.isfinite(features_arr).all():
        qc_pass = False
        qc_failures.append("QC-4: Non-finite/NaN values detected in feature tensor")

    # QC-5: Normalization validity check (Stability)
    norm_valid_ratio = norm_valid_count / max(frames_read, 1)
    expired_ref_ratio = sum(1 for r in ref_codes if r == 3) / max(frames_read, 1)
    if expired_ref_ratio > 0.20:
        qc_pass = False
        qc_failures.append(f"QC-5: Expired/Frame-center fallback ratio {expired_ref_ratio:.2f} > 0.20")

    # QC-6: Hand Landmark Coverage & Asymmetry
    left_ratio = left_hand_detected_count / max(frames_read, 1)
    right_ratio = right_hand_detected_count / max(frames_read, 1)
    max_hand_ratio = max(left_ratio, right_ratio)
    min_hand_ratio = min(left_ratio, right_ratio)

    if max_hand_ratio == 0.0:
        qc_pass = False
        qc_failures.append("QC-6_FAIL: Zero hand detections across entire video")
    elif max_hand_ratio < 0.20:
        qc_warnings.append(f"QC-W_LOW_HANDS: Max hand detection ratio {max_hand_ratio:.1%} < 20%")
    
    if min_hand_ratio < 0.10 and max_hand_ratio >= 0.20:
        qc_warnings.append(f"QC-W_ASYM: Asymmetric/one-handed detection (L={left_ratio:.1%}, R={right_ratio:.1%})")

    # Training suitability: Requires technical pass, verified signer identity, and train split
    # Note: signer_verified was already resolved above during orientation detection
    suitability_for_training = bool(qc_pass and signer_verified and split == "train")

    # Append orientation warning to qc_warnings if set
    if orientation_warning:
        qc_warnings.append(f"ORIENTATION_WARNING: {orientation_warning}")

    # Determine overall status string
    if not qc_pass:
        overall_status = "REJECT" if (frames_read == 0 or not np.isfinite(features_arr).all()) else "FAIL"
    elif not signer_verified:
        overall_status = "QUARANTINE_UNVERIFIED"
    elif qc_warnings:
        overall_status = "PASS_WITH_WARNING"
    else:
        overall_status = "PASS"

    # Metadata payload
    expected_dim = TOPOLOGY_DIMS[topology]
    metadata_payload = {
        "clip_id": clip_id,
        "video_filename": video_item["filename"],
        "sentence_text": video_item["sentence_text"],
        "gloss_text": video_item["gloss_text"],
        "signer_id": video_item["signer_id"],
        "signer_type": video_item["signer_type"],
        "signer_identity_verified": signer_verified,
        "split": split,
        # Orientation correction fields (Step 2.4B)
        "original_dimensions": f"{orig_width}x{orig_height}",
        "processed_dimensions": f"{proc_width}x{proc_height}",
        "applied_rotation_deg": applied_rotation_deg,
        "fps_declared": fps_declared,
        "frames_extracted": frames_read,
        "expected_dim": expected_dim,
        "actual_dim": int(features_arr.shape[1]) if len(features_arr.shape) > 1 else 0,
        "topology": topology,
        "left_hand_ratio": round(left_ratio, 4),
        "right_hand_ratio": round(right_ratio, 4),
        "pose_ratio": round(pose_ratio, 4),
        "norm_valid_ratio": round(norm_valid_ratio, 4),
        "expired_ref_ratio": round(expired_ref_ratio, 4),
        "extraction_time_sec": round(t_extract_elapsed, 2),
        "qc_pass": qc_pass,
        "qc_status": overall_status,
        "qc_warnings": qc_warnings,
        "qc_failures": qc_failures,
        "suitability_for_training": suitability_for_training,
    }

    # Serialize to compressed .npz archive
    np.savez_compressed(
        npz_out_path,
        features=features_arr,
        presence_mask=masks_arr,
        metadata=json.dumps(metadata_payload)
    )

    return {
        "status": overall_status,
        "clip_id": clip_id,
        "npz_path": str(npz_out_path),
        "output_shape": features_arr.shape,
        "qc_pass": qc_pass,
        "qc_warnings": qc_warnings,
        "qc_failures": qc_failures,
        "metadata": metadata_payload,
    }


# ---------------------------------------------------------------------------
# Batch Driver & Manifest Generation
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(
    sample_dir: Path = SAMPLE_VIDEOS_DIR,
    output_dir: Path = PROCESSED_DIR,
    topology: str = "candidate_b_3d",
    limit: Optional[int] = 5,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Executes the preprocessing prototype over discovered sample videos.
    """
    print("=" * 70)
    print("SignBridge AI — Step 2.3B: Preprocessing Prototype Pipeline")
    print("=" * 70)
    print(f"Topology Selected : {topology} ({TOPOLOGY_DIMS.get(topology, 'Unknown')}D)")
    print(f"Input Directory   : {sample_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Sample Limit      : {limit} videos")
    print(f"Max Fallback Gap  : {MAX_STALE_NORMALIZATION_GAP} frames")
    print()

    # 1. Load Gloss Mappings
    gloss_map = load_gloss_mapping(GLOSS_CSV_PATH)
    print(f"[INFO] Loaded {len(gloss_map)} sentence-to-gloss mappings.")

    # 2. Discover Videos
    videos = discover_sample_videos(sample_dir, gloss_map)
    if limit is not None:
        videos = videos[:limit]

    print(f"[INFO] Discovered {len(videos)} candidate video(s) for processing.\n")

    results = []
    manifest_rows = []

    for idx, vitem in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Processing: {vitem['filename']}")
        print(f"  Sentence : '{vitem['sentence_text']}' -> Gloss: '{vitem['gloss_text']}'")
        print(f"  Signer ID: {vitem['signer_id']} (Verified: {vitem['signer_identity_verified']}) | Split: {vitem['split']}")

        res = preprocess_video(
            video_item=vitem,
            hand_model=HAND_MODEL_PATH,
            pose_model=POSE_MODEL_PATH,
            output_dir=output_dir,
            topology=topology,
            overwrite=overwrite,
        )
        results.append(res)

        if "metadata" in res:
            meta = res["metadata"]
            rot_note = f" | Rotation: {meta['applied_rotation_deg']}°" if meta.get("applied_rotation_deg", 0) != 0 else ""
            print(f"  Result   : {res['status']} | Shape: {res['output_shape']} | Pose: {meta['pose_ratio']*100:.1f}% | Hands: L={meta['left_hand_ratio']*100:.1f}%, R={meta['right_hand_ratio']*100:.1f}%{rot_note}")
            if res.get("qc_warnings"):
                print(f"  QC Warns : {res['qc_warnings']}")
            if not res["qc_pass"]:
                print(f"  QC Flags : {res['qc_failures']}")

            manifest_rows.append({
                "clip_id": meta["clip_id"],
                "video_filename": meta["video_filename"],
                "sentence_text": meta["sentence_text"],
                "gloss_text": meta["gloss_text"],
                "signer_id": meta["signer_id"],
                "signer_type": meta["signer_type"],
                "signer_identity_verified": meta["signer_identity_verified"],
                "split": meta["split"],
                "original_dimensions": meta.get("original_dimensions", ""),
                "processed_dimensions": meta.get("processed_dimensions", ""),
                "applied_rotation_deg": meta.get("applied_rotation_deg", 0),
                "frames_extracted": meta["frames_extracted"],
                "feature_dim": meta["actual_dim"],
                "topology": meta["topology"],
                "left_hand_ratio": meta["left_hand_ratio"],
                "right_hand_ratio": meta["right_hand_ratio"],
                "pose_ratio": meta["pose_ratio"],
                "norm_valid_ratio": meta["norm_valid_ratio"],
                "qc_status": meta["qc_status"],
                "qc_warnings": "; ".join(meta["qc_warnings"]),
                "qc_failures": "; ".join(meta["qc_failures"]),
                "suitability_for_training": meta["suitability_for_training"],
                "npz_path": res["npz_path"],
            })
        print()

    # 3. Write Master Manifest CSV
    manifest_csv_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"[INFO] Master manifest saved to: {manifest_csv_path}")

    # Summary Report
    successful = sum(1 for r in results if r.get("qc_pass") is True and r.get("metadata", {}).get("signer_identity_verified") is True)
    quarantined = sum(1 for r in results if r.get("metadata", {}).get("split") == "quarantine" or r.get("qc_pass") is False)
    print("\n" + "=" * 70)
    print(f"PROTOTYPE PREPROCESSING COMPLETE: {successful} Verified Train/Test, {quarantined} Quarantined")
    print("=" * 70)

    return {
        "results": results,
        "manifest_path": str(manifest_csv_path),
        "passed_count": successful,
        "quarantined_count": quarantined,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI ISL-CSLTR Preprocessing Prototype")
    parser.add_argument(
        "--topology",
        choices=list(TOPOLOGY_DIMS.keys()),
        default="candidate_b_3d",
        help="Feature topology candidate to generate (default: candidate_b_3d, 229D)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of sample videos to process (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files",
    )
    args = parser.parse_args()

    run_preprocessing_pipeline(
        topology=args.topology,
        limit=args.limit,
        overwrite=args.overwrite,
    )

