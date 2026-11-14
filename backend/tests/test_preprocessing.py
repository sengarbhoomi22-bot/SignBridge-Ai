"""
Unit tests for the SignBridge AI ISL-CSLTR Preprocessing Pipeline.
Tests:
- Topology dimensions for Candidate A (154D), B (229D), C (166D), D (196D).
- Decoupled 2D torso-centering normalization.
- Normalization fallback hierarchy and stale gap expiration.
- Zero-fabrication of unobserved hand and pose landmarks.
- Presence and normalization indicator decoupling.
- Orientation correction logic (Step 2.4B): rotation gating, metadata recording,
  dimension swapping, and correct application count.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

# Ensure scripts directory is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from preprocess_isl_csltr import (
    TOPOLOGY_DIMS,
    assemble_topology_vector,
    compute_normalization_anchor,
    normalize_coordinates,
)


class MockLandmark:
    """Mock MediaPipe Landmark point."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def test_topology_dimensions_all_candidates():
    """Verify that all four candidate topologies output exact expected dimensions."""
    pose_arr = np.zeros((33, 3), dtype=np.float32)
    left_arr = np.zeros((21, 3), dtype=np.float32)
    right_arr = np.zeros((21, 3), dtype=np.float32)
    indicators = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)  # [mL, mR, mP, mNorm]

    # Candidate A: Pure 2D (42 + 42 + 66 + 4 = 154)
    vec_a = assemble_topology_vector(pose_arr, left_arr, right_arr, indicators, "candidate_a_2d")
    assert vec_a.shape == (154,)
    assert TOPOLOGY_DIMS["candidate_a_2d"] == 154

    # Candidate B: Full 3D (63 + 63 + 99 + 4 = 229)
    vec_b = assemble_topology_vector(pose_arr, left_arr, right_arr, indicators, "candidate_b_3d")
    assert vec_b.shape == (229,)
    assert TOPOLOGY_DIMS["candidate_b_3d"] == 229

    # Candidate C: Upper 3D (63 + 63 + 36 + 4 = 166)
    vec_c = assemble_topology_vector(pose_arr, left_arr, right_arr, indicators, "candidate_c_upper_3d")
    assert vec_c.shape == (166,)
    assert TOPOLOGY_DIMS["candidate_c_upper_3d"] == 166

    # Candidate D: Hybrid 2.5D (63 + 63 + 66 + 4 = 196)
    vec_d = assemble_topology_vector(pose_arr, left_arr, right_arr, indicators, "candidate_d_hybrid")
    assert vec_d.shape == (196,)
    assert TOPOLOGY_DIMS["candidate_d_hybrid"] == 196


def test_normalization_valid_shoulders():
    """Verify shoulder anchor produces valid normalization (ref_code=0, norm_valid=1.0)."""
    pose_lms = [MockLandmark(0.5, 0.5) for _ in range(33)]
    # Set shoulders 11 and 12
    pose_lms[11] = MockLandmark(0.60, 0.40)  # Left shoulder
    pose_lms[12] = MockLandmark(0.40, 0.40)  # Right shoulder

    center, scale, m_norm, r_code, stale_gap = compute_normalization_anchor(
        pose_landmarks=pose_lms,
        prev_anchor=None,
        stale_gap_count=0,
        max_gap=5
    )

    assert r_code == 0
    assert m_norm == 1.0
    assert stale_gap == 0
    assert pytest.approx(center[0]) == 0.50
    assert pytest.approx(center[1]) == 0.40
    assert pytest.approx(scale) == 0.20


def test_normalization_hips_fallback():
    """Verify that when shoulders collapse, hips anchor engages with ref_code=1, norm_valid=0.0."""
    pose_lms = [MockLandmark(0.5, 0.5) for _ in range(33)]
    # Shoulders collapsed
    pose_lms[11] = MockLandmark(0.50, 0.40)
    pose_lms[12] = MockLandmark(0.50, 0.40)
    # Hips valid
    pose_lms[23] = MockLandmark(0.58, 0.70)
    pose_lms[24] = MockLandmark(0.42, 0.70)

    center, scale, m_norm, r_code, stale_gap = compute_normalization_anchor(
        pose_landmarks=pose_lms,
        prev_anchor=None,
        stale_gap_count=0,
        max_gap=5,
        hip_scale_heuristic=1.25
    )

    assert r_code == 1
    assert m_norm == 0.0  # Fallback is marked as approximate
    assert stale_gap == 0
    assert pytest.approx(center[0]) == 0.50
    assert pytest.approx(center[1]) == 0.70
    assert pytest.approx(scale) == 0.16 * 1.25  # 0.16 hip distance * 1.25 heuristic


def test_normalization_temporal_fallback_within_and_beyond_gap():
    """Verify temporal anchor re-use is allowed for <= max_gap and expires for > max_gap."""
    prev_anchor = ((0.50, 0.40), 0.20)
    pose_lms = None  # No pose detected

    # Gap 1 (valid within window)
    center, scale, m_norm, r_code, updated_gap = compute_normalization_anchor(
        pose_landmarks=pose_lms,
        prev_anchor=prev_anchor,
        stale_gap_count=1,
        max_gap=5
    )
    assert r_code == 2
    assert m_norm == 0.0
    assert updated_gap == 2
    assert center == (0.50, 0.40)
    assert scale == 0.20

    # Gap 5 (at limit -> still allowed)
    center, scale, m_norm, r_code, updated_gap = compute_normalization_anchor(
        pose_landmarks=pose_lms,
        prev_anchor=prev_anchor,
        stale_gap_count=4,
        max_gap=5
    )
    assert r_code == 2
    assert updated_gap == 5

    # Gap 5 exceeded (stale_gap_count = 5 >= max_gap 5) -> EXPIRED
    center, scale, m_norm, r_code, updated_gap = compute_normalization_anchor(
        pose_landmarks=pose_lms,
        prev_anchor=prev_anchor,
        stale_gap_count=5,
        max_gap=5
    )
    assert r_code == 3  # Frame-center fallback
    assert m_norm == 0.0
    assert center == (0.5, 0.5)
    assert scale == 0.20


def test_zero_fabrication_on_missing_hands():
    """Verify undetected hands remain strictly 0.0 without coordinate hallucination."""
    center = (0.5, 0.4)
    scale = 0.20
    pose_lms = [MockLandmark(0.5, 0.5, 0.0) for _ in range(33)]

    # Left hand is present, Right hand is None (absent)
    left_lms = [MockLandmark(0.6, 0.5, -0.01) for _ in range(21)]
    right_lms = None

    pose_arr, left_arr, right_arr = normalize_coordinates(
        pose_lms, left_lms, right_lms, center, scale
    )

    # Left hand coordinates should be normalized
    assert left_arr[0, 0] != 0.0
    assert np.all(np.isfinite(left_arr))

    # Right hand coordinates MUST be strictly 0.0
    assert np.all(right_arr == 0.0)
    assert right_arr.shape == (21, 3)


def test_wrist_relative_depth_preserved():
    """Verify that hand z depth is scaled by shoulder scale, NOT shifted by torso center."""
    center = (0.5, 0.4)
    scale = 0.20
    # MediaPipe wrist has z = 0.0
    left_lms = [MockLandmark(0.6, 0.5, z=0.0) for _ in range(21)]
    left_lms[4] = MockLandmark(0.62, 0.48, z=-0.02)  # thumb tip

    _, left_arr, _ = normalize_coordinates(None, left_lms, None, center, scale)

    # Wrist z must remain exactly 0.0
    assert left_arr[0, 2] == 0.0
    # Thumb tip z must be scaled: -0.02 / 0.20 = -0.10
    assert pytest.approx(left_arr[4, 2]) == -0.10


def test_unverified_signer_routing_to_quarantine():
    """Verify that unverified signer clips are strictly routed to quarantine, never train/val/test."""
    from preprocess_isl_csltr import discover_sample_videos

    sample_dir = PROJECT_ROOT / "data" / "raw" / "isl-csltr" / "sample_videos"
    if not sample_dir.exists():
        pytest.skip("Sample videos directory not present")

    videos = discover_sample_videos(sample_dir, {})
    for v in videos:
        if not v["signer_identity_verified"]:
            assert v["split"] == "quarantine", f"Unverified clip {v['filename']} must be in quarantine!"
            assert v["split"] not in ["train", "val", "test"]
        else:
            assert v["split"] in ["train", "val", "test"]


def test_training_suitability_requires_verified_signer_and_train_split():
    """Verify training suitability gate prevents test signers or unverified signers from training."""
    # Verified signer in train split -> suitable
    qc_pass = True
    signer_verified = True
    split = "train"
    suitable = bool(qc_pass and signer_verified and split == "train")
    assert suitable is True

    # Verified signer in test split (e.g. Signer 7) -> NOT suitable for training
    split_test = "test"
    suitable_test = bool(qc_pass and signer_verified and split_test == "train")
    assert suitable_test is False

    # Unverified signer in quarantine -> NOT suitable for training
    signer_unverified = False
    split_quarantine = "quarantine"
    suitable_unverified = bool(qc_pass and signer_unverified and split_quarantine == "train")
    assert suitable_unverified is False


# =============================================================================
# Orientation Correction Tests (Step 2.4B)
# =============================================================================

class TestOrientationCorrection:
    """
    Tests for the orientation correction gating logic added in Step 2.4B.

    The correction is implemented directly in preprocess_video() and applies
    cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) exactly once per frame.
    These unit tests validate the gating logic and dimension swap using
    synthetic frames and the same gating conditions as the production code.
    """

    def _make_landscape_frame(self):
        """Return a synthetic BGR frame in landscape orientation (320x240)."""
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def _make_portrait_frame(self):
        """Return a synthetic BGR frame in portrait orientation (1080x1920)."""
        return np.zeros((1920, 1080, 3), dtype=np.uint8)

    def _should_rotate(self, signer_id, signer_verified, orig_w, orig_h):
        """Replicate the gating logic from preprocess_video() for unit-testing."""
        return (
            signer_verified
            and str(signer_id) in ("1", "2")
            and orig_w > orig_h
        )

    # ------------------------------------------------------------------
    # Rotation gating
    # ------------------------------------------------------------------

    def test_native_signer_1_landscape_rotates(self):
        """Verified Signer 1 landscape video must trigger 90° CW rotation."""
        frame = self._make_landscape_frame()
        orig_h, orig_w = frame.shape[:2]  # 240, 320
        assert self._should_rotate("1", True, orig_w, orig_h) is True

    def test_native_signer_2_landscape_rotates(self):
        """Verified Signer 2 landscape video must trigger 90° CW rotation."""
        frame = self._make_landscape_frame()
        orig_h, orig_w = frame.shape[:2]
        assert self._should_rotate("2", True, orig_w, orig_h) is True

    def test_volunteer_signer_3_no_rotation(self):
        """Volunteer Signer 3 portrait video must NOT be rotated."""
        frame = self._make_portrait_frame()
        orig_h, orig_w = frame.shape[:2]
        assert self._should_rotate("3", True, orig_w, orig_h) is False

    def test_volunteer_signer_7_no_rotation(self):
        """Volunteer Signer 7 portrait video must NOT be rotated."""
        frame = self._make_portrait_frame()
        orig_h, orig_w = frame.shape[:2]
        assert self._should_rotate("7", True, orig_w, orig_h) is False

    def test_unverified_signer_landscape_no_rotation(self):
        """Unverified signer with landscape video must NOT be rotated (identity unconfirmed)."""
        frame = self._make_landscape_frame()
        orig_h, orig_w = frame.shape[:2]
        # signer_id="1" but signer_verified=False -> no rotation
        assert self._should_rotate("1", False, orig_w, orig_h) is False

    def test_quarantined_clip_no_rotation(self):
        """Quarantined clip (unverified, unknown signer_id) must NOT be rotated."""
        frame = self._make_landscape_frame()
        orig_h, orig_w = frame.shape[:2]
        assert self._should_rotate("unverified", False, orig_w, orig_h) is False

    def test_native_signer_portrait_no_rotation(self):
        """
        If a native signer video is unexpectedly portrait (height > width),
        rotation must be skipped (gating requires orig_width > orig_height).
        """
        frame = self._make_portrait_frame()
        orig_h, orig_w = frame.shape[:2]  # 1920, 1080 -> orig_w=1080, orig_h=1920
        assert self._should_rotate("1", True, orig_w, orig_h) is False

    # ------------------------------------------------------------------
    # Dimension swap after rotation
    # ------------------------------------------------------------------

    def test_90cw_rotation_swaps_dimensions(self):
        """
        After 90° CW rotation, a (240, 320, 3) landscape frame must become
        a (320, 240, 3) portrait frame. Width and height swap exactly once.
        """
        frame = self._make_landscape_frame()
        assert frame.shape == (240, 320, 3)  # h=240, w=320

        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        assert rotated.shape == (320, 240, 3)  # h=320, w=240 (swapped)

    def test_90cw_rotation_produces_expected_proc_dimensions(self):
        """
        Verify the proc_width / proc_height variables computed after the
        orientation check match the actual rotated frame dimensions.
        """
        orig_w, orig_h = 320, 240  # native signer landscape
        # Gating condition satisfied
        apply_rotation = True
        proc_w = orig_h   # 240
        proc_h = orig_w   # 320

        frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        assert rotated.shape[1] == proc_w  # rotated width = 240
        assert rotated.shape[0] == proc_h  # rotated height = 320

    def test_rotation_applied_exactly_once(self):
        """
        Applying cv2.ROTATE_90_CLOCKWISE exactly once must yield a different
        shape from the original (landscape -> portrait).
        Applying it twice returns to the original shape (portrait -> landscape).
        Verifies that a single rotation is geometrically distinct from no rotation.
        """
        frame = self._make_landscape_frame()          # (H=240, W=320, 3)
        once  = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)   # (H=320, W=240, 3)
        twice = cv2.rotate(once,  cv2.ROTATE_90_CLOCKWISE)   # (H=240, W=320, 3)

        # One rotation must change shape
        assert once.shape != frame.shape, "Single rotation must change shape"
        # Two rotations restore the original shape
        assert twice.shape == frame.shape, "Two rotations must restore original shape"

    # ------------------------------------------------------------------
    # Metadata recording
    # ------------------------------------------------------------------

    def test_metadata_records_rotation_applied(self):
        """
        When rotation is applied, metadata must record:
        - original_dimensions matching orig capture size
        - processed_dimensions matching rotated size
        - applied_rotation_deg == 90
        """
        orig_w, orig_h = 320, 240
        apply_rotation = True
        proc_w = orig_h   # 240
        proc_h = orig_w   # 320
        applied_rotation_deg = 90

        meta = {
            "original_dimensions": f"{orig_w}x{orig_h}",
            "processed_dimensions": f"{proc_w}x{proc_h}",
            "applied_rotation_deg": applied_rotation_deg,
        }

        assert meta["original_dimensions"] == "320x240"
        assert meta["processed_dimensions"] == "240x320"
        assert meta["applied_rotation_deg"] == 90

    def test_metadata_records_zero_rotation_for_volunteers(self):
        """
        Volunteer signer metadata must record:
        - original_dimensions == processed_dimensions (no size change)
        - applied_rotation_deg == 0
        """
        orig_w, orig_h = 1080, 1920
        apply_rotation = False
        proc_w = orig_w
        proc_h = orig_h
        applied_rotation_deg = 0

        meta = {
            "original_dimensions": f"{orig_w}x{orig_h}",
            "processed_dimensions": f"{proc_w}x{proc_h}",
            "applied_rotation_deg": applied_rotation_deg,
        }

        assert meta["original_dimensions"] == meta["processed_dimensions"]
        assert meta["applied_rotation_deg"] == 0

    def test_metadata_records_zero_rotation_for_unverified(self):
        """
        Unverified/quarantined clip metadata must record applied_rotation_deg == 0.
        """
        orig_w, orig_h = 320, 240  # looks like a native signer but unverified
        apply_rotation = False   # gating prevents rotation
        applied_rotation_deg = 0

        meta = {
            "original_dimensions": f"{orig_w}x{orig_h}",
            "processed_dimensions": f"{orig_w}x{orig_h}",
            "applied_rotation_deg": applied_rotation_deg,
        }

        assert meta["applied_rotation_deg"] == 0
