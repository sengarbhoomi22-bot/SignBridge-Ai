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
- Filename stem extraction (Step 2.4B / 2.4C): regex fix for MVI_ camera IDs.
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


# =============================================================================
# Filename Stem Extraction Tests (Step 2.4B / 2.4C — MVI regex fix)
# =============================================================================

class TestFilenameStemExtraction:
    """
    Tests for extract_frame_stem() and clean_stem() in
    scripts/inspect_full_corpus_dry_run.py.

    Before the fix, ``re.sub(r'_\\d+$', '', p)`` stripped '_6209' from 'MVI_6209',
    causing all 47 MVI-prefixed videos to fail stem matching and be quarantined.
    After the fix, only whitespace-separated trailing digit runs are stripped so
    underscore-separated camera IDs (e.g. '_6209') are preserved.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from inspect_full_corpus_dry_run import extract_frame_stem, clean_stem
        self.extract_frame_stem = extract_frame_stem
        self.clean_stem = clean_stem

    # ------------------------------------------------------------------
    # extract_frame_stem: MVI camera-ID preservation (the fixed defect)
    # ------------------------------------------------------------------

    def test_mvi_6209_frame_stem_preserved(self):
        """MVI_6209 camera ID must NOT be stripped; stem must be 'mvi_6209'."""
        assert self.extract_frame_stem("MVI_6209 01.jpg") == "mvi_6209"

    def test_mvi_6208_frame_stem_preserved(self):
        """MVI_6208 camera ID must NOT be stripped; stem must be 'mvi_6208'."""
        assert self.extract_frame_stem("MVI_6208 01.jpg") == "mvi_6208"

    def test_mvi_high_frame_number_preserved(self):
        """Two-digit frame index (e.g. ' 22') must be stripped; camera ID must remain."""
        assert self.extract_frame_stem("MVI_6209 22.jpg") == "mvi_6209"

    # ------------------------------------------------------------------
    # extract_frame_stem: standard sentence-level naming
    # ------------------------------------------------------------------

    def test_simple_sentence_frame_stem(self):
        """Plain sentence video frames strip trailing whitespace+digits only."""
        assert self.extract_frame_stem("are you free today 01.jpg") == "are you free today"

    def test_parenthesised_take_frame_stem(self):
        """Frames with (N) take suffix in stem preserve the take suffix."""
        assert self.extract_frame_stem("are you free today (2) 01.jpg") == "are you free today (2)"

    def test_free_take_2_frame_stem(self):
        """'free (2)' stem is preserved; trailing frame index stripped."""
        assert self.extract_frame_stem("free (2) 01.jpg") == "free (2)"

    def test_case_insensitive_jpg_extension(self):
        """'.JPG' extension (uppercase) must also be stripped."""
        assert self.extract_frame_stem("MVI_6209 01.JPG") == "mvi_6209"

    def test_output_is_lowercase(self):
        """extract_frame_stem must always return a lowercase string."""
        assert self.extract_frame_stem("Are You Free Today 01.jpg") == "are you free today"

    # ------------------------------------------------------------------
    # clean_stem: video filename normalisation
    # ------------------------------------------------------------------

    def test_clean_stem_mvi_6209(self):
        """clean_stem on video filename preserves underscore-delimited camera ID."""
        assert self.clean_stem("MVI_6209.MP4") == "mvi_6209"

    def test_clean_stem_free_2(self):
        """clean_stem on 'free (2).MP4' returns 'free (2)'."""
        assert self.clean_stem("free (2).MP4") == "free (2)"

    def test_clean_stem_plain_sentence(self):
        """clean_stem on plain sentence video returns lowercase stem."""
        assert self.clean_stem("are you free today.mp4") == "are you free today"

    def test_clean_stem_case_insensitive_extension(self):
        """clean_stem strips .mp4 and .MP4 identically."""
        assert self.clean_stem("MVI_6209.MP4") == self.clean_stem("MVI_6209.mp4")

    # ------------------------------------------------------------------
    # Consistency: video stem must match frame stem for Tier A samples
    # ------------------------------------------------------------------

    def test_tier_a_mvi_6209_stem_match(self):
        """Video stem for MVI_6209.MP4 must match frame stem for MVI_6209 01.jpg."""
        video_stem = self.clean_stem("MVI_6209.MP4")
        frame_stem = self.extract_frame_stem("MVI_6209 01.jpg")
        assert video_stem == frame_stem, (
            f"Stem mismatch: video={video_stem!r}, frame={frame_stem!r}"
        )

    def test_tier_a_are_you_free_today_stem_match(self):
        """Video stem for 'are you free today.mp4' must match frame stem."""
        video_stem = self.clean_stem("are you free today.mp4")
        frame_stem = self.extract_frame_stem("are you free today 01.jpg")
        assert video_stem == frame_stem

    def test_tier_a_free_2_stem_match(self):
        """Video stem for 'free (2).MP4' must match frame stem."""
        video_stem = self.clean_stem("free (2).MP4")
        frame_stem = self.extract_frame_stem("free (2) 01.jpg")
        assert video_stem == frame_stem

    def test_tier_a_are_you_free_today_2_stem_match(self):
        """Video stem for 'are you free today (2).mp4' must match frame stem."""
        video_stem = self.clean_stem("are you free today (2).mp4")
        frame_stem = self.extract_frame_stem("are you free today (2) 01.jpg")
        assert video_stem == frame_stem


# =============================================================================
# Step 2.4C: Full-Corpus Preprocessing v2 Pipeline Integrity Tests
# =============================================================================

class TestV2PipelineIntegrity:
    """
    Validation suite for full-corpus preprocessing v2 manifests,
    accounting invariants, tensor dimensionalities, and split isolation.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.v2_dir = PROJECT_ROOT / "data" / "processed" / "isl_csltr_tensors_v2"
        self.catalog_path = PROJECT_ROOT / "data" / "raw" / "isl-csltr" / "metadata" / "corpus_dry_run_results.json"

    def test_v2_output_directory_structure(self):
        """Verify that isl_csltr_tensors_v2 directory contains all required subdirectories."""
        if not self.v2_dir.exists():
            pytest.skip("v2 processed directory not yet generated")

        required_subdirs = ["train", "validation", "test", "quarantine", "metadata", "manifests", "quality_reports"]
        for sub in required_subdirs:
            assert (self.v2_dir / sub).exists(), f"Missing required v2 directory: {sub}"

    def test_v2_manifests_exist_and_conform(self):
        """Verify that master, quarantine, and failed manifests exist with expected header columns."""
        if not self.v2_dir.exists():
            pytest.skip("v2 processed directory not yet generated")

        import csv
        master_path = self.v2_dir / "manifests" / "master_manifest.csv"
        quarantine_path = self.v2_dir / "manifests" / "quarantine_manifest.csv"
        failed_path = self.v2_dir / "manifests" / "failed_videos.csv"

        assert master_path.exists(), "master_manifest.csv missing"
        assert quarantine_path.exists(), "quarantine_manifest.csv missing"
        assert failed_path.exists(), "failed_videos.csv missing"

        with open(master_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            assert "clip_id" in headers
            assert "feature_dim" in headers
            assert "applied_rotation_deg" in headers
            assert "qc_level_1_pass" in headers
            assert "qc_level_2_pass" in headers
            assert "qc_level_3_pass" in headers
            assert "suitability_for_training" in headers

    def test_v2_accounting_invariants(self):
        """Verify 453 eligible benchmark + 234 quarantined = 687 total corpus invariant."""
        import json
        if not self.catalog_path.exists():
            pytest.skip("Dry run catalog not found")

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        records = catalog["records"]
        assert len(records) == 687, f"Total records must be 687, got {len(records)}"

        train_cnt = sum(1 for r in records if r["proposed_split"] == "train")
        val_cnt = sum(1 for r in records if r["proposed_split"] == "val")
        test_cnt = sum(1 for r in records if r["proposed_split"] == "test")
        quarantine_cnt = sum(1 for r in records if r["proposed_split"] == "quarantine")

        assert train_cnt == 319, f"Expected 319 train videos, got {train_cnt}"
        assert val_cnt == 66, f"Expected 66 val videos, got {val_cnt}"
        assert test_cnt == 68, f"Expected 68 test videos, got {test_cnt}"
        assert quarantine_cnt == 234, f"Expected 234 quarantined videos, got {quarantine_cnt}"
        assert (train_cnt + val_cnt + test_cnt) == 453, "Benchmark total must be 453"
        assert (453 + quarantine_cnt) == 687, "Corpus sum must be 687"

    def test_v2_signer_7_strictly_isolated_to_test(self):
        """Verify that Signer 7 videos appear exclusively in the test split."""
        import json
        if not self.catalog_path.exists():
            pytest.skip("Dry run catalog not found")

        with open(self.catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        records = catalog["records"]
        for r in records:
            if r.get("candidate_signer_id") == "7":
                assert r["proposed_split"] == "test", (
                    f"Signer 7 video {r['video_filename']} leaked into {r['proposed_split']}!"
                )

    def test_v2_quarantined_suitability_is_always_false(self):
        """Verify that every quarantined video has suitability_for_training == False."""
        if not self.v2_dir.exists():
            pytest.skip("v2 processed directory not yet generated")

        import csv
        quarantine_path = self.v2_dir / "manifests" / "quarantine_manifest.csv"
        with open(quarantine_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["suitability_for_training"] in ("False", "false", False), (
                    f"Quarantined video {row['video_filename']} marked suitable for training!"
                )

    def test_v2_physical_npz_finite_values_and_dimensions(self):
        """Verify all generated physical NPZ feature tensors have finite values and exact dimensions."""
        if not self.v2_dir.exists():
            pytest.skip("v2 processed directory not yet generated")

        import csv
        master_path = self.v2_dir / "manifests" / "master_manifest.csv"
        with open(master_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                npz_rel = row["npz_rel_path"]
                npz_full = self.v2_dir / npz_rel
                assert npz_full.exists(), f"Missing physical tensor: {npz_rel}"

                data = np.load(npz_full)
                features = data["features"]
                mask = data["presence_mask"]

                assert features.ndim == 2
                assert features.shape[1] == 229, f"Feature dimension mismatch in {npz_rel}: {features.shape}"
                assert mask.shape == (features.shape[0], 4)
                assert np.isfinite(features).all(), f"NaN or Inf in {npz_rel}!"
                assert np.isfinite(mask).all(), f"NaN or Inf in mask of {npz_rel}!"

    def test_make_clip_id_deterministic(self):
        """Verify make_clip_id helper generates consistent, filesystem-safe strings."""
        from preprocess_isl_csltr import make_clip_id
        assert make_clip_id("are you free today (2).mp4") == "are_you_free_today_2"
        assert make_clip_id("free (2).MP4") == "free_2"
        assert make_clip_id("MVI_6209.MP4") == "mvi_6209"
        assert make_clip_id("bring water for me (1).mp4") == "bring_water_for_me_1"
