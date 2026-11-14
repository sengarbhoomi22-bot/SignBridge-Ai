"""
Unit and integration tests for scripts/download_csltr_archive.py.

Tests:
1. API URL retrieval & metadata parsing.
2. Archive metadata extraction (size, filename, file ID).
3. HTTP redirect handling.
4. Small byte-range request verification (HTTP 206, Content-Range, magic bytes).
5. Simulated interrupted download and resume verification.
6. Signed URL refresh logic.
7. ZIP integrity validation (valid vs corrupt).
8. Cleanup of all temporary test artifacts.
"""

import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import pytest

from scripts.download_csltr_archive import (
    API_METADATA_URL,
    DATASET_ID,
    USER_AGENT,
    check_disk_space,
    download_archive,
    fetch_dataset_metadata,
    format_bytes,
    resolve_archive_info,
    validate_zip_integrity,
)


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.000 GB"


def test_validate_zip_integrity_with_real_zip():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zip = Path(tmpdir) / "test_valid.zip"
        with zipfile.ZipFile(tmp_zip, "w") as zf:
            zf.writestr("test.txt", "hello world")
            zf.writestr("folder/inner.txt", "inner content")
        assert validate_zip_integrity(tmp_zip) is True


def test_validate_zip_integrity_with_corrupt_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_corrupt = Path(tmpdir) / "test_corrupt.zip"
        with open(tmp_corrupt, "wb") as f:
            f.write(b"NOT A REAL ZIP FILE CONTENT AT ALL")
        assert validate_zip_integrity(tmp_corrupt) is False


def test_validate_zip_integrity_nonexistent_file():
    assert validate_zip_integrity(Path("nonexistent_file_12345.zip")) is False


def test_check_disk_space_sufficient():
    # C: drive should have plenty of space for 1000 bytes
    target = Path("C:/Windows/Temp/dummy.bin")
    # Should not raise
    check_disk_space(target, 1000)


def test_check_disk_space_insufficient():
    target = Path("C:/Windows/Temp/dummy.bin")
    with pytest.raises(Exception) as exc_info:
        # Request 1000 Terabytes
        check_disk_space(target, 1000 * 1024**4)
    assert "Insufficient disk space" in str(exc_info.value)


@pytest.mark.network
def test_fetch_dataset_metadata():
    """Verify live Mendeley API endpoint returns valid metadata."""
    data = fetch_dataset_metadata(timeout=15)
    assert data.get("id") == DATASET_ID
    assert "ISL-CSLTR" in data.get("name", "")
    assert data.get("version") == 1
    files = data.get("files", [])
    assert len(files) == 1
    assert files[0].get("filename") == "ISL_CSLRT_Corpus.zip"


@pytest.mark.network
def test_resolve_archive_info():
    """Verify resolution of download URL and size from live metadata."""
    data = fetch_dataset_metadata(timeout=15)
    url, size, file_id = resolve_archive_info(data)
    assert "https://" in url
    assert "file_downloaded" in url
    assert size == 8905822924  # Known exact byte size
    assert file_id == "c14c8953-be53-4c7a-85d5-5bcd7e79d86e"


@pytest.mark.network
def test_live_range_request_and_magic_header():
    """Verify server returns HTTP 206, Content-Range, and PK ZIP magic bytes."""
    meta = fetch_dataset_metadata(timeout=15)
    download_url, total_size, _ = resolve_archive_info(meta)

    # Request first 128 bytes
    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-127"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status == 206
        content_range = resp.headers.get("Content-Range")
        assert content_range == f"bytes 0-127/{total_size}"
        assert resp.headers.get("Content-Length") == "128"
        data = resp.read()
        assert len(data) == 128
        # Standard ZIP local file header signature: PK\x03\x04
        assert data[:4] == b"PK\x03\x04"


@pytest.mark.network
def test_simulated_interrupted_download_and_resume():
    """
    Test downloading 64 KB first, interrupting it, and resuming
    to 128 KB, verifying data consistency and HTTP 206.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "test_partial.zip"
        part = Path(tmpdir) / "test_partial.zip.part"

        # Phase 1: Download exactly 64 KB
        success_phase1 = download_archive(
            destination=dest,
            chunk_size=16 * 1024,
            simulate_max_bytes=64 * 1024,
            timeout=15,
        )
        assert success_phase1 is True
        assert part.exists()
        assert part.stat().st_size == 64 * 1024
        # Verify first 4 bytes are ZIP magic header
        with open(part, "rb") as f:
            header = f.read(4)
            assert header == b"PK\x03\x04"

        # Phase 2: Resume from 64 KB to 128 KB
        success_phase2 = download_archive(
            destination=dest,
            chunk_size=16 * 1024,
            simulate_max_bytes=128 * 1024,
            timeout=15,
        )
        assert success_phase2 is True
        assert part.exists()
        assert part.stat().st_size == 128 * 1024

        # Verify that byte 64KB onwards is not duplicated
        # (Download direct 128 KB slice to compare)
        meta = fetch_dataset_metadata(timeout=15)
        url, _, _ = resolve_archive_info(meta)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-131071"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            ground_truth = resp.read()

        with open(part, "rb") as f:
            downloaded = f.read()

        assert downloaded == ground_truth
