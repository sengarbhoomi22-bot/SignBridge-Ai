"""
SignBridge AI — Resumable ISL-CSLTR Full Archive Downloader.

Dataset: ISL-CSLTR (kcmpdxky7p/1)
Target File: ISL_CSLRT_Corpus.zip (8,905,822,924 bytes)
Target Destination: D:\\ISL_CSLRT_Corpus.zip (via D:\\ISL_CSLRT_Corpus.zip.part)

Features:
- Dynamic official Mendeley Data API lookup (no hardcoded S3 URLs).
- Automatic HTTP redirect handling (302 -> AWS S3).
- HTTP Range header resume support (bytes={existing}-).
- Verification of 206 Partial Content and Content-Range.
- Chunked streaming with ETA, download speed, and progress reporting.
- Bounded retries with exponential backoff and jitter.
- Pre-flight disk space validation.
- Final size validation against API metadata (8,905,822,924 bytes).
- ZIP central directory integrity validation via zipfile.is_zipfile().
- Atomic rename from .part to final destination upon verification.
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# --- Configuration & Constants ---
DATASET_ID = "kcmpdxky7p"
API_METADATA_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
USER_AGENT = "SignBridgeAI-Downloader/1.0 (Research Pipeline)"
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
DEFAULT_MAX_RETRIES = 30
DEFAULT_TIMEOUT = 30  # seconds
BACKOFF_FACTOR = 2.0
MAX_BACKOFF = 60.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("csltr_downloader")


class DownloadError(Exception):
    """Custom exception for non-recoverable download errors."""
    pass


def fetch_dataset_metadata(api_url: str = API_METADATA_URL, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Query official Mendeley Data Public API for dataset file details."""
    req = urllib.request.Request(
        api_url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise DownloadError(f"Mendeley API returned HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        raise DownloadError(f"Failed to query Mendeley Data API: {e}") from e


def resolve_archive_info(metadata: dict, expected_filename: str = "ISL_CSLRT_Corpus.zip") -> tuple[str, int, str]:
    """
    Extract download URL, total expected size, and file ID from metadata.
    Returns: (download_url, size_bytes, file_id)
    """
    files = metadata.get("files", [])
    for f in files:
        if f.get("filename") == expected_filename:
            content_details = f.get("content_details", {})
            download_url = content_details.get("download_url")
            size = f.get("size") or content_details.get("size")
            file_id = f.get("id")
            if not download_url or not size:
                raise DownloadError(f"Incomplete file details for {expected_filename} in metadata response.")
            return download_url, int(size), str(file_id)
    raise DownloadError(f"Archive '{expected_filename}' not found in Mendeley dataset metadata files list.")


def check_disk_space(target_path: Path, required_bytes: int) -> None:
    """Verify target partition has enough free space with a 1.1x safety buffer."""
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    needed_with_buffer = int(required_bytes * 1.1)
    logger.info(
        f"Disk space check on {target_dir}: {free_bytes / (1024**3):.2f} GB free, "
        f"need {needed_with_buffer / (1024**3):.2f} GB (1.1x buffer)."
    )
    if free_bytes < needed_with_buffer:
        raise DownloadError(
            f"Insufficient disk space on {target_dir}. Free: {free_bytes / (1024**3):.2f} GB, "
            f"Required (with buffer): {needed_with_buffer / (1024**3):.2f} GB."
        )


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    else:
        return f"{num_bytes / (1024**3):.3f} GB"


def validate_zip_integrity(file_path: Path) -> bool:
    """
    Validate that the completed file is a valid ZIP archive
    by inspecting its central directory and end-of-central-directory record.
    """
    logger.info(f"Validating ZIP integrity for {file_path}...")
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist for ZIP validation.")
        return False

    if not zipfile.is_zipfile(file_path):
        logger.error(f"{file_path} is NOT recognized as a valid ZIP archive.")
        return False

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            # Test zipfile structural consistency
            first_bad = zf.testzip()
            if first_bad is not None:
                logger.error(f"Corrupt file found in ZIP archive: {first_bad}")
                return False
            num_files = len(zf.namelist())
            logger.info(f"ZIP validation passed successfully ({num_files:,} entries verified).")
            return True
    except Exception as e:
        logger.error(f"Exception during ZIP validation: {e}")
        return False


def download_archive(
    destination: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    simulate_max_bytes: int | None = None,
) -> bool:
    """
    Resumable download of ISL_CSLRT_Corpus.zip.

    Args:
        destination: Target final path (e.g. D:\\ISL_CSLRT_Corpus.zip)
        chunk_size: Streaming chunk size in bytes
        max_retries: Maximum consecutive retry attempts
        timeout: Socket timeout in seconds
        simulate_max_bytes: If set, stop downloading after this many bytes (for unit/integration tests)

    Returns:
        True if successfully completed and validated, False otherwise.
    """
    destination = Path(destination)
    part_path = Path(str(destination) + ".part")

    # If destination already exists and is complete
    if destination.exists():
        logger.info(f"Final archive already exists at {destination}.")
        # Pre-check metadata to confirm size
        meta = fetch_dataset_metadata(timeout=timeout)
        _, expected_size, _ = resolve_archive_info(meta)
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            logger.info(f"Archive matches expected size ({format_bytes(actual_size)}). Validating ZIP...")
            if validate_zip_integrity(destination):
                logger.info("Existing archive is intact and complete. Skipping download.")
                return True
            else:
                logger.warning("Existing archive failed ZIP validation. Moving to backup and re-downloading.")
                backup_path = destination.with_suffix(".corrupt." + str(int(time.time())))
                destination.rename(backup_path)
                logger.info(f"Renamed corrupt file to {backup_path}")
        else:
            logger.warning(
                f"Existing archive size {actual_size} != expected {expected_size}. "
                "Moving to backup and re-downloading."
            )
            backup_path = destination.with_suffix(".size_mismatch." + str(int(time.time())))
            destination.rename(backup_path)

    # 1. Fetch metadata and resolve target
    logger.info("Fetching fresh dataset metadata from official Mendeley Data API...")
    metadata = fetch_dataset_metadata(timeout=timeout)
    download_endpoint, expected_size, file_id = resolve_archive_info(metadata)
    logger.info(f"Resolved archive: expected size = {format_bytes(expected_size)} ({expected_size:,} bytes)")

    effective_target_size = min(expected_size, simulate_max_bytes) if simulate_max_bytes else expected_size

    # 2. Check disk space (target size minus already downloaded bytes)
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    remaining_bytes = max(0, effective_target_size - existing_bytes)
    check_disk_space(part_path, remaining_bytes)

    if existing_bytes > expected_size:
        logger.warning(
            f"Existing partial file size ({existing_bytes:,}) exceeds expected size ({expected_size:,}). "
            "Backing up partial file."
        )
        backup_part = part_path.with_suffix(".part.overlarge." + str(int(time.time())))
        part_path.rename(backup_part)
        existing_bytes = 0

    # 3. Main resumable download loop
    retry_count = 0
    start_time = time.time()
    last_log_time = start_time
    session_bytes_downloaded = 0

    logger.info(
        f"Starting download to {part_path} "
        f"(already present: {format_bytes(existing_bytes)} / {format_bytes(effective_target_size)})"
    )

    while existing_bytes < effective_target_size and retry_count < max_retries:
        try:
            # Always request fresh download URL upon retry to handle S3 token expiry
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
            }
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"
                logger.info(f"Requesting Range: bytes={existing_bytes}- from Mendeley redirect...")
            else:
                logger.info("Requesting full stream from byte 0...")

            req = urllib.request.Request(download_endpoint, headers=headers)

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_code = resp.status
                content_range = resp.headers.get("Content-Range")
                content_length = resp.headers.get("Content-Length")

                # Validate response code
                if existing_bytes > 0:
                    if resp_code != 206:
                        raise DownloadError(
                            f"Server returned HTTP {resp_code} instead of 206 Partial Content on Range request."
                        )
                    if not content_range:
                        raise DownloadError("Server did not return Content-Range header for Range request.")
                    # Format: bytes start-end/total
                    logger.info(f"Server acknowledged partial content: {content_range}")
                else:
                    if resp_code not in (200, 206):
                        raise DownloadError(f"Server returned unexpected HTTP {resp_code}.")
                    logger.info(f"Server responded HTTP {resp_code}, Content-Length: {content_length}")

                # Open partial file in append mode (or write if 0)
                mode = "ab" if existing_bytes > 0 else "wb"
                with open(part_path, mode) as f:
                    while existing_bytes < effective_target_size:
                        bytes_to_read = chunk_size
                        if simulate_max_bytes:
                            bytes_to_read = min(bytes_to_read, simulate_max_bytes - existing_bytes)
                        if bytes_to_read <= 0:
                            break

                        chunk = resp.read(bytes_to_read)
                        if not chunk:
                            # Stream ended
                            break

                        f.write(chunk)
                        chunk_len = len(chunk)
                        existing_bytes += chunk_len
                        session_bytes_downloaded += chunk_len

                        # Progress reporting every 3 seconds or on completion
                        now = time.time()
                        if now - last_log_time >= 3.0 or existing_bytes >= effective_target_size:
                            elapsed = now - start_time
                            speed = session_bytes_downloaded / elapsed if elapsed > 0 else 0
                            pct = (existing_bytes / effective_target_size) * 100
                            rem_bytes = effective_target_size - existing_bytes
                            eta_sec = rem_bytes / speed if speed > 0 else 0

                            logger.info(
                                f"Progress: {format_bytes(existing_bytes)} / {format_bytes(effective_target_size)} "
                                f"({pct:.2f}%) | Speed: {format_bytes(int(speed))}/s | "
                                f"ETA: {int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                            )
                            last_log_time = now

                # If we successfully read without error, reset retry count
                retry_count = 0

                if simulate_max_bytes and existing_bytes >= simulate_max_bytes:
                    logger.info(f"Simulated cap of {simulate_max_bytes} bytes reached.")
                    return True

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Exceeded max retries ({max_retries}). Halting download. Error: {e}")
                return False

            backoff = min(MAX_BACKOFF, (BACKOFF_FACTOR ** min(retry_count, 6)) + random.uniform(0.5, 2.0))
            logger.warning(
                f"[Retry {retry_count}/{max_retries}] Connection interrupted: {e}. "
                f"Backing off for {backoff:.1f}s before resuming..."
            )
            time.sleep(backoff)

            # Re-fetch metadata in case file download URL expired
            try:
                metadata = fetch_dataset_metadata(timeout=timeout)
                download_endpoint, _, _ = resolve_archive_info(metadata)
            except Exception as meta_err:
                logger.warning(f"Failed to refresh metadata URL on retry: {meta_err}")

    # 4. Final verification and atomic rename
    if existing_bytes < expected_size:
        logger.error(
            f"Download terminated prematurely: {existing_bytes:,} / {expected_size:,} bytes downloaded."
        )
        return False

    logger.info(f"Download complete: {existing_bytes:,} bytes received.")
    logger.info(f"Verifying final size against metadata ({expected_size:,} bytes)...")
    if part_path.stat().st_size != expected_size:
        logger.error(
            f"Final size mismatch: {part_path.stat().st_size} != {expected_size}. "
            "Refusing to rename incomplete file."
        )
        return False

    # 5. ZIP Integrity validation
    if not validate_zip_integrity(part_path):
        logger.error("Downloaded file failed ZIP validation. Retaining .part file for inspection.")
        return False

    # 6. Atomic rename
    logger.info(f"Renaming {part_path} -> {destination}...")
    try:
        part_path.replace(destination)
        logger.info(f"SUCCESS: Archive safely downloaded and verified at {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to rename .part file to final destination: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI: Resumable ISL-CSLTR Archive Downloader")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("D:/ISL_CSLRT_Corpus.zip"),
        help="Target archive destination path (default: D:/ISL_CSLRT_Corpus.zip)",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=1,
        help="Chunk size in megabytes (default: 1 MB)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retry attempts (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Socket timeout in seconds (default: {DEFAULT_TIMEOUT}s)",
    )
    parser.add_argument(
        "--simulate-bytes",
        type=int,
        default=None,
        help="Cap download to N bytes for testing purposes",
    )

    args = parser.parse_args()

    chunk_bytes = args.chunk_size_mb * 1024 * 1024
    success = download_archive(
        destination=args.dest,
        chunk_size=chunk_bytes,
        max_retries=args.max_retries,
        timeout=args.timeout,
        simulate_max_bytes=args.simulate_bytes,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
