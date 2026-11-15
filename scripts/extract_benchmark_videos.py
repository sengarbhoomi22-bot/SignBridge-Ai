"""
SignBridge AI — Step 2.5E: Selective Extraction Script.

Extracts exclusively the 449 verified benchmark-eligible MP4 files from
D:\\ISL_CSLRT_Corpus.zip into data/raw/isl-csltr/Videos_Sentence_Level/,
preserving original category subfolder structure while strictly excluding
all 234 quarantined videos.

Safety Invariants:
- Never extract files not present in the audited failed_videos.csv / VERIFIED_UNIQUE catalog.
- Never extract quarantined files (status != VERIFIED_UNIQUE).
- Never overwrite existing files without byte-size and CRC validation.
- Validates extracted file size and CRC32 against ZIP central directory.
- Produces extraction_manifest.csv with complete provenance and audit metadata.
"""

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
import time
import zipfile
import zlib
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("csltr_extractor")

ARCHIVE_DEFAULT = pathlib.Path("D:/ISL_CSLRT_Corpus.zip")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data/raw/isl-csltr/metadata/corpus_dry_run_results.json"
FAILED_MANIFEST_PATH = PROJECT_ROOT / "data/processed/isl_csltr_tensors_v2/manifests/failed_videos.csv"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "data/raw/isl-csltr/Videos_Sentence_Level"
MANIFEST_OUT_PATH = PROJECT_ROOT / "data/raw/isl-csltr/metadata/extraction_manifest.csv"
TARGET_PREFIX = "ISL_CSLRT_Corpus/Videos_Sentence_Level/"


def load_audited_targets(catalog_path: pathlib.Path, failed_path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Load and cross-validate extraction targets from the audited catalog and failed manifest.
    Returns: List of target dictionaries with metadata.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
    if not failed_path.exists():
        raise FileNotFoundError(f"Failed videos manifest not found: {failed_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    with open(failed_path, "r", encoding="utf-8", newline="") as f:
        failed_rows = list(csv.DictReader(f))

    catalog_map = {r["video_filename"]: r for r in catalog.get("records", [])}
    logger.info(f"Loaded {len(catalog_map)} total records from corpus catalog.")
    logger.info(f"Loaded {len(failed_rows)} pending discovery records from failed_videos.csv.")

    targets = []
    quarantined_count = 0

    for row in failed_rows:
        fn = row["video_filename"]
        if fn not in catalog_map:
            raise ValueError(f"Pending file '{fn}' not found in catalog!")

        cat_rec = catalog_map[fn]

        # Strict check: Must be VERIFIED_UNIQUE
        if cat_rec.get("mapping_status") != "VERIFIED_UNIQUE":
            quarantined_count += 1
            logger.error(f"FATAL: Quarantined video '{fn}' found in failed_videos.csv!")
            raise ValueError(f"Quarantined video '{fn}' cannot be targeted for extraction!")

        # Verify split and signer integrity
        assert row["assigned_split"] == cat_rec["proposed_split"], (
            f"Split mismatch for {fn}: manifest={row['assigned_split']} vs catalog={cat_rec['proposed_split']}"
        )
        assert row["candidate_signer_id"] == cat_rec["candidate_signer_id"], (
            f"Signer mismatch for {fn}: manifest={row['candidate_signer_id']} vs catalog={cat_rec['candidate_signer_id']}"
        )

        arch_path = cat_rec["video_path"]
        if not arch_path.startswith(TARGET_PREFIX):
            raise ValueError(f"Archive path '{arch_path}' does not start with expected prefix '{TARGET_PREFIX}'")

        rel_path = arch_path[len(TARGET_PREFIX):]

        targets.append({
            "video_filename": fn,
            "archive_path": arch_path,
            "extracted_rel_path": rel_path,
            "sentence_category": cat_rec.get("sentence", ""),
            "expected_size_bytes": cat_rec.get("size_bytes", 0),
            "candidate_signer_id": cat_rec.get("candidate_signer_id", ""),
            "signer_type": cat_rec.get("signer_type", ""),
            "assigned_split": cat_rec.get("proposed_split", ""),
            "mapping_status": cat_rec.get("mapping_status", ""),
            "mapping_evidence": cat_rec.get("mapping_evidence", ""),
        })

    logger.info(f"Successfully validated {len(targets)} extraction targets (0 quarantined).")
    return targets


def extract_selective_videos(
    archive_path: pathlib.Path,
    output_dir: pathlib.Path,
    manifest_out: pathlib.Path,
    dry_run: bool = False,
) -> bool:
    """
    Perform selective extraction of approved 449 benchmark videos from ZIP archive.
    """
    logger.info(f"Starting selective extraction. Archive: {archive_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry run mode: {dry_run}")

    if not archive_path.exists():
        logger.error(f"Archive file not found: {archive_path}")
        return False

    targets = load_audited_targets(CATALOG_PATH, FAILED_MANIFEST_PATH)
    if len(targets) != 449:
        logger.error(f"Expected exactly 449 targets, got {len(targets)}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    extracted_count = 0
    skipped_count = 0
    error_count = 0

    start_time = time.time()

    with zipfile.ZipFile(archive_path, "r") as zf:
        name_to_info = {zi.filename: zi for zi in zf.infolist()}
        logger.info(f"Opened archive. Total ZIP entries: {len(name_to_info)}")

        # Validate that all targets exist in archive
        for t in targets:
            if t["archive_path"] not in name_to_info:
                logger.error(f"Target archive path '{t['archive_path']}' not found in ZIP!")
                return False

        logger.info("All 449 targets confirmed present in ZIP central directory.")

        for idx, t in enumerate(targets, 1):
            arch_path = t["archive_path"]
            rel_path = t["extracted_rel_path"]
            dest_file = output_dir / rel_path
            zi = name_to_info[arch_path]

            status = "PENDING"
            extracted_bytes = 0
            crc_match = False

            if dest_file.exists():
                # Check if already extracted properly
                actual_sz = dest_file.stat().st_size
                if actual_sz == zi.file_size:
                    # Calculate CRC
                    with open(dest_file, "rb") as f:
                        file_crc = zlib.crc32(f.read())
                    if file_crc == zi.CRC:
                        status = "ALREADY_PRESENT_VALID"
                        extracted_bytes = actual_sz
                        crc_match = True
                        skipped_count += 1
                        logger.info(f"[{idx}/449] Already present & valid: {rel_path}")
                    else:
                        logger.warning(f"[{idx}/449] Present but CRC mismatch: {rel_path}. Re-extracting...")
                else:
                    logger.warning(f"[{idx}/449] Present but size mismatch: {rel_path}. Re-extracting...")

            if status == "PENDING" and not dry_run:
                try:
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    # Extract single member
                    with zf.open(zi) as source, open(dest_file, "wb") as target:
                        buf_size = 1024 * 1024
                        hasher = 0
                        while True:
                            chunk = source.read(buf_size)
                            if not chunk:
                                break
                            target.write(chunk)
                            hasher = zlib.crc32(chunk, hasher)

                    actual_sz = dest_file.stat().st_size
                    if actual_sz == zi.file_size and hasher == zi.CRC:
                        status = "EXTRACTED_SUCCESS"
                        extracted_bytes = actual_sz
                        crc_match = True
                        extracted_count += 1
                        if idx % 25 == 0 or idx == len(targets):
                            logger.info(f"[{idx}/449] Extracted: {rel_path} ({actual_sz:,} B, CRC OK)")
                    else:
                        status = "CORRUPT_EXTRACTION"
                        error_count += 1
                        logger.error(
                            f"[{idx}/449] Extraction verification failed for {rel_path}: "
                            f"size {actual_sz} vs {zi.file_size}, CRC {hasher} vs {zi.CRC}"
                        )
                except Exception as e:
                    status = f"ERROR: {e}"
                    error_count += 1
                    logger.error(f"[{idx}/449] Failed to extract {rel_path}: {e}")
            elif dry_run:
                status = "DRY_RUN_OK"
                extracted_bytes = zi.file_size
                crc_match = True

            manifest_rows.append({
                "archive_path": arch_path,
                "extracted_rel_path": str(pathlib.Path(rel_path).as_posix()),
                "file_size": extracted_bytes if not dry_run else zi.file_size,
                "expected_size": zi.file_size,
                "crc32_match": crc_match,
                "eligibility_status": t["mapping_status"],
                "candidate_signer_id": t["candidate_signer_id"],
                "signer_type": t["signer_type"],
                "assigned_split": t["assigned_split"],
                "sentence_category": t["sentence_category"],
                "extraction_status": status,
            })

    elapsed = time.time() - start_time
    logger.info(
        f"Extraction run complete in {elapsed:.1f}s. "
        f"Extracted: {extracted_count}, Skipped (Already Valid): {skipped_count}, Errors: {error_count}"
    )

    # Write manifest
    if manifest_rows:
        fieldnames = [
            "archive_path",
            "extracted_rel_path",
            "file_size",
            "expected_size",
            "crc32_match",
            "eligibility_status",
            "candidate_signer_id",
            "signer_type",
            "assigned_split",
            "sentence_category",
            "extraction_status",
        ]
        with open(manifest_out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info(f"Extraction manifest written to: {manifest_out}")

    return error_count == 0


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI: Selective Benchmark Video Extractor")
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        default=ARCHIVE_DEFAULT,
        help=f"Path to ISL_CSLRT_Corpus.zip (default: {ARCHIVE_DEFAULT})",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory for extracted videos (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--manifest-out",
        type=pathlib.Path,
        default=MANIFEST_OUT_PATH,
        help=f"Path for extraction manifest CSV (default: {MANIFEST_OUT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate targets without extracting bytes to disk",
    )

    args = parser.parse_args()

    success = extract_selective_videos(
        archive_path=args.archive,
        output_dir=args.out_dir,
        manifest_out=args.manifest_out,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
