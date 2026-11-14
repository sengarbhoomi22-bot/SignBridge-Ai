"""
Full Corpus Preprocessing Dry Run Analysis Script
SignBridge AI — Step 2.4A
Scans all 687 sentence-level videos in ISL-CSLTR against frame directories,
applies the benchmark signer split, verifies video-to-frame correspondence,
and outputs detailed corpus-level statistics and quarantine reasons.
"""

import os
import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "isl-csltr" / "metadata"
FILE_LIST_JSON = METADATA_DIR / "csltr_file_list.json"
GLOSS_CSV = METADATA_DIR / "ISL Corpus sign glosses.csv"


def clean_stem(name: str) -> str:
    """Removes extension and trailing whitespace/casing."""
    base = re.sub(r'\.mp4$', '', name, flags=re.IGNORECASE).strip().lower()
    return base


def extract_frame_stem(frame_filename: str) -> str:
    """
    Extracts the base semantic stem from an extracted frame filename.
    Strips trailing whitespace-separated frame numbers only:
      e.g. 'foo 01.jpg'   -> 'foo'
           'foo (1).jpg'  -> 'foo'
           'free (2) 01.jpg' -> 'free (2)'
           'MVI_6209 01.jpg' -> 'mvi_6209'  (underscore-digit NOT stripped)
    Does NOT strip underscore-separated camera IDs such as the '_6209' in 'MVI_6209'.
    """
    base = re.sub(r'\.jpg$', '', frame_filename, flags=re.IGNORECASE).strip()
    # Strip trailing (N) suffix (e.g. ' (2)') – must come first
    p = re.sub(r'\s*\(\d+\)$', '', base)
    # Strip trailing whitespace-separated frame index (e.g. ' 01', ' 10')
    # Requires at least one whitespace character before the digits so that
    # underscore-prefixed numeric camera IDs (e.g. '_6209') are preserved.
    p = re.sub(r'\s+\d+$', '', p)
    return p.strip().lower()


def run_dry_run():
    print("=" * 70)
    print("SignBridge AI — Step 2.4A: Full-Corpus Preprocessing Dry Run")
    print("=" * 70)

    if not FILE_LIST_JSON.exists():
        print(f"Error: {FILE_LIST_JSON} not found!")
        return

    with open(FILE_LIST_JSON, "r", encoding="utf-8") as f:
        archive_files = json.load(f)

    # 1. Load sentence-to-gloss mappings
    gloss_map = {}
    if GLOSS_CSV.exists():
        with open(GLOSS_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0].strip():
                    gloss_map[r[0].strip().lower()] = r[1].strip()

    # 2. Extract videos and frames
    v_prefix = "ISL_CSLRT_Corpus/Videos_Sentence_Level/"
    f_prefix = "ISL_CSLRT_Corpus/Frames_Sentence_Level/"

    video_entries = [x for x in archive_files if x["name"].startswith(v_prefix) and not x["name"].endswith("/")]
    frame_entries = [x for x in archive_files if x["name"].startswith(f_prefix) and not x["name"].endswith("/") and not x["name"].endswith(".db")]

    print(f"Total Video Files Discovered : {len(video_entries)}")
    print(f"Total Frame Files Discovered : {len(frame_entries)}")

    # Group frames: sentence -> signer_id -> list of filenames
    frame_catalog = defaultdict(lambda: defaultdict(list))
    for f in frame_entries:
        rel = f["name"][len(f_prefix):]
        parts = rel.split("/")
        if len(parts) == 3:
            sent, signer, fname = parts
            frame_catalog[sent][signer].append(fname)

    # Group videos: sentence -> list of video entries
    video_catalog = defaultdict(list)
    for v in video_entries:
        rel = v["name"][len(v_prefix):]
        parts = rel.split("/")
        if len(parts) == 2:
            sent, vname = parts
            video_catalog[sent].append(v)

    unique_sent_videos = sorted(video_catalog.keys())
    unique_sent_frames = sorted(frame_catalog.keys())

    print(f"Unique Sentences in Videos   : {len(unique_sent_videos)}")
    print(f"Unique Sentences in Frames   : {len(unique_sent_frames)}")

    # Identify missing frame directories
    missing_frame_cats = [s for s in unique_sent_videos if s not in frame_catalog]
    print(f"Categories lacking frames    : {len(missing_frame_cats)} -> {missing_frame_cats}")

    # 3. Analyze each video
    records = []

    # Track duplicates
    seen_stems = defaultdict(list)

    for v in video_entries:
        vpath = v["name"]
        parts = vpath[len(v_prefix):].split("/")
        sent = parts[0]
        vname = parts[1]
        vstem = clean_stem(vname)
        size_bytes = v["uncompressed_size"]

        # Duplicate detection key: (sentence, stem)
        seen_stems[(sent, vstem)].append(vname)

        gloss = gloss_map.get(sent.lower(), "UNKNOWN_GLOSS")

        # Check category existence in frames
        if sent not in frame_catalog:
            records.append({
                "video_path": vpath,
                "sentence": sent,
                "video_filename": vname,
                "clean_stem": vstem,
                "gloss": gloss,
                "size_bytes": size_bytes,
                "matching_frame_dir": None,
                "candidate_signer_id": None,
                "signer_type": "unknown",
                "mapping_evidence": "None (Sentence directory completely missing from Frames_Sentence_Level)",
                "mapping_status": "NO_FRAME_DIR",
                "is_unambiguous": False,
                "proposed_split": "quarantine",
                "quarantine_reason": "Missing extracted frame directory in archive",
            })
            continue

        f_signers = frame_catalog[sent]

        # Check signer stem matches
        # For each signer in f_signers, extract all frame stems
        signer_stems = {}
        for sid, flist in f_signers.items():
            stems = set(extract_frame_stem(fn) for fn in flist)
            signer_stems[sid] = stems

        # Match video stem against signer stems
        matches = []
        for sid, stems in signer_stems.items():
            if vstem in stems:
                matches.append(sid)

        if len(matches) == 1:
            sid = matches[0]
            stype = "native" if sid in ["1", "2"] else "volunteer"

            # Fixed benchmark assignment
            if sid in ["1", "2", "3", "4", "5"]:
                split = "train"
            elif sid == "6":
                split = "val"
            elif sid == "7":
                split = "test"
            else:
                split = "quarantine"

            records.append({
                "video_path": vpath,
                "sentence": sent,
                "video_filename": vname,
                "clean_stem": vstem,
                "gloss": gloss,
                "size_bytes": size_bytes,
                "matching_frame_dir": f"Frames_Sentence_Level/{sent}/{sid}/",
                "candidate_signer_id": sid,
                "signer_type": stype,
                "mapping_evidence": f"Unambiguous exact stem match between video '{vname}' and frame files in folder '{sid}'",
                "mapping_status": "VERIFIED_UNIQUE",
                "is_unambiguous": True,
                "proposed_split": split,
                "quarantine_reason": None,
            })
        elif len(matches) > 1:
            records.append({
                "video_path": vpath,
                "sentence": sent,
                "video_filename": vname,
                "clean_stem": vstem,
                "gloss": gloss,
                "size_bytes": size_bytes,
                "matching_frame_dir": None,
                "candidate_signer_id": None,
                "signer_type": "unknown",
                "mapping_evidence": f"Stem matches frame files in multiple signer folders: {matches}",
                "mapping_status": "AMBIGUOUS",
                "is_unambiguous": False,
                "proposed_split": "quarantine",
                "quarantine_reason": f"Ambiguous stem match across multiple signer folders: {matches}",
            })
        else:
            records.append({
                "video_path": vpath,
                "sentence": sent,
                "video_filename": vname,
                "clean_stem": vstem,
                "gloss": gloss,
                "size_bytes": size_bytes,
                "matching_frame_dir": None,
                "candidate_signer_id": None,
                "signer_type": "unknown",
                "mapping_evidence": "No frame files in any signer folder match the video stem (generic '(N).jpg' frame naming)",
                "mapping_status": "UNVERIFIED",
                "is_unambiguous": False,
                "proposed_split": "quarantine",
                "quarantine_reason": "No matching frame stem in archive; generic frame numbering",
            })

    # Summary Statistics
    status_counts = Counter(r["mapping_status"] for r in records)
    split_counts = Counter(r["proposed_split"] for r in records)
    signer_counts = Counter(r["candidate_signer_id"] for r in records if r["candidate_signer_id"] is not None)
    quarantine_reasons = Counter(r["quarantine_reason"] for r in records if r["quarantine_reason"] is not None)

    print("\n" + "=" * 70)
    print("MAPPING STATUS SUMMARY:")
    for k, v in status_counts.items():
        print(f"  {k:20s}: {v:4d} videos ({v/len(records):.2%})")

    print("\nPROPOSED SPLIT DISTRIBUTION:")
    for k, v in split_counts.items():
        print(f"  {k:20s}: {v:4d} videos ({v/len(records):.2%})")

    print("\nVERIFIED SIGNER DISTRIBUTION:")
    for sid in sorted(signer_counts.keys(), key=lambda x: int(x)):
        stype = "native" if sid in ["1", "2"] else "volunteer"
        print(f"  Signer {sid} ({stype:9s}): {signer_counts[sid]:4d} videos")

    print("\nQUARANTINE REASONS:")
    for reason, cnt in quarantine_reasons.most_common():
        print(f"  [{cnt:3d} videos] {reason}")

    # Check duplicates
    duplicates = {k: v for k, v in seen_stems.items() if len(v) > 1}
    print(f"\nDuplicate Video Takes Detected: {len(duplicates)}")
    for (sent, stem), vnames in list(duplicates.items())[:10]:
        print(f"  Sentence: '{sent}' | Stem: '{stem}' -> Files: {vnames}")

    # Output detailed dry run results as a JSON file for the report
    out_json = METADATA_DIR / "corpus_dry_run_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_videos": len(records),
            "unique_sentence_categories": len(unique_sent_videos),
            "status_counts": dict(status_counts),
            "split_counts": dict(split_counts),
            "signer_counts": dict(signer_counts),
            "quarantine_reasons": dict(quarantine_reasons),
            "duplicate_takes_count": len(duplicates),
            "missing_frame_categories": missing_frame_cats,
            "records": records,
        }, f, indent=2)
    print(f"\n[INFO] Full dry run catalog saved to: {out_json}")


if __name__ == "__main__":
    run_dry_run()
