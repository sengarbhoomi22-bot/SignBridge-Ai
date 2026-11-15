"""SignBridge AI: PyTorch Dataset for ISL-CSLTR landmark tensor archives (.npz).

Supports:
- Signer-independent split loading (train, validation, test)
- On-demand or cached loading of compressed tensors
- Quality control filtering (suitability_for_training, qc_pass, signer_verified)
- Metadata extraction and LabelEncoder integration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:
    # Minimal fallback Dataset class when torch is not yet installed
    class Dataset:  # type: ignore
        def __len__(self) -> int:
            raise NotImplementedError
        def __getitem__(self, idx: int) -> Any:
            raise NotImplementedError

from backend.app.ml.labels import LabelEncoder


class ISLLandmarkDataset(Dataset):
    """PyTorch Dataset for loading preprocessed ISL-CSLTR sequence tensors."""

    def __init__(
        self,
        split_dir: Union[str, Path],
        label_encoder: Optional[LabelEncoder] = None,
        label_key: str = "sentence_text",
        min_frames: int = 10,
        filter_suitability: bool = True,
        cache_in_memory: bool = False,
    ) -> None:
        """
        Args:
            split_dir: Path to directory containing .npz files for this split
                       (e.g., data/processed/isl_csltr_tensors_v2/train).
            label_encoder: Fitted LabelEncoder instance.
            label_key: Metadata field to use for classification label
                       ('sentence_text' or 'gloss_text').
            min_frames: Minimum frame threshold to accept a sample.
            filter_suitability: If True, applies partition-appropriate QC checks:
                                - train: suitability_for_training == True and qc_pass == True
                                - val/test: qc_pass == True and signer_identity_verified == True
            cache_in_memory: If True, preload all numpy tensors into memory.
        """
        self.split_dir = Path(split_dir)
        self.label_encoder = label_encoder
        self.label_key = label_key
        self.min_frames = min_frames
        self.filter_suitability = filter_suitability
        self.cache_in_memory = cache_in_memory

        self.samples: List[Dict[str, Any]] = []
        self._cache: Dict[int, Dict[str, Any]] = {}

        self._discover_and_index()

    def _discover_and_index(self) -> None:
        """Indexes all valid .npz files in the split directory."""
        if not self.split_dir.exists():
            return

        split_name = self.split_dir.name.lower()
        npz_files = sorted(self.split_dir.glob("*.npz"))

        for npz_path in npz_files:
            try:
                # Read metadata without loading large feature arrays yet
                with np.load(npz_path, allow_pickle=True) as data:
                    meta_raw = data["metadata"]
                    meta = json.loads(str(meta_raw)) if isinstance(meta_raw, (str, np.ndarray)) else meta_raw
                    feat_shape = data["features"].shape

                # Partition-appropriate quality checks
                if self.filter_suitability:
                    qc_pass = meta.get("qc_pass", False)
                    if not qc_pass:
                        continue

                    if split_name == "train":
                        if not meta.get("suitability_for_training", False):
                            continue
                    elif split_name in ("val", "validation", "test"):
                        if not meta.get("signer_identity_verified", False):
                            continue
                    elif split_name == "quarantine":
                        continue

                t_frames = feat_shape[0]
                if t_frames < self.min_frames:
                    continue

                label_str = meta.get(self.label_key, "").strip()
                if not label_str:
                    continue

                entry = {
                    "path": npz_path,
                    "clip_id": meta.get("clip_id", npz_path.stem),
                    "label_str": label_str,
                    "num_frames": t_frames,
                    "metadata": meta,
                }
                self.samples.append(entry)

            except Exception:
                # Corrupted or unreadable files are skipped from indexing
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> List[str]:
        """Returns all string labels in the dataset."""
        return [s["label_str"] for s in self.samples]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a sample dictionary."""
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        entry = self.samples[idx]
        with np.load(entry["path"], allow_pickle=True) as data:
            features = data["features"].astype(np.float32)
            presence_mask = data["presence_mask"].astype(np.float32) if "presence_mask" in data else None
            meta_raw = data["metadata"]
            meta = json.loads(str(meta_raw)) if isinstance(meta_raw, (str, np.ndarray)) else meta_raw

        label_str = entry["label_str"]
        label_id = self.label_encoder.transform([label_str])[0] if self.label_encoder is not None else -1

        item = {
            "features": features,
            "presence_mask": presence_mask,
            "label_id": label_id,
            "label_str": label_str,
            "clip_id": entry["clip_id"],
            "metadata": meta,
        }

        if self.cache_in_memory:
            self._cache[idx] = item

        return item
