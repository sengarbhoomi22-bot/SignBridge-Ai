"""Variable-length sequence collation and mask separation for SignBridge AI.

Batches variable-duration landmark sequences by:
1. Identifying batch maximum sequence length T_max.
2. Zero-padding feature vectors to (B, T_max, 229).
3. Preserving true sequence lengths (B,).
4. Generating a dedicated binary/boolean sequence padding mask (B, T_max).
   NOTE: This padding mask is strictly distinct from the 4 landmark presence
   indicators [m_left, m_right, m_pose, m_norm] embedded in each frame feature vector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


class SequenceBatchCollator:
    """Collates a list of variable-length sequence samples into a padded batch."""

    def __init__(self, feature_dim: int = 229, pad_value: float = 0.0) -> None:
        self.feature_dim = feature_dim
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates samples into padded numpy arrays or PyTorch tensors.
        
        Args:
            batch: List of dictionaries, each containing:
                - 'features': np.ndarray of shape (T_i, feature_dim)
                - 'presence_mask': np.ndarray of shape (T_i, 4)
                - 'label_id': int (optional if inference only)
                - 'label_str': str
                - 'metadata': dict
                
        Returns:
            Dictionary with:
                - 'features': padded tensor/array (B, T_max, feature_dim)
                - 'lengths': integer array/tensor (B,)
                - 'padding_mask': binary mask (B, T_max) where 1.0 indicates a valid frame,
                                  and 0.0 indicates a zero-padded frame.
                - 'presence_masks': padded modality flags (B, T_max, 4)
                - 'labels': integer array/tensor of shape (B,) if available
                - 'metadata': list of sample metadata dicts
        """
        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        batch_size = len(batch)
        lengths = [len(item["features"]) for item in batch]
        max_t = max(lengths)

        # Allocate zero-padded arrays
        padded_features = np.full(
            (batch_size, max_t, self.feature_dim),
            self.pad_value,
            dtype=np.float32,
        )
        padded_presence = np.zeros((batch_size, max_t, 4), dtype=np.float32)
        padding_mask = np.zeros((batch_size, max_t), dtype=np.float32)

        has_labels = "label_id" in batch[0]
        labels = np.zeros(batch_size, dtype=np.int64) if has_labels else None
        metadata_list = []

        for i, item in enumerate(batch):
            feat = item["features"]
            t_len = len(feat)
            padded_features[i, :t_len, :] = feat
            padding_mask[i, :t_len] = 1.0

            if "presence_mask" in item and item["presence_mask"] is not None:
                padded_presence[i, :t_len, :] = item["presence_mask"]

            if has_labels:
                labels[i] = item["label_id"]

            metadata_list.append(item.get("metadata", {}))

        result: Dict[str, Any] = {
            "features": padded_features,
            "lengths": np.array(lengths, dtype=np.int64),
            "padding_mask": padding_mask,
            "presence_masks": padded_presence,
            "metadata": metadata_list,
        }
        if labels is not None:
            result["labels"] = labels

        # Optional conversion to PyTorch tensors if torch is installed
        try:
            import torch
            result["features"] = torch.from_numpy(result["features"]).float()
            result["lengths"] = torch.from_numpy(result["lengths"]).long()
            result["padding_mask"] = torch.from_numpy(result["padding_mask"]).float()
            result["presence_masks"] = torch.from_numpy(result["presence_masks"]).float()
            if labels is not None:
                result["labels"] = torch.from_numpy(result["labels"]).long()
        except ImportError:
            pass

        return result
