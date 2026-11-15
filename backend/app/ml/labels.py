"""Label mapping and encoding utilities for ISL-CSLTR sentence recognition.

Enforces strict fitting on the training partition to prevent label leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


class LabelEncoder:
    """Encoder for mapping sentence categories to discrete integer class IDs.
    
    Attributes:
        classes_: List of unique label strings sorted deterministically.
        label_to_id: Dictionary mapping label string to integer index.
        id_to_label: Dictionary mapping integer index to label string.
    """

    def __init__(self, classes: Optional[Sequence[str]] = None) -> None:
        self.classes_: List[str] = []
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        if classes is not None:
            self._set_classes(classes)

    def _set_classes(self, classes: Sequence[str]) -> None:
        # Sort uniquely for deterministic, reproducible mappings
        unique_classes = sorted(set(classes))
        self.classes_ = unique_classes
        self.label_to_id = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.id_to_label = {idx: cls_name for idx, cls_name in enumerate(unique_classes)}

    @property
    def num_classes(self) -> int:
        """Returns the number of encoded classes."""
        return len(self.classes_)

    def fit(self, labels: Iterable[str]) -> "LabelEncoder":
        """Fits the encoder strictly on an iterable of labels.
        
        Args:
            labels: Iterable of sentence labels (must be from training split only).
            
        Returns:
            Self instance.
        """
        self._set_classes(list(labels))
        return self

    def transform(self, labels: Sequence[str]) -> List[int]:
        """Encodes a sequence of string labels into integer class IDs.
        
        Args:
            labels: Sequence of string labels.
            
        Returns:
            List of integer IDs.
            
        Raises:
            KeyError: If a label was not seen during fitting.
        """
        if not self.label_to_id:
            raise ValueError("LabelEncoder has not been fitted yet.")
        encoded = []
        for lbl in labels:
            if lbl not in self.label_to_id:
                raise KeyError(
                    f"Label '{lbl}' was not present in the fitted training set vocabulary. "
                    f"Available classes ({len(self.classes_)}): {self.classes_[:5]}..."
                )
            encoded.append(self.label_to_id[lbl])
        return encoded

    def fit_transform(self, labels: Sequence[str]) -> List[int]:
        """Fits the encoder and transforms the input sequence."""
        return self.fit(labels).transform(labels)

    def inverse_transform(self, class_ids: Sequence[int]) -> List[str]:
        """Decodes integer class IDs back into string labels."""
        if not self.id_to_label:
            raise ValueError("LabelEncoder has not been fitted yet.")
        decoded = []
        for cid in class_ids:
            if cid not in self.id_to_label:
                raise KeyError(f"Class ID {cid} is out of bounds for vocabulary of size {self.num_classes}.")
            decoded.append(self.id_to_label[cid])
        return decoded

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes label mapping to a JSON file."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_classes": self.num_classes,
            "classes": self.classes_,
            "label_to_id": self.label_to_id,
            "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LabelEncoder":
        """Loads a label mapping from a JSON file."""
        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"Label mapping file not found: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        encoder = cls(classes=payload["classes"])
        return encoder
