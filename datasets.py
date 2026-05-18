"""Dataset registry for the Voice Deepfake Forensic Pipeline.

To register a new dataset, append a DatasetSpec entry to
DATASET_REGISTRY.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from utils import (
    DURATION_SHORT_THRESHOLD,
    DURATION_LONG_THRESHOLD,
    generation_model,
    duration_bin,
    n_synth_segs_bin,
)
from config import LLAMAPARTIALSPOOF_AUDIO_DIR, LLAMAPARTIALSPOOF_LABEL_FILE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    """Static specification for a registered dataset.

    Fields
    ------
    key          : Unique string identifier used in cache filenames and UI.
    display_name : Human-readable name shown in the UI.
    audio_dir    : Root directory containing audio files.
    label_files  : List of label file paths.
    description  : Free-text description of the dataset.
    license      : Licence / usage terms.
    citation     : Citation string (BibTeX or plain text).
    audio_ext    : Audio file extension (default: 'wav').
    train_ratio  : Fraction of each stratum assigned to train (default: 0.8).
    split_seed   : Random seed for the BM train/test split (default: 42).
    """
    key:          str
    display_name: str
    audio_dir:    str
    label_files:  List[str]
    description:  str
    license:      str
    citation:     str
    audio_ext:    str   = "wav"
    train_ratio:  float = 0.8
    split_seed:   int   = 42

    def audio_path(self) -> Path:
        """Resolved Path to the audio root directory."""
        return Path(self.audio_dir)

    def label_paths(self) -> List[Path]:
        """Resolved Paths for all label files."""
        return [Path(p) for p in self.label_files]

    def exists(self) -> bool:
        """Return True if the audio directory and all label files exist."""
        return (
            self.audio_path().exists()
            and all(p.exists() for p in self.label_paths())
        )

    def compute_bm_split(
        self,
        utterances: dict,
        cache_dir:  Path,
    ) -> Tuple[List[str], List[str]]:
        """Compute and persist the BM train/test split.

        Stratification key:
            (effective_label, generation_model, duration_bin, n_synth_segs_bin)

        A corrupt or unreadable cache file is discarded with a warning and
        the split is recomputed from scratch.

        Parameters
        ----------
        utterances:
            Dict mapping UID -> Utterance.
        cache_dir:
            Directory where the split JSON is stored.

        Returns
        -------
        (train_uids, test_uids) : Tuple[List[str], List[str]]
            Both lists are sorted for deterministic ordering.

        Raises
        ------
        ValueError
            If *utterances* is empty (cannot compute a meaningful split).
        """
        if not utterances:
            raise ValueError(
                f"compute_bm_split: utterances dict is empty for dataset {self.key!r}."
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        split_file = cache_dir / f"{self.key}_bm_split.json"

        # Return cached split if it exists.
        if split_file.exists():
            try:
                data = json.loads(split_file.read_text(encoding="utf-8"))
                train_uids = data["train_uids"]
                test_uids  = data["test_uids"]
                logger.info(
                    "Loaded BM split from cache: %d train / %d test.",
                    len(train_uids), len(test_uids),
                )
                return train_uids, test_uids
            except Exception as exc:
                logger.warning(
                    "Cached split file %s is unreadable (%s) — recomputing.",
                    split_file, exc,
                )
                split_file.unlink(missing_ok=True)

        # Build strata.
        strata: Dict[Tuple, List[str]] = {}
        for uid, utt in utterances.items():
            n_syn = sum(1 for s in utt.segments if s.label != "bonafide")
            stratum_key = (
                utt.effective_label,
                generation_model(uid),
                duration_bin(utt.duration),
                n_synth_segs_bin(n_syn),
            )
            strata.setdefault(stratum_key, []).append(uid)

        # Shuffle within each stratum and split at train_ratio.
        rng        = random.Random(self.split_seed)
        train_uids: List[str] = []
        test_uids:  List[str] = []
        stratum_log: List[dict] = []

        for stratum_key in sorted(strata):
            group   = sorted(strata[stratum_key])
            rng.shuffle(group)
            n_train = max(1, round(len(group) * self.train_ratio))
            n_test  = 0 if len(group) == 1 else len(group) - n_train

            train_uids.extend(group[:n_train])
            if n_test > 0:
                test_uids.extend(group[n_train:])

            stratum_log.append({
                "stratum": list(stratum_key),
                "total":   len(group),
                "train":   n_train,
                "test":    n_test,
            })

        train_uids = sorted(train_uids)
        test_uids  = sorted(test_uids)

        # Full documentation for traceability.
        split_doc = {
            "dataset_key":  self.key,
            "computed_at":  datetime.now().isoformat(timespec="seconds"),
            "train_ratio":  self.train_ratio,
            "split_seed":   self.split_seed,
            "stratification_key": [
                "effective_label",
                "generation_model",
                "duration_bin",
                "n_synth_segs_bin",
            ],
            "bin_definitions": {
                "duration_bin": {
                    "short":  f"< {DURATION_SHORT_THRESHOLD} s",
                    "medium": f"{DURATION_SHORT_THRESHOLD} – {DURATION_LONG_THRESHOLD} s",
                    "long":   f">= {DURATION_LONG_THRESHOLD} s",
                },
                "n_synth_segs_bin": {
                    "0":   "0 synthetic segments (bonafide or fully synthetic)",
                    "1-3": "1 to 3 synthetic segments",
                    "3+":  "more than 3 synthetic segments",
                },
            },
            "counts": {
                "total": len(train_uids) + len(test_uids),
                "train": len(train_uids),
                "test":  len(test_uids),
            },
            "strata":     stratum_log,
            "train_uids": train_uids,
            "test_uids":  test_uids,
        }
        split_file.write_text(json.dumps(split_doc, indent=2), encoding="utf-8")
        logger.info(
            "BM split computed and cached: %d train / %d test (%d strata).",
            len(train_uids), len(test_uids), len(stratum_log),
        )
        return train_uids, test_uids


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: List[DatasetSpec] = [
    DatasetSpec(
        key          = "llamapartialspoof",
        display_name = "LlamaPartialSpoof",
        audio_dir   = LLAMAPARTIALSPOOF_AUDIO_DIR,
        label_files = [LLAMAPARTIALSPOOF_LABEL_FILE],
        description  = (
            "Partially synthetic speech corpus generated with different TTS"
            "systems. Contains bonafide, fully synthetic, and partially synthetic "
            "utterances with segment-level ground truth labels and TTS model "
            "provenance. 76,228 utterances across 10,573 bonafide, 33,461 fully "
            "synthetic, and 32,194 partially synthetic files."
        ),
        license  = "?",
        citation = (
            "Luong, H. T., Li, H., Zhang, L., Lee, K. A., & Chng, E. S. (2025, April). "
            "Llamapartialspoof: An llm-driven fake speech dataset simulating disinformation generation. "
            "In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE."
        ),
        train_ratio = 0.8,
        split_seed  = 42,
    ),
]

DATASET_REGISTRY_BY_KEY: Dict[str, DatasetSpec] = {
    ds.key: ds for ds in DATASET_REGISTRY
}


def get_dataset(key: str) -> DatasetSpec:
    """Return the DatasetSpec for *key*.

    Raises
    ------
    KeyError
        If *key* is not in the registry.
    """
    try:
        return DATASET_REGISTRY_BY_KEY[key]
    except KeyError:
        raise KeyError(key) from None