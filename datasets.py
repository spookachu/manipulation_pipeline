"""Dataset registry for the Voice Deepfake Forensic Pipeline.

Architecture
------------
Each dataset is described by a DatasetSpec. DatasetSpec.label_parser is
a strategy: a callable with the signature

    (label_file: Path, audio_dir: Path, audio_ext: str) -> Dict[str, Utterance]

that knows how to turn that dataset's label file into the canonical
Dict[uid, Utterance] used everywhere else in the pipeline.

To add a new dataset:
    1. Write a parser function that matches the strategy signature above.
    2. Optionally define a StratificationSpec with the fields that matter
       for that dataset's train/test split.
    3. Append a DatasetSpec to DATASET_REGISTRY.
    4. Add path constants to config.py.
"""

import json
import logging
import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import soundfile as sf

from utils import Segment, Utterance
from config import (
    LLAMAPARTIALSPOOF_AUDIO_DIR,
    LLAMAPARTIALSPOOF_LABEL_FILE,
    ASVSPOOF2021_LA_AUDIO_DIR,
    ASVSPOOF2021_LA_LABEL_FILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stratification helpers 
# ---------------------------------------------------------------------------
DURATION_SHORT_THRESHOLD   = 3.0   # seconds; below this is 'short'
DURATION_LONG_THRESHOLD    = 8.0   # seconds; at or above this is 'long'
SYNTH_SEG_MEDIAN_THRESHOLD = 3     # segments


# ---------------------------------------------------------------------------
# Generic stratification field extractors
#
# Each extractor has the signature  (utt: Utterance) -> str
# and is used as a StratificationField.extractor value.
# Dataset-specific ones are defined alongside their parser below.
# ---------------------------------------------------------------------------
def get_attack_id(utt: Utterance) -> str:
    """Return the attack-system ID stored on *utt*, or 'bonafide' if absent.

    Stratification dimension for ASVspoof-style datasets where the spoofing
    system identifier is embedded in the metadata rather than the UID.
    """
    return utt.attack_id if utt.attack_id is not None else "bonafide"


def get_codec(utt: Utterance) -> str:
    """Return the codec condition stored on *utt*, or 'unknown' if absent.

    Stratification dimension for ASVspoof-style datasets.
    """
    return utt.codec if utt.codec is not None else "unknown"


def duration_bin(duration_s: Union[int, float]) -> str:
    """Bin an audio duration into one of three named categories.

    Thresholds derived from LlamaPartialSpoof duration statistics:
        short  : < 3 s
        medium : 3 s - 8 s
        long   : ≥ 8 s

    Raises
    ------
    TypeError  if *duration_s* is not int or float.
    ValueError if *duration_s* is negative.
    """
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise TypeError(
            f"duration_s must be int or float, got {type(duration_s).__name__!r}"
        )
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")

    if duration_s < DURATION_SHORT_THRESHOLD:
        return "short"
    if duration_s < DURATION_LONG_THRESHOLD:
        return "medium"
    return "long"


def n_synth_segs_bin(n: int) -> str:
    """Bin a synthetic segment count into one of three named categories.

    Thresholds derived from LlamaPartialSpoof segment statistics
    (median = 3):
        0   : bonafide or fully synthetic (no splicing)
        1-3 : below-or-at-median partial synthetic
        3+  : above-median partial synthetic

    Raises
    ------
    TypeError  if *n* is not an int.
    ValueError if *n* is negative.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__!r}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if n == 0:
        return "0"
    if n <= SYNTH_SEG_MEDIAN_THRESHOLD:
        return "1-3"
    return "3+"


# ---------------------------------------------------------------------------
# LlamaPartialSpoof — parser + UID extractor
# ---------------------------------------------------------------------------
_LPS_UTT_LABELS = frozenset({"bonafide", "synthetic", "spoof"})
_LPS_SEGMENT_RE = re.compile(r"^(\d+\.?\d*)-(\d+\.?\d*)-(\w+)$")


def _lps_is_valid_uid(tok: str) -> bool:
    """Return True if *tok* looks like a LlamaPartialSpoof UID."""
    return (
        bool(tok)
        and tok[0].isalpha()
        and "_" in tok
        and not _LPS_SEGMENT_RE.match(tok)
    )


def _lps_parse_segment(tok: str) -> Segment:
    """Parse a LlamaPartialSpoof segment token '<start>-<end>-<label>'."""
    m = _LPS_SEGMENT_RE.match(tok)
    if not m:
        raise ValueError(f"Invalid segment token: {tok!r}")
    return Segment(float(m.group(1)), float(m.group(2)), m.group(3))


def generation_model(uid: str) -> str:
    """Extract the TTS generation model identifier from a LlamaPartialSpoof UID.

    Naming convention::

        <partition>-<model>_<speaker>_...

    Returns 'bonafide' when the partition suffix is 'clean' or absent.

    Raises
    ------
    TypeError  if *uid* is not a string.
    ValueError if *uid* is blank.
    """
    if not isinstance(uid, str):
        raise TypeError(f"uid must be a string, got {type(uid).__name__!r}")
    if not uid.strip():
        raise ValueError("uid must not be empty or blank.")

    prefix = uid.split("_")[0]
    parts  = prefix.split("-", 1)
    if len(parts) < 2 or parts[1] == "clean":
        return "bonafide"
    return parts[1]


def parse_llamapartialspoof_label_file(
    path:      Union[str, Path],
    audio_dir: Union[str, Path],  
    audio_ext: str = "wav",        
) -> Dict[str, Utterance]:
    """Parse a LlamaPartialSpoof label file into a dict of Utterances.

    Expected format — one record per line::

        <uid> <duration_s> <utterance_label> [<start>-<end>-<label> ...]

    Examples::

        test-clean_123 5.0 bonafide
        dev01-cosyvoice_456 3.2 spoof 0.0-1.5-spoof 2.0-3.2-bonafide

    Parameters
    ----------
    path:
        Path to the label file.
    audio_dir, audio_ext:
        Unused — present only to match the DatasetSpec.label_parser.

    Returns
    -------
    Dict[str, Utterance]
        Mapping of UID -> Utterance. Duplicate UIDs keep the last occurrence.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    utterances: Dict[str, Utterance] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) < 3:
                logger.debug("Line %d: too few tokens, skipping.", line_num)
                continue

            uid, duration_str, utt_label = tokens[0], tokens[1], tokens[2]

            if not _lps_is_valid_uid(uid):
                logger.debug("Line %d: invalid UID %r, skipping.", line_num, uid)
                continue

            try:
                duration = float(duration_str)
            except ValueError:
                logger.debug(
                    "Line %d: non-numeric duration %r, skipping.", line_num, duration_str
                )
                continue

            if utt_label not in _LPS_UTT_LABELS:
                logger.debug("Line %d: unknown label %r, skipping.", line_num, utt_label)
                continue

            segments: List[Segment] = []
            for tok in tokens[3:]:
                if _LPS_SEGMENT_RE.match(tok):
                    try:
                        segments.append(_lps_parse_segment(tok))
                    except ValueError as exc:
                        logger.warning(
                            "Line %d: skipping invalid segment %r — %s",
                            line_num, tok, exc,
                        )

            utterances[uid] = Utterance(
                uid=uid, duration=duration, label=utt_label, segments=segments
            )

    return utterances


# ---------------------------------------------------------------------------
# ASVspoof 2021 LA — parser
# ---------------------------------------------------------------------------
def parse_asvspoof2021_label_file(
    path:          Union[str, Path],
    audio_dir:     Union[str, Path],
    audio_ext:     str  = "flac",
    read_duration: bool = False,
) -> Dict[str, Utterance]:
    """Parse an ASVspoof 2021 LA trial_metadata.txt into a dict of Utterances.

    Format — 8 space-separated columns per line::

        <SPEAKER_ID> <TRIAL_ID> <CODEC> <TRANSMISSION> <ATTACK_ID> <KEY> <TRIM> <SUBSET>

    Only rows with subset == 'eval' are loaded.

    Parameters
    ----------
    path:
        Path to trial_metadata.txt.
    audio_dir:
        Root directory containing the .flac audio files.
    audio_ext:
        Audio file extension (default: 'flac').
    read_duration:
        If True, read duration from each audio file header via soundfile.info().
        Defaults to False (stores 0.0) since duration is not used by the ASVspoof.

    Returns
    -------
    Dict[str, Utterance]
        Mapping of trial ID -> Utterance.

    Raises
    ------
    FileNotFoundError
        If the label file does not exist.
    """
    path      = Path(path)
    audio_dir = Path(audio_dir)

    if not path.exists():
        raise FileNotFoundError(f"ASVspoof 2021 label file not found: {path}")

    utterances: Dict[str, Utterance] = {}
    n_skipped = 0
    n_missing = 0

    logger.info("ASVspoof2021: indexing audio dir %s ...", audio_dir)
    flat = list(audio_dir.glob(f"*.{audio_ext}"))
    disk_index: Dict[str, Path] = {p.stem: p for p in flat}
    logger.info(
        "ASVspoof2021: found %d %s files on disk.", len(disk_index), audio_ext
    )

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            if len(tokens) < 8:
                n_skipped += 1
                continue

            trial_id  = tokens[1]
            codec_val = tokens[2]
            attack_id = tokens[4]
            key       = tokens[5].lower()
            subset    = tokens[7].lower()

            if subset != "eval":
                continue
            if key not in ("bonafide", "spoof"):
                n_skipped += 1
                continue

            stem = trial_id.split("-")[0]
            if stem not in disk_index:
                n_missing += 1
                continue

            if read_duration:
                try:
                    duration = float(sf.info(str(disk_index[stem])).duration)
                except Exception:
                    duration = 0.0
            else:
                duration = 0.0

            utterances[stem] = Utterance(
                uid       = stem,
                duration  = duration,
                label     = key,
                segments  = [],
                attack_id = attack_id if attack_id != "-" else None,
                codec     = codec_val,
            )

    if n_missing:
        logger.warning(
            "parse_asvspoof2021_label_file: %d audio files missing on disk "
            "(expected for subset downloads).", n_missing,
        )
    if n_skipped:
        logger.warning(
            "parse_asvspoof2021_label_file: %d lines skipped (bad format/key/subset).",
            n_skipped,
        )
    logger.info(
        "parse_asvspoof2021_label_file: loaded %d utterances (missing=%d, skipped=%d).",
        len(utterances), n_missing, n_skipped,
    )
    return utterances


# ---------------------------------------------------------------------------
# StratificationSpec — strategy for building the BM train/test split key
# ---------------------------------------------------------------------------
@dataclass
class StratificationField:
    """One stratification dimension.

    Fields
    ------
    name:      Human-readable column name used in UI and log messages.
    extractor: Callable[[Utterance], str] — returns the stratum value.
    """
    name:      str
    extractor: Callable  # (Utterance) -> str


@dataclass
class StratificationSpec:
    """Ordered list of stratification dimensions for a dataset.

    The stratum key is always prepended by ``effective_label``.  Extra
    fields are appended in ``fields`` order.

    Example — LlamaPartialSpoof::

        StratificationSpec(fields=[
            StratificationField("TTS Model", lambda utt: generation_model(utt.uid)),
            StratificationField("Duration",  lambda utt: duration_bin(utt.duration)),
            StratificationField("N Syn Segs", lambda utt: n_synth_segs_bin(
                sum(1 for s in utt.segments if s.label != "bonafide"))),
        ])

    Example — ASVspoof 2021 LA::

        StratificationSpec(fields=[
            StratificationField("Attack", get_attack_id),
            StratificationField("Codec",  get_codec),
        ])
    """
    fields: List[StratificationField] = field(default_factory=list)

    def stratum_key(self, utt: Utterance) -> tuple:
        """Build the stratum key tuple for *utt*."""
        return (utt.effective_label,) + tuple(f.extractor(utt) for f in self.fields)

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]


# Pre-built specs used in the registry.
_LLAMAPARTIALSPOOF_STRAT = StratificationSpec(fields=[
    StratificationField(
        name      = "TTS Model",
        extractor = lambda utt: generation_model(utt.uid),
    ),
    StratificationField(
        name      = "Duration",
        extractor = lambda utt: duration_bin(utt.duration),
    ),
    StratificationField(
        name      = "N Syn Segs",
        extractor = lambda utt: n_synth_segs_bin(
            sum(1 for s in utt.segments if s.label != "bonafide")
        ),
    ),
])

_ASVSPOOF2021_LA_STRAT = StratificationSpec(fields=[
    StratificationField(name="Attack", extractor=get_attack_id),
    StratificationField(name="Codec",  extractor=get_codec),
])


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
    citation     : Citation string.
    audio_ext    : Audio file extension (default: 'wav').
    train_ratio  : Fraction of each stratum assigned to train (default: 0.8).
    split_seed   : Random seed for the BM train/test split (default: 42).
    label_parser : Strategy callable for label parsing. Signature:
                       (path: Path, audio_dir: Path, audio_ext: str)
                       -> Dict[str, Utterance]
                   Defaults to parse_llamapartialspoof_label_file.
    strat_spec   : StratificationSpec for the BM split. Defaults to
                   label-only stratification when None.
    """
    key:          str
    display_name: str
    audio_dir:    str
    label_files:  List[str]
    description:  str
    license:      str
    citation:     str
    audio_ext:    str                      = "wav"
    train_ratio:  float                    = 0.8
    split_seed:   int                      = 42
    label_parser: Optional[Callable]       = None
    strat_spec:   Optional[StratificationSpec] = None

    def audio_path(self) -> Path:
        return Path(self.audio_dir)

    def label_paths(self) -> List[Path]:
        return [Path(p) for p in self.label_files]

    def exists(self) -> bool:
        """Return True if the audio directory and all label files exist."""
        return (
            self.audio_path().exists()
            and all(p.exists() for p in self.label_paths())
        )

    def load_utterances(self) -> Dict[str, Utterance]:
        """Load utterances using the registered label_parser strategy.

        Falls back to parse_llamapartialspoof_label_file when no parser is set.
        """
        parser = self.label_parser or parse_llamapartialspoof_label_file
        return parser(
            self.label_paths()[0],
            self.audio_path(),
            self.audio_ext,
        )

    def compute_bm_split(
        self,
        utterances: Dict[str, Utterance],
        cache_dir:  Path,
    ) -> Tuple[List[str], List[str]]:
        """Compute and persist the stratified BM train/test split.
        
        Parameters
        ----------
        utterances:
            Dict mapping UID -> Utterance.
        cache_dir:
            Directory where the split JSON is stored.

        Returns
        -------
        (train_uids, test_uids) : both lists are sorted.

        Raises
        ------
        ValueError
            If *utterances* is empty.
        """
        if not utterances:
            raise ValueError(
                f"compute_bm_split: utterances dict is empty for dataset {self.key!r}."
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        split_file = cache_dir / f"{self.key}_bm_split.json"

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

        spec       = self.strat_spec
        field_names = spec.field_names if spec is not None else []

        strata: Dict[tuple, List[str]] = {}
        for uid, utt in utterances.items():
            key = spec.stratum_key(utt) if spec is not None else (utt.effective_label,)
            strata.setdefault(key, []).append(uid)

        rng         = random.Random(self.split_seed)
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

        split_doc = {
            "dataset_key":        self.key,
            "computed_at":        datetime.now().isoformat(timespec="seconds"),
            "train_ratio":        self.train_ratio,
            "split_seed":         self.split_seed,
            "stratification_key": ["effective_label"] + field_names,
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
        audio_dir    = LLAMAPARTIALSPOOF_AUDIO_DIR,
        label_files  = [LLAMAPARTIALSPOOF_LABEL_FILE],
        description  = (
            "Partially synthetic speech corpus generated with different TTS systems. "
            "Contains bonafide, fully synthetic, and partially synthetic utterances "
            "with segment-level ground truth labels and TTS model provenance. "
            "76,228 utterances: 10,573 bonafide, 33,461 synthetic, 32,194 partial."
        ),
        license  = "?",
        citation = (
            "Luong, H. T., Li, H., Zhang, L., Lee, K. A., & Chng, E. S. (2025). "
            "LlamaPartialSpoof: An LLM-driven fake speech dataset simulating "
            "disinformation generation. ICASSP 2025."
        ),
        train_ratio  = 0.8,
        split_seed   = 42,
        label_parser = parse_llamapartialspoof_label_file,
        strat_spec   = _LLAMAPARTIALSPOOF_STRAT,
    ),
    DatasetSpec(
        key          = "asvspoof2021_la",
        display_name = "ASVspoof 2021 LA",
        audio_dir    = ASVSPOOF2021_LA_AUDIO_DIR,
        label_files  = [ASVSPOOF2021_LA_LABEL_FILE],
        audio_ext    = "flac",
        description  = (
            "ASVspoof 2021 Logical Access evaluation set. Bonafide and fully "
            "synthetic (TTS/VC) utterances passed through telephony and VoIP codecs. "
            "13 spoofing systems (A07-A19) across 7 codec conditions. "
            "Ground truth from LA/CM/trial_metadata.txt (LA-keys-full.tar.gz)."
        ),
        license  = "CC BY 4.0",
        citation = (
            "Liu, X., Wang, X., Sahidullah, M., et al. (2023). "
            "ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild. "
            "IEEE/ACM TASLP. doi:10.1109/TASLP.2023.3285283"
        ),
        train_ratio  = 0.8,
        split_seed   = 42,
        label_parser = parse_asvspoof2021_label_file,
        strat_spec   = _ASVSPOOF2021_LA_STRAT,
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