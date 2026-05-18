"""
Deepfake speech detection — model loading, inference, result types,
and score caching.

The countermeasure (CM) score is defined as:

    cm_score = logit[synthetic] - logit[bonafide]

Higher values indicate stronger synthetic evidence.

Score caching
-------------
Background-model scores are expensive to compute across a full dataset.
score_and_cache runs inference on a list of utterances and writes
results to a JSON cache file in a crash-safe partial-write pattern.
load_cache reads one or more cache files and merges them into a
unified scores dict for downstream LR computation.
pipeline_cache_key produces a deterministic string identifier for a
manipulation pipeline, used to name M-BM cache files distinctly from the
unmanipulated BM cache.
"""

import hashlib
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR          = 16_000
MIN_WINDOW_S       = 0.5
MIN_WINDOW_SAMPLES = int(MIN_WINDOW_S * TARGET_SR)
_PROB_EPS          = 1e-7
CACHE_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Static specification for a deepfake detection model.

    Fields
    ------
    key              : Unique string identifier used throughout the codebase.
    display_name     : Human-readable name shown in the UI.
    hf_repo          : HuggingFace Hub repository string.
    target_sr        : Expected input sample rate (Hz).
    architecture     : Free-text description of the model architecture.
    ref              : Citation / licence information.
    spoof_idx        : Output index for the spoof class (set during loading).
    bonafide_idx     : Output index for the bonafide class (set during loading).
    polarity_override: Optional (spoof_idx, bonafide_idx) to force label
                       polarity when automatic resolution fails.
    pipeline_based   : If True, load via HuggingFace pipeline API.
    """
    key:               str
    display_name:      str
    hf_repo:           str
    target_sr:         int
    architecture:      str
    ref:               str
    spoof_idx:         int                       = -1
    bonafide_idx:      int                       = -1
    polarity_override: Optional[Tuple[int, int]] = None
    pipeline_based:    bool                      = False


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        key          = "df_arena_1b_v1",
        display_name = "DF-Arena-1B-V1",
        hf_repo      = "Speech-Arena-2025/DF_Arena_1B_V_1",
        target_sr    = TARGET_SR,
        architecture = (
            "~1B-parameter universal antispoofing model."
        ),
        ref = (
            "Speech-Arena-2025/DF_Arena_1B_V_1 (non-commercial licence). "
            "See https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1"
        ),
        pipeline_based = True,
    ),
]

_SPEC_BY_KEY: Dict[str, ModelSpec] = {s.key: s for s in MODEL_SPECS}


# ---------------------------------------------------------------------------
# Loaded model cache
# ---------------------------------------------------------------------------
@dataclass
class _LoadedModel:
    """Runtime state for a successfully loaded model."""
    pipeline_based:  bool
    id2label:        Dict[int, str]
    spoof_idx:       int
    bonafide_idx:    int
    polarity_source: str
    pipeline:        Optional[object] = None
    model:           Optional[object] = None
    processor:       Optional[object] = None


_models: Dict[str, _LoadedModel] = {}


# ---------------------------------------------------------------------------
# Label polarity resolution
# ---------------------------------------------------------------------------
_SPOOF_LABEL_VARIANTS    = frozenset({"spoof", "fake", "synthetic", "spoofed"})
_BONAFIDE_LABEL_VARIANTS = frozenset({"bonafide", "real", "genuine", "authentic"})


def _resolve_label_indices(
    id2label:          Dict[int, str],
    polarity_override: Optional[Tuple[int, int]],
) -> Tuple[int, int, str]:
    """Resolve spoof and bonafide output indices from a model's label map.

    Resolution order: explicit override -> label-string matching.

    Raises
    ------
    ValueError
        If neither strategy can resolve both indices.
    """
    if polarity_override is not None:
        si, bi = polarity_override
        logger.info("Label polarity resolved via override: spoof=%d, bonafide=%d.", si, bi)
        return si, bi, "override"

    spoof_idx = bonafide_idx = None
    for idx, label in id2label.items():
        lower = label.lower()
        if lower in _SPOOF_LABEL_VARIANTS:
            spoof_idx = idx
        elif lower in _BONAFIDE_LABEL_VARIANTS:
            bonafide_idx = idx

    if spoof_idx is None or bonafide_idx is None:
        raise ValueError(
            f"Cannot resolve label polarity from id2label={id2label!r}. "
            "Add a polarity_override to the ModelSpec."
        )

    logger.info(
        "Label polarity resolved via label_string: spoof=%d, bonafide=%d.",
        spoof_idx, bonafide_idx,
    )
    return spoof_idx, bonafide_idx, "label_string"


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Resample *audio* from *src_sr* to *tgt_sr* Hz."""
    if src_sr == tgt_sr:
        return audio
    try:
        from scipy.signal import resample_poly
        g = gcd(src_sr, tgt_sr)
        return resample_poly(audio, tgt_sr // g, src_sr // g).astype(np.float32)
    except Exception:
        logger.warning(
            "scipy.signal.resample_poly unavailable — falling back to linear "
            "interpolation. Quality may be reduced."
        )
        n = int(len(audio) * tgt_sr / src_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, n),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


def _prepare_wav(wav: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Clip, validate, and zero-pad a waveform for model input.

    Returns
    -------
    (wav, low_confidence)
        low_confidence is True when the input was shorter than
        MIN_WINDOW_SAMPLES and had to be padded.
    """
    wav  = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        logger.warning("Audio peak %.4f exceeds 1.0 — clipping.", peak)
        wav = np.clip(wav, -1.0, 1.0)

    low_confidence = len(wav) < MIN_WINDOW_SAMPLES
    if low_confidence:
        logger.debug(
            "Window %.3f s < %.1f s minimum — zero-padding (LOW CONFIDENCE).",
            len(wav) / TARGET_SR, MIN_WINDOW_S,
        )
        wav = np.pad(wav, (0, MIN_WINDOW_SAMPLES - len(wav)))

    return wav, low_confidence


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _infer_pipeline(
    entry: _LoadedModel, wav: np.ndarray
) -> Tuple[float, float, Dict[str, float]]:
    """Run inference using a HuggingFace Pipeline model."""
    result     = entry.pipeline(wav)
    spoof_prob = float(result["score"])
    logits     = result["logits"][0]
    si, bi     = entry.spoof_idx, entry.bonafide_idx
    cm_score   = float(logits[si] - logits[bi])
    raw_scores = {k: round(float(v), 4) for k, v in result["all_scores"].items()}
    return spoof_prob, cm_score, raw_scores


def _infer_classifier(
    entry: _LoadedModel, wav: np.ndarray
) -> Tuple[float, float, Dict[str, float]]:
    """Run inference using an AutoModelForAudioClassification model."""
    import torch

    inputs     = entry.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits_tensor = entry.model(**inputs).logits

    logits_np = logits_tensor.squeeze().cpu().numpy()
    si, bi    = entry.spoof_idx, entry.bonafide_idx
    cm_score  = float(logits_np[si] - logits_np[bi])

    shifted    = logits_np - np.max(logits_np)
    probs      = np.exp(shifted) / np.sum(np.exp(shifted))
    spoof_prob = float(np.clip(probs[si], _PROB_EPS, 1.0 - _PROB_EPS))
    raw_scores = {entry.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return spoof_prob, cm_score, raw_scores


def _infer_one(
    entry: _LoadedModel, wav: np.ndarray
) -> Tuple[float, float, Dict[str, float]]:
    """Dispatch to the correct inference backend."""
    if entry.pipeline_based:
        return _infer_pipeline(entry, wav)
    return _infer_classifier(entry, wav)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_pipeline_model(spec: ModelSpec) -> _LoadedModel:
    from transformers import pipeline as hf_pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    pipe   = hf_pipeline(
        "antispoofing", model=spec.hf_repo,
        trust_remote_code=True, device=device,
    )
    logger.info("Loaded pipeline model %s (device=%s).", spec.hf_repo, device)
    return _LoadedModel(
        pipeline_based  = True,
        pipeline        = pipe,
        id2label        = {0: "spoof", 1: "bonafide"},
        spoof_idx       = 0,
        bonafide_idx    = 1,
        polarity_source = "label_string",
    )


def _load_classifier_model(spec: ModelSpec) -> _LoadedModel:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    processor = AutoFeatureExtractor.from_pretrained(spec.hf_repo)
    model     = AutoModelForAudioClassification.from_pretrained(spec.hf_repo)
    model.eval()
    si, bi, source = _resolve_label_indices(model.config.id2label, spec.polarity_override)
    logger.info("Loaded classifier model %s (polarity: %s).", spec.hf_repo, source)
    return _LoadedModel(
        pipeline_based  = False,
        model           = model,
        processor       = processor,
        id2label        = model.config.id2label,
        spoof_idx       = si,
        bonafide_idx    = bi,
        polarity_source = source,
    )


def _load(spec: ModelSpec) -> bool:
    """Ensure *spec* is loaded into _models. No-ops if already cached."""
    if spec.key in _models:
        return True
    try:
        _models[spec.key] = (
            _load_pipeline_model(spec) if spec.pipeline_based
            else _load_classifier_model(spec)
        )
        return True
    except Exception as exc:
        logger.warning("Failed to load model %r: %s", spec.key, exc)
        return False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Outcome of running one detection model on one audio segment.

    Fields
    ------
    model_key      : Registry key of the model.
    display_name   : Human-readable model name.
    spoof_prob     : Spoof probability in [0, 1] (nan on error).
    cm_score       : logit[synthetic] - logit[bonafide]. Higher -> more synthetic.
    prediction     : 'synthetic', 'bonafide', or 'error'.
    threshold      : Decision threshold applied to spoof_prob.
    raw_scores     : Per-class probabilities keyed by label string.
    low_confidence : True if the input was zero-padded to reach minimum length.
    polarity_source: How label indices were resolved.
    error          : Error message if inference failed, else None.
    """
    model_key:       str
    display_name:    str
    spoof_prob:      float
    cm_score:        float
    prediction:      str
    threshold:       float
    raw_scores:      Dict[str, float]
    low_confidence:  bool          = False
    polarity_source: str           = "unknown"
    error:           Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model":           self.display_name,
            "spoof_prob":      round(self.spoof_prob, 4) if not np.isnan(self.spoof_prob) else None,
            "cm_score":        round(self.cm_score,   4) if not np.isnan(self.cm_score)   else None,
            "prediction":      self.prediction,
            "threshold":       self.threshold,
            "low_confidence":  self.low_confidence,
            "polarity_source": self.polarity_source,
            "error":           self.error,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _error_result(spec: ModelSpec, threshold: float, message: str) -> DetectionResult:
    return DetectionResult(
        model_key    = spec.key,
        display_name = spec.display_name,
        spoof_prob   = float("nan"),
        cm_score     = float("nan"),
        prediction   = "error",
        threshold    = threshold,
        raw_scores   = {},
        error        = message,
    )


def _run_inference(
    spec: ModelSpec, audio: np.ndarray, sr: int,
    threshold: float, max_duration_s: float,
) -> DetectionResult:
    """Resample, prepare, and score one audio array with one model."""
    max_samples   = int(max_duration_s * spec.target_sr)
    wav           = _resample(audio, sr, spec.target_sr)[:max_samples]
    wav, low_conf = _prepare_wav(wav)
    sp, cm, raw   = _infer_one(_models[spec.key], wav)

    return DetectionResult(
        model_key       = spec.key,
        display_name    = spec.display_name,
        spoof_prob      = sp,
        cm_score        = cm,
        prediction      = "synthetic" if sp >= threshold else "bonafide",
        threshold       = threshold,
        raw_scores      = raw,
        low_confidence  = low_conf,
        polarity_source = _models[spec.key].polarity_source,
    )


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------
def predict(
    audio:          np.ndarray,
    sr:             int,
    model_keys:     Optional[List[str]] = None,
    threshold:      float = 0.5,
    max_duration_s: float = 30.0,
) -> List[DetectionResult]:
    """Score *audio* with one or more detection models.

    Parameters
    ----------
    audio:
        Mono float32 waveform.
    sr:
        Sample rate of *audio* in Hz.
    model_keys:
        Models to use. Defaults to all registered models.
    threshold:
        Spoof-probability decision threshold.
    max_duration_s:
        Input is truncated to this length before inference.

    Returns
    -------
    List[DetectionResult]
        One result per requested model, in request order.
    """
    keys    = model_keys or list(_SPEC_BY_KEY.keys())
    results: List[DetectionResult] = []

    for key in keys:
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            logger.warning("predict: unknown model key %r — skipping.", key)
            continue
        if not _load(spec):
            results.append(_error_result(spec, threshold, "Model failed to load."))
            continue
        try:
            results.append(_run_inference(spec, audio, sr, threshold, max_duration_s))
        except Exception as exc:
            logger.error("Inference error %r: %s\n%s", key, exc, traceback.format_exc())
            results.append(_error_result(spec, threshold, str(exc)))

    return results


def predict_batch(
    items:          List[Tuple[str, np.ndarray, int]],
    model_keys:     Optional[List[str]] = None,
    threshold:      float = 0.5,
    max_duration_s: float = 30.0,
    progress_cb:    Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Dict[str, DetectionResult]]:
    """Score a list of (uid, audio, sr) items with one or more models.

    Parameters
    ----------
    items:
        List of (uid, audio, sr) tuples.
    model_keys:
        Models to use. Defaults to all registered models.
    threshold:
        Spoof-probability decision threshold.
    max_duration_s:
        Input is truncated to this length before inference.
    progress_cb:
        Optional callback(completed, total) where total is
        len(items) * len(keys).

    Returns
    -------
    Dict[str, Dict[str, DetectionResult]]
        result[uid][model_key] -> DetectionResult.
    """
    keys  = model_keys or list(_SPEC_BY_KEY.keys())
    out:  Dict[str, Dict[str, DetectionResult]] = {uid: {} for uid, _, _ in items}
    total = len(items) * len(keys)
    done  = 0

    for key in keys:
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            logger.warning("predict_batch: unknown model key %r — skipping.", key)
            continue
        if not _load(spec):
            err = _error_result(spec, threshold, "Model failed to load.")
            for uid, _, _ in items:
                out[uid][key] = err
            done += len(items)
            if progress_cb:
                progress_cb(done, total)
            continue

        for uid, audio, sr in items:
            try:
                out[uid][key] = _run_inference(spec, audio, sr, threshold, max_duration_s)
            except Exception as exc:
                logger.error(
                    "Inference error uid=%r model=%r: %s\n%s",
                    uid, key, exc, traceback.format_exc(),
                )
                out[uid][key] = _error_result(spec, threshold, str(exc))
            done += 1
            if progress_cb:
                progress_cb(done, total)

    return out


# ---------------------------------------------------------------------------
# Score cache
# ---------------------------------------------------------------------------
def pipeline_cache_key(steps: list) -> str:
    """Return a short deterministic identifier for a manipulation pipeline.

    Used to name M-BM cache files distinctly from the unmanipulated BM cache.
    Two pipelines with identical step names, params, region_mode, and
    boundary_pad_s produce the same key. An empty pipeline returns
    'identity'.

    UID-order sensitivity
    ---------------------
    'White Noise' and 'Soundscape Mix' steps receive a per-utterance seed /
    noise offset derived from uid_index (the position of the UID in the
    sorted dataset list). The cache is therefore only valid for a fixed,
    stable UID ordering. 

    Parameters
    ----------
    steps:
        List of ManipulationStep objects. Expected attributes: name,
        params, region_mode, boundary_pad_s.

    Returns
    -------
    str
        'identity' or a 12-character hex digest.
    """
    if not steps:
        return "identity"

    _UID_ORDER_SENSITIVE = frozenset({"White Noise", "Soundscape Mix"})
    uid_sensitive = any(s.name in _UID_ORDER_SENSITIVE for s in steps)

    parts = [
        f"{s.name}|{sorted(s.params.items())}|{s.region_mode}|{s.boundary_pad_s}"
        for s in steps
    ]
    if uid_sensitive:
        parts.append("uid_order_sensitive:true")

    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def score_and_cache(
    uids:           List[str],
    wav_index:      Dict[str, Path],
    utterances:     dict,
    dataset_key:    str,
    model_key:      str,
    cache_dir:      Path,
    threshold:      float = 0.5,
    max_duration_s: float = 30.0,
    existing_scores: Optional[Dict[str, dict]] = None,
    progress_cb:    Optional[Callable[[int, int], None]] = None,
) -> Dict[str, dict]:
    """Score *uids* with *model_key* and persist results to *cache_dir*.

    Writes a partial cache file every CACHE_BATCH_SIZE utterances so
    that a crash does not lose all progress. On completion the partial file
    is atomically renamed to the final cache file.

    Already-scored UIDs (from *existing_scores*) are preserved and not
    re-scored.

    Parameters
    ----------
    uids:
        Ordered list of UIDs to score.
    wav_index:
        Mapping of UID -> audio file Path.
    utterances:
        Dict mapping UID -> Utterance, used to look up ground-truth labels.
    dataset_key:
        Registry key of the dataset (used in the cache filename).
    model_key:
        Registry key of the model to use.
    cache_dir:
        Directory where cache files are written.
    threshold:
        Detection threshold (stored in cache metadata for traceability).
    max_duration_s:
        Audio is truncated to this length before inference.
    existing_scores:
        Pre-loaded scores dict to resume from. UIDs present here are skipped.
    progress_cb:
        Optional callback(completed, total) where total = len(uids).

    Returns
    -------
    Dict[str, dict]
        Merged scores dict: {uid: {"cm": float, "gt": str, "split": str}}.
        Includes both newly scored and pre-existing entries.
    """
    spec = _SPEC_BY_KEY.get(model_key)
    if spec is None:
        raise ValueError(f"Unknown model key: {model_key!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file   = cache_dir / f"{dataset_key}_{model_key}.json"
    partial_file = cache_dir / f"{dataset_key}_{model_key}.partial.json"

    scores_out     = dict(existing_scores or {})
    already_scored = set(scores_out.keys())
    to_score       = [uid for uid in uids if uid not in already_scored]

    parameters = {
        "scoring":        "utterance_level",
        "max_duration_s": max_duration_s,
        "threshold":      threshold,
        "cm_convention":  "logit_synthetic - logit_bonafide",
    }

    for i, uid in enumerate(to_score):
        # Load audio.
        wav_path = wav_index.get(uid)
        if wav_path is None:
            logger.warning("score_and_cache: no audio file for UID %r — skipping.", uid)
            if progress_cb:
                progress_cb(i + 1, len(to_score))
            continue

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            logger.warning("score_and_cache: cannot read %s: %s", wav_path, exc)
            if progress_cb:
                progress_cb(i + 1, len(to_score))
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        utt = utterances.get(uid)
        gt  = utt.effective_label if utt else "unknown"

        # Run inference.
        try:
            results = predict(
                audio, sr,
                model_keys     = [model_key],
                threshold      = threshold,
                max_duration_s = max_duration_s,
            )
            for r in results:
                if not r.error:
                    scores_out[uid] = {"cm": r.cm_score, "gt": gt, "split": "train"}
        except Exception as exc:
            logger.warning("score_and_cache: inference error for %r: %s", uid, exc)

        # Periodic partial save — crash-safe progress persistence.
        if (i + 1) % CACHE_BATCH_SIZE == 0 or (i + 1) == len(to_score):
            _write_cache(
                partial_file, dataset_key, model_key,
                spec.display_name, parameters, scores_out,
            )

        if progress_cb:
            progress_cb(i + 1, len(to_score))

    if partial_file.exists():
        partial_file.rename(cache_file)

    return scores_out


def _write_cache(
    path:         Path,
    dataset_key:  str,
    model_key:    str,
    display_name: str,
    parameters:   dict,
    scores:       dict,
) -> None:
    """Serialise a scores dict to *path* as JSON."""
    path.write_text(json.dumps({
        "dataset_key":        dataset_key,
        "model_key":          model_key,
        "model_display_name": display_name,
        "scored_at":          datetime.now().isoformat(timespec="seconds"),
        "parameters":         parameters,
        "scores":             scores,
    }, indent=2))


def load_cache(
    dataset_key: str,
    cache_dir:   Path,
    model_keys:  Optional[List[str]] = None,
    split:       str = "train",
) -> Dict[str, List[dict]]:
    """Load and merge CM scores from cache files for one or more models.

    Parameters
    ----------
    dataset_key:
        Registry key of the dataset.
    cache_dir:
        Directory containing cache files.
    model_keys:
        Models to load. Defaults to all registered models.
    split:
        Only entries with this split tag are included (default: 'train').

    Returns
    -------
    Dict[str, List[dict]]
        {uid: [{"model": key, "cm": float, "gt": str}, ...]}.
        A UID appears only once per model but may appear for multiple models.
    """
    keys   = model_keys or list(_SPEC_BY_KEY.keys())
    merged: Dict[str, List[dict]] = {}

    for spec in MODEL_SPECS:
        if spec.key not in keys:
            continue

        cache_file   = cache_dir / f"{dataset_key}_{spec.key}.json"
        partial_file = cache_dir / f"{dataset_key}_{spec.key}.partial.json"
        source       = cache_file if cache_file.exists() else (
                       partial_file if partial_file.exists() else None)

        if source is None:
            continue

        try:
            data = json.loads(source.read_text())
        except Exception as exc:
            logger.warning("load_cache: cannot read %s: %s", source, exc)
            continue

        for uid, entry in data.get("scores", {}).items():
            if not isinstance(entry, dict):
                continue   # skip legacy windowed-format entries
            if entry.get("split") != split:
                continue
            merged.setdefault(uid, []).append({
                "model": spec.key,
                "cm":    entry["cm"],
                "gt":    entry["gt"],
            })

    return merged


def cache_status(
    dataset_key: str,
    cache_dir:   Path,
) -> List[dict]:
    """Return a status row for each registered model's cache file.

    Returns
    -------
    List[dict]
        Each dict has keys: model_key, display_name, status,
        n_files, scored_at, is_complete.
        status is one of 'complete', 'partial', or 'missing'.
    """
    rows = []
    for spec in MODEL_SPECS:
        cache_file   = cache_dir / f"{dataset_key}_{spec.key}.json"
        partial_file = cache_dir / f"{dataset_key}_{spec.key}.partial.json"

        if cache_file.exists():
            try:
                meta = json.loads(cache_file.read_text())
                rows.append({
                    "model_key":    spec.key,
                    "display_name": spec.display_name,
                    "status":       "complete",
                    "n_files":      len(meta.get("scores", {})),
                    "scored_at":    meta.get("scored_at", "unknown"),
                    "is_complete":  True,
                })
            except Exception:
                rows.append(_missing_status_row(spec, "corrupt"))
        elif partial_file.exists():
            try:
                meta = json.loads(partial_file.read_text())
                rows.append({
                    "model_key":    spec.key,
                    "display_name": spec.display_name,
                    "status":       "partial",
                    "n_files":      len(meta.get("scores", {})),
                    "scored_at":    meta.get("scored_at", "unknown"),
                    "is_complete":  False,
                })
            except Exception:
                rows.append(_missing_status_row(spec, "corrupt"))
        else:
            rows.append(_missing_status_row(spec, "missing"))

    return rows


def _missing_status_row(spec: ModelSpec, status: str) -> dict:
    return {
        "model_key":    spec.key,
        "display_name": spec.display_name,
        "status":       status,
        "n_files":      0,
        "scored_at":    "—",
        "is_complete":  False,
    }


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------
def is_loaded(key: str) -> bool:
    """Return True if the model with *key* is currently loaded."""
    return key in _models


def loaded_model_keys() -> List[str]:
    """Return the keys of all currently loaded models."""
    return list(_models.keys())


def get_label_polarity(key: str) -> Optional[dict]:
    """Return label polarity metadata for a loaded model, or None."""
    entry = _models.get(key)
    if entry is None:
        return None
    return {
        "spoof_label":          entry.id2label[entry.spoof_idx],
        "spoof_idx":            entry.spoof_idx,
        "bonafide_label":       entry.id2label[entry.bonafide_idx],
        "bonafide_idx":         entry.bonafide_idx,
        "polarity_source":      entry.polarity_source,
    }
   
# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------
def compute_det_curve(scores: np.ndarray, labels: np.ndarray):
    """FAR/FRR at every unique threshold. Returns (thresholds, far, frr)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_synth = int(np.sum(labels == 1))
    n_bonaf = int(np.sum(labels == 0))

    if n_synth == 0 or n_bonaf == 0:
        raise ValueError(
            f"Need both synthetic and bonafide samples. "
            f"Got n_synth={n_synth}, n_bonaf={n_bonaf}."
        )

    unique_t = np.unique(scores)  # ascending

    far = np.array([
        np.sum((labels == 0) & (scores >= t)) / n_bonaf
        for t in unique_t
    ], dtype=np.float64)

    frr = np.array([
        np.sum((labels == 1) & (scores < t)) / n_synth
        for t in unique_t
    ], dtype=np.float64)

    thresholds = np.concatenate([[-np.inf], unique_t, [np.inf]])
    far        = np.concatenate([[1.0], far, [0.0]])
    frr        = np.concatenate([[0.0], frr, [1.0]])

    return thresholds, far, frr    
    
def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """EER by linear interpolation at FAR=FRR crossing. Returns (eer, threshold, far, frr)."""
    thresholds, far, frr = compute_det_curve(scores, labels)
    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        i = int(np.argmin(np.abs(diff)))
        eer = float((far[i] + frr[i]) / 2)
        return eer, float(thresholds[i]), float(far[i]), float(frr[i])

    i = int(sign_changes[0])

    d0, d1 = diff[i], diff[i + 1]
    if d0 == d1:
        alpha = 0.5
    else:
        alpha = d0 / (d0 - d1)

    eer_threshold = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
    far_at_eer    = float(far[i]  + alpha * (far[i + 1]  - far[i]))
    frr_at_eer    = float(frr[i]  + alpha * (frr[i + 1]  - frr[i]))
    eer           = float((far_at_eer + frr_at_eer) / 2)

    return eer, eer_threshold, far_at_eer, frr_at_eer

def compute_cllr(lr_values: np.ndarray, labels: np.ndarray) -> float:
    """Compute the log-likelihood-ratio cost (Cllr) — a calibration quality metric.

    Cllr measures how well-calibrated the LR system is, combining both
    discrimination and calibration into a single scalar. A perfectly
    calibrated system with perfect discrimination gives Cllr = 0.
    A system no better than chance gives Cllr = 1.

        Cllr = 0.5 * [ mean_synth(log2(1 + 1/LR)) + mean_bonaf(log2(1 + LR)) ]

    Parameters
    ----------
    lr_values:
        Array of LR values, one per utterance. Must be strictly positive.
    labels:
        Binary labels: 1 = synthetic, 0 = bonafide.

    Returns
    -------
    float
        Cllr in [0, ∞). Values above 1 indicate the system is
        worse than a naive equal-prior LR=1 system.

    Raises
    ------
    ValueError
        If either class is absent, or if any LR is non-positive.
    """
    lrs  = np.asarray(lr_values,  dtype=np.float64)
    labs = np.asarray(labels,     dtype=np.int32)

    if np.any(lrs <= 0):
        raise ValueError("All LR values must be strictly positive.")

    synth_lrs = lrs[labs == 1]
    bonaf_lrs = lrs[labs == 0]

    if len(synth_lrs) == 0 or len(bonaf_lrs) == 0:
        raise ValueError(
            f"Need both classes. Got n_synthetic={len(synth_lrs)}, n_bonafide={len(bonaf_lrs)}."
        )

    c_synth = float(np.mean(np.log2(1.0 + 1.0 / synth_lrs)))
    c_bonaf = float(np.mean(np.log2(1.0 + bonaf_lrs)))
    return 0.5 * (c_synth + c_bonaf)
