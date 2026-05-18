"""Audio manipulation pipeline for forensic synthetic speech investigation.

Manipulations
----------------------
  Telephone Filter  — ITU-T G.711 bandpass 300-3400 Hz      
  White Noise       — AWGN at 20 dB SNR                     
  Resampling        — 8 kHz down/up round-trip              
  Soundscape Mix    — ISD London urban noise at 20 dB SNR    

Preset pipelines
----------------
Use PRESET_PIPELINES to populate the MBM cache directly without
building pipelines manually in the sidebar. Each preset has a short
alphanumeric key used in cache filenames so cache files are immediately
identifiable.
"""

import io
import logging
import os
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import resample_poly

import config as _config

logger = logging.getLogger(__name__)


# ===========================================================================
# Internal helpers
# ===========================================================================
def _time_to_sample(t: float, sr: int) -> int:
    return int(t * sr)


def _apply_to_region(audio, sr, fn, start, end, **kw):
    """Apply *fn* to the time region [start, end] seconds and stitch back."""
    s = _time_to_sample(start, sr)
    e = min(_time_to_sample(end, sr), len(audio))
    out = audio.copy()
    out[s:e] = fn(audio[s:e], sr, **kw)
    return out


def _resample_signal(audio, src_sr, tgt_sr):
    """Polyphase rational resampler (scipy.signal.resample_poly).

    Kaiser-windowed anti-aliasing FIR, parameters chosen automatically
    by scipy. This is the resampler used in CLAD and ASVspoof pipelines.
    """
    if src_sr == tgt_sr:
        return audio
    g = gcd(int(src_sr), int(tgt_sr))
    return resample_poly(
        audio.astype(np.float64), tgt_sr // g, src_sr // g
    ).astype(np.float32)


def _load_noise_files(soundscape_dir: str) -> list:
    """Return a sorted list of WAV paths from *soundscape_dir*."""
    d = Path(soundscape_dir)
    if not d.exists():
        raise RuntimeError(
            f"Soundscape directory not found: {d}\n"
            "Download the ISD London zip from "
            "https://zenodo.org/records/10672568 and set SOUNDSCAPE_DIR."
        )
    wavs = sorted(d.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(
            f"No WAV files found in {d}. "
            "Check that the ISD London archive has been extracted there."
        )
    return wavs


def _load_noise_segment(
    noise_path: Path,
    n_samples: int,
    tgt_sr: int,
    offset_index: int = 0,
) -> np.ndarray:
    """Load a noise file, resample to *tgt_sr*, return exactly *n_samples*.

    Starts at offset_index * n_samples within the file so different
    utterances assigned to the same noise file get different segments.
    Tiles the file if it is shorter than the required length.
    """
    noise_audio, noise_sr = sf.read(
        str(noise_path), dtype="float32", always_2d=False
    )
    if noise_audio.ndim > 1:
        noise_audio = noise_audio.mean(axis=1)

    noise_audio = _resample_signal(noise_audio, noise_sr, tgt_sr)

    if len(noise_audio) < n_samples * 2:
        repeats = int(np.ceil((n_samples * 2) / max(1, len(noise_audio))))
        noise_audio = np.tile(noise_audio, repeats)

    max_start   = len(noise_audio) - n_samples
    start       = (offset_index * n_samples) % max(1, max_start)
    return noise_audio[start: start + n_samples].astype(np.float32)


# ===========================================================================
# Manipulation Functions
# ===========================================================================
def resampling(audio, sr, target_sr=8000, **_):
    """Down-sample to 8 kHz then up-sample back to the original rate.

    Permanently removes all spectral energy above 4 kHz (the Nyquist
    frequency of the 8 kHz intermediate rate). Models telephone-quality
    transmission or storage.

    Implementation: two _resample_signal calls. Output trimmed or padded
    to the original length because resample_poly can return ±1 sample.

    Fixed parameter: 8000 Hz intermediate rate.
    """
    n   = len(audio)
    tgt = max(4000, int(target_sr))
    out = _resample_signal(_resample_signal(audio, sr, tgt), tgt, sr)
    if len(out) > n:
        return out[:n]
    if len(out) < n:
        return np.pad(out, (0, n - len(out)))
    return out


def telephone_filter(audio, sr, **_):
    """Butterworth bandpass 300-3400 Hz, order 4 — ITU-T G.711 PSTN band.

    Removes energy below 300 Hz and above 3400 Hz, matching the frequency
    response of the public switched telephone network. Implemented in SOS
    form for numerical stability (standard practice for digital filters).

    Fixed parameters: 300-3400 Hz, order 4. 
    """
    nyq = sr / 2.0
    lo  = 300.0 / nyq
    hi  = min(3400.0, nyq - 1.0) / nyq
    sos = signal.butter(4, [lo, hi], btype="band", output="sos")
    return np.clip(
        signal.sosfilt(sos, audio.astype(np.float64)), -1.0, 1.0
    ).astype(np.float32)


def white_noise(audio, sr, snr_db=20.0, seed=42, **_):
    """Additive white Gaussian noise at 20 dB SNR.

    Formula:
        signal_rms   = sqrt(mean(audio^2))
        noise_rms    = signal_rms / 10^(snr_db / 20)
        noise        = randn(N) scaled to noise_rms
        output       = clip(audio + noise, -1, 1)

    SNR is in the power sense. When called via apply_pipeline, seed is set
    to uid_index so each utterance receives a distinct but deterministic
    noise realisation. The default seed=42 applies only to direct calls
    (e.g. target-utterance inspection).
    Silent inputs (RMS < 1e-4) are returned unchanged.

    Fixed parameters: 20 dB SNR.

    Reference:
        Tak et al. (2022). RawBoost. ICASSP 2022.
    """
    rng = np.random.default_rng(int(seed))
    x   = audio.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-4:
        logger.warning("white_noise: near-silent input — skipping.")
        return audio.astype(np.float32)
    noise_rms = rms / (10.0 ** (float(snr_db) / 20.0))
    noise     = rng.standard_normal(len(x))
    noise     = noise / (float(np.sqrt(np.mean(noise ** 2))) + 1e-9) * noise_rms
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def soundscape_mix(audio, sr, snr_db=20.0, noise_index=0,
                   soundscape_dir=None, **_):
    """Mix speech with an ISD London urban soundscape segment at 20 dB SNR.

    Uses the same RMS-based SNR formula as white_noise.

    Pairing: each utterance is assigned a noise file by
        noise_file = sorted_noise_files[noise_index % n_noise_files]
    and a segment offset by
        start = (noise_index * n_samples) % file_length
    so no two utterances share the same noise segment.

    Fixed parameters: 20 dB SNR (matching white_noise).

    Reference:
        Ragnarsdóttir et al. (2024). ISD. Zenodo:10672568. London subset.
        Tak et al. (2022). RawBoost. ICASSP 2022.
    """
    sdir        = soundscape_dir or _config.SOUNDSCAPE_DIR
    noise_files = _load_noise_files(sdir)
    noise_path  = noise_files[int(noise_index) % len(noise_files)]
    noise       = _load_noise_segment(
        noise_path, len(audio), sr, offset_index=int(noise_index)
    )

    x   = audio.astype(np.float64)
    n   = noise.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-4:
        logger.warning("soundscape_mix: near-silent input — skipping.")
        return audio.astype(np.float32)

    noise_rms_current = float(np.sqrt(np.mean(n ** 2))) + 1e-9
    target_noise_rms  = rms / (10.0 ** (float(snr_db) / 20.0))
    n_scaled          = n * (target_noise_rms / noise_rms_current)

    return np.clip(x + n_scaled, -1.0, 1.0).astype(np.float32)


# ===========================================================================
# Registry
# ===========================================================================
MANIPULATIONS = {
    "Resampling": {
        "fn":                resampling,
        "category":          "Signal Degradation",
        "description":       (
            "Down-sample to 8 kHz then up-sample back. "
        ),
        "ref":               "CLAD (2024), arXiv:2404.15854.",
        "params":            {"target_sr": 8000},
        "whole_signal_only": True,
    },
    "Telephone Filter": {
        "fn":                telephone_filter,
        "category":          "Signal Degradation",
        "description":       (
            "Butterworth bandpass 300-3400 Hz, order 4. "
        ),
        "ref":               "ITU-T G.711 (1972). ASVspoof 2021 LA.",
        "params":            {},
        "whole_signal_only": False,
    },
    "White Noise": {
        "fn":                white_noise,
        "category":          "Environment Simulation",
        "description":       (
            "Additive white Gaussian noise at 20 dB SNR. "
        ),
        "ref":               "Tak et al. (2022). RawBoost. ICASSP 2022.",
        "params":            {"snr_db": 20.0, "seed": 42},
        "whole_signal_only": False,
    },
    "Soundscape Mix": {
        "fn":                soundscape_mix,
        "category":          "Environment Simulation",
        "description":       (
            "ISD London urban soundscape at 20 dB SNR. "
        ),
        "ref":               (
            "Ragnarsdóttir et al. (2024). ISD. Zenodo:10672568. "
            "Tak et al. (2022). RawBoost. ICASSP 2022."
        ),
        "params":            {"snr_db": 20.0},
        "whole_signal_only": False,
    },
}


# ===========================================================================
# Preset pipelines
# ===========================================================================
PRESET_PIPELINE_DEFS: list = [
    #  Singles 
    {
        "key":         "TF",
        "display":     "Telephone Filter",
        "steps":       [("Telephone Filter", {})],
        "description": "ITU-T G.711 bandpass only. Weakest single manipulation.",
    },
    {
        "key":         "WN",
        "display":     "White Noise",
        "steps":       [("White Noise", {})],
        "description": "AWGN 20 dB SNR. Best single manipulation.",
    },
    {
        "key":         "RS",
        "display":     "Resampling",
        "steps":       [("Resampling", {})],
        "description": "8 kHz down/up round-trip only.",
    },
    {
        "key":         "SM",
        "display":     "Soundscape Mix",
        "steps":       [("Soundscape Mix", {})],
        "description": "ISD London urban noise 20 dB SNR only.",
    },
    {
        "key":         "TF+WN",
        "display":     "Telephone Filter + White Noise",
        "steps":       [("Telephone Filter", {}), ("White Noise", {})],
        "description": (
            "Core pair — recommended primary MBM condition."
        ),
    },
    {
        "key":         "TF+WN+SM",
        "display":     "Telephone Filter + White Noise + Soundscape Mix",
        "steps":       [
            ("Telephone Filter", {}),
            ("White Noise",      {}),
            ("Soundscape Mix",   {}),
        ],
        "description": (
            "Tests whether real urban noise adds evidence over synthetic noise. "),
    },
    {
        "key":         "RS+TF+WN",
        "display":     "Resampling + Telephone Filter + White Noise",
        "steps":       [
            ("Resampling",       {}),
            ("Telephone Filter", {}),
            ("White Noise",      {}),
        ],
        "description": (
            "Tests whether spectral truncation before filtering adds value."),
    },
    {
        "key":         "RS+TF+WN+SM",
        "display":     "Resampling + Telephone Filter + White Noise + Soundscape Mix",
        "steps":       [
            ("Resampling",       {}),
            ("Telephone Filter", {}),
            ("White Noise",      {}),
            ("Soundscape Mix",   {}),
        ],
        "description": (
            "Best overall Δ synthetic in 63-combination mini scoring. "
            "Full four-step pipeline."
        ),
    },
]

# ===========================================================================
# Pipeline infrastructure
# ===========================================================================
@dataclass
class ManipulationStep:
    name:           str
    params:         dict   = field(default_factory=dict)
    region:         object = None
    region_mode:    str    = "whole"
    boundary_pad_s: float  = 0.15


PRESET_PIPELINES: dict = {}
for _pdef in PRESET_PIPELINE_DEFS:
    PRESET_PIPELINES[_pdef["key"]] = {
        "display":     _pdef["display"],
        "description": _pdef["description"],
        "steps": [
            ManipulationStep(name=name, params=dict(params))
            for name, params in _pdef["steps"]
        ],
    }

MANIPULATION_CATEGORIES = [
    "Signal Degradation",
    "Environment Simulation",
]

CATEGORY_LABELS = {
    "Signal Degradation":    "Signal Degradation",
    "Environment Simulation": "Environment Simulation",
}

MANIPULATIONS_BY_CATEGORY = {cat: [] for cat in MANIPULATION_CATEGORIES}
for _name, _info in MANIPULATIONS.items():
    _cat = _info.get("category", "")
    if _cat in MANIPULATIONS_BY_CATEGORY:
        MANIPULATIONS_BY_CATEGORY[_cat].append(_name)

PARAM_SCHEMA:  dict = {}
PARAM_PRESETS: dict = {}

REGION_MODES = ["whole", "manual", "splice_boundaries"]

REGION_MODE_LABELS = {
    "whole":             "Whole audio",
    "manual":            "Manual time range",
    "splice_boundaries": "Splice boundaries (window around each transition)",
}


def _resolve_regions(step, duration_s, segments):
    """Return a list of (start_s, end_s) time regions for this step."""
    mode = step.region_mode
    if mode == "whole":
        return []
    if mode == "manual":
        return [step.region] if step.region else []
    if not segments:
        return []
    if mode == "spoof_segments":
        return [(s.start, s.end) for s in segments if s.label == "spoof"]
    if mode == "bonafide_segments":
        return [(s.start, s.end) for s in segments if s.label == "bonafide"]
    if mode == "splice_boundaries":
        pad        = step.boundary_pad_s
        boundaries = []
        for i in range(len(segments) - 1):
            if segments[i].label != segments[i + 1].label:
                t = segments[i].end
                boundaries.append(
                    (max(0.0, t - pad), min(duration_s, t + pad))
                )
        if segments and segments[0].label == "spoof":
            boundaries.append(
                (0.0, min(duration_s, segments[0].start + pad))
            )
        if segments and segments[-1].label == "spoof":
            boundaries.append(
                (max(0.0, segments[-1].end - pad), duration_s)
            )
        return boundaries
    return []


def apply_pipeline(audio, sr, steps, utterance=None, uid_index=0):
    """Apply a sequence of ManipulationSteps to audio in order.

    Parameters
    ----------
    audio     : np.ndarray  mono float32
    sr        : int         sample rate in Hz
    steps     : list        of ManipulationStep
    utterance : Utterance   used for splice_boundary region mode
    uid_index : int         position of this utterance in the sorted UID
                            list, passed to soundscape_mix as noise_index
                            for deterministic noise pairing.

    Output is always the same length as the input.
    """
    original_len = len(audio)
    duration_s   = original_len / sr
    segments     = utterance.segments if utterance is not None else []
    out          = audio.astype(np.float32)

    for step in steps:
        if step.name not in MANIPULATIONS:
            logger.warning("Unknown manipulation '%s' — skipped.", step.name)
            continue

        info              = MANIPULATIONS[step.name]
        fn                = info["fn"]
        whole_signal_only = info.get("whole_signal_only", False)

        # Pass uid_index for deterministic but utterance-specific randomness.
        params = {**info["params"], **step.params}
        if step.name == "Soundscape Mix":
            params["noise_index"] = uid_index
        if step.name == "White Noise":
            params["seed"] = uid_index

        if whole_signal_only and step.region_mode != "whole":
            logger.warning(
                "'%s' is a whole-signal operation — applying to whole "
                "signal regardless of region mode.", step.name
            )
            out = fn(out, sr, **params)
        else:
            regions = _resolve_regions(step, duration_s, segments)
            if not regions:
                out = fn(out, sr, **params)
            else:
                for t_start, t_end in regions:
                    out = _apply_to_region(
                        out, sr, fn, t_start, t_end, **params
                    )

        if len(out) > original_len:
            out = out[:original_len]
        elif len(out) < original_len:
            out = np.pad(out, (0, original_len - len(out)))

    return out