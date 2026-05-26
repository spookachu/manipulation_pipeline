"""Shared utility functions for the Voice Deepfake Forensic Pipeline.

Provides:
  - Core data types : Segment, Utterance
  - Audio I/O       : audio_to_bytes, load_wav_bytes
  - UI helpers      : waveform_player, label_badge
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.io import wavfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A time-bounded region within an utterance with a spoofing label."""
    start: float
    end:   float
    label: str


@dataclass
class Utterance:
    """A parsed audio utterance with metadata and optional segment-level detail.

    Fields
    ------
    uid:       Unique utterance identifier.
    duration:  Duration in seconds. May be 0.0 when the audio file was not
               available at parse time (e.g. subset downloads).
    label:     Ground-truth utterance label ('bonafide', 'spoof', 'synthetic').
    segments:  Optional list of time-bounded segment labels.
    attack_id: Spoofing system identifier (e.g. ASVspoof A07-A19).
               None for datasets that do not carry this field.
    codec:     Transmission codec condition (e.g. ASVspoof 'alaw', 'none').
               None for datasets that do not carry this field.
    """
    uid:       str
    duration:  float
    label:     str
    segments:  List[Segment] = field(default_factory=list)
    attack_id: Optional[str] = None
    codec:     Optional[str] = None

    @property
    def is_partial(self) -> bool:
        """True if the utterance contains a mix of bonafide and synthetic segments."""
        return len({s.label for s in self.segments}) > 1

    @property
    def effective_label(self) -> str:
        """Canonical three-way label: 'bonafide', 'synthetic', or 'partial synthetic'.

        Rules
        -----
        - Utterances labelled 'bonafide' are always bonafide.
        - Utterances labelled 'spoof' or 'synthetic' that contain a mix of
          segment labels are 'partial synthetic'.
        - All other spoof/synthetic utterances are 'synthetic'.
        """
        if self.label == "bonafide":
            return "bonafide"
        if self.label in ("spoof", "synthetic"):
            return "partial synthetic" if self.is_partial else "synthetic"
        return "synthetic"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def label_badge(effective_label: str) -> str:
    """Return a bracketed display string for an effective label.

    Parameters
    ----------
    effective_label:
        One of 'bonafide', 'synthetic', or 'partial synthetic'.
        Unknown values pass through unchanged.

    Returns
    -------
    str
        e.g. '[bonafide]', '[synthetic]', '[partial synthetic]'.
    """
    _DISPLAY = {
        "bonafide":          "bonafide",
        "synthetic":         "synthetic",
        "partial synthetic": "partial synthetic",
    }
    return f"[{_DISPLAY.get(effective_label, effective_label)}]"


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def audio_to_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode a float32 audio array as a metadata-stripped 16-bit WAV.

    Parameters
    ----------
    audio:
        Mono or stereo float32 array. Values outside [-1, 1] are clipped.
    sr:
        Sample rate in Hz.

    Returns
    -------
    bytes
        Raw WAV bytes with no embedded metadata.
    """
    pcm = np.clip(audio, -1.0, 1.0)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm_int16)
    return buf.getvalue()


def load_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV/FLAC/MP3 bytes into a mono float32 array.

    Parameters
    ----------
    data:
        Raw audio file bytes (any format supported by soundfile).

    Returns
    -------
    Tuple[np.ndarray, int]
        (audio, sample_rate) where *audio* is mono float32.
    """
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def waveform_player(
    audio: np.ndarray,
    sr: int,
    label: str | None = None,
) -> None:
    """Render a waveform plot and audio player.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 waveform array. Values should lie in [-1, 1];
        samples outside this range will appear beyond the clipping
        threshold markers.
    sr : int
        Sample rate in Hz.
    label : str or None, optional
        Short caption displayed above the waveform. Typically used to
        distinguish 'Original' from 'Processed' in side-by-side views.
        Defaults to None (no caption rendered).

    Notes
    -----
    Time axis granularity
        Tick spacing adapts automatically to file duration:
        ≤ 5 s  -> 0.1 s ticks
        ≤ 30 s -> 0.5 s ticks
        > 30 s -> 1 s ticks
    """
  
    import streamlit as st
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if label:
        st.caption(label)

    duration_s = len(audio) / sr
    times = np.linspace(0, duration_s, num=len(audio))

    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.plot(times, audio, color="#1a1a1a", linewidth=0.5, alpha=0.85)
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    if duration_s <= 5:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    elif duration_s <= 30:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.axhline(1.0,  color="#f88", linewidth=0.4, linestyle="--", zorder=0)
    ax.axhline(-1.0, color="#f88", linewidth=0.4, linestyle="--", zorder=0)
    ax.grid(axis="x", color="#eee", linewidth=0.4, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_edgecolor("#ccc")

    fig.tight_layout(pad=0.4)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.audio(audio_to_bytes(audio, sr), format="audio/wav")