"""
Shared utility functions for the Voice Deepfake Forensic Pipeline.

Provides:
  - Core data types: Segment, Utterance
  - Label file parsing: parse_label_file
  - UID utilities: generation_model
  - Binning helpers: duration_bin, n_synth_segs_bin
  - Display helpers: label_badge
  - Audio I/O: audio_to_bytes, load_wav_bytes

Constants:
    DURATION_SHORT_THRESHOLD  : Upper bound (exclusive) for 'short' duration bin (s).
    DURATION_LONG_THRESHOLD   : Lower bound (inclusive) for 'long' duration bin (s).
    SYNTH_SEG_MEDIAN_THRESHOLD: Upper bound (inclusive) for '1-3' synthetic segments bin.
"""

import io, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import base64

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DURATION_SHORT_THRESHOLD   = 3.0   # seconds
DURATION_LONG_THRESHOLD    = 8.0   # seconds
SYNTH_SEG_MEDIAN_THRESHOLD = 3     # segments

# Valid utterance-level labels in the LlamaPartialSpoof format.
_UTT_LABELS = frozenset({"bonafide", "synthetic", "spoof"})
_SEGMENT_RE = re.compile(r"^(\d+\.?\d*)-(\d+\.?\d*)-(\w+)$")

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
    """A parsed audio utterance with metadata and optional segment-level detail."""

    uid:      str
    duration: float
    label:    str
    segments: List[Segment] = field(default_factory=list)

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

    @property
    def tts_model(self) -> Optional[str]:
        """TTS model identifier extracted from the UID, or None for bonafide."""
        model = generation_model(self.uid)
        return None if model == "bonafide" else model


# ---------------------------------------------------------------------------
# Label file parsing
# ---------------------------------------------------------------------------
def _is_valid_uid(tok: str) -> bool:
    """Return True if *tok* looks like a LlamaPartialSpoof UID."""
    return (
        bool(tok)
        and tok[0].isalpha()
        and "_" in tok
        and not _SEGMENT_RE.match(tok)
    )


def _parse_segment(tok: str) -> Segment:
    """Parse a segment token '<start>-<end>-<label>' into a Segment.

    Raises
    ------
    ValueError
        If the token does not match the expected format.
    """
    m = _SEGMENT_RE.match(tok)
    if not m:
        raise ValueError(f"Invalid segment token: {tok!r}")
    return Segment(float(m.group(1)), float(m.group(2)), m.group(3))


def parse_label_file(path: Union[str, Path]) -> Dict[str, Utterance]:
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
            uid, duration_str, utt_label = tokens[0], tokens[1], tokens[2]

            if not _is_valid_uid(uid):
                logger.debug("Line %d: invalid UID %r, skipping.", line_num, uid)
                continue

            try:
                duration = float(duration_str)
            except ValueError:
                logger.debug("Line %d: non-numeric duration %r, skipping.", line_num, duration_str)
                continue

            if utt_label not in _UTT_LABELS:
                logger.debug("Line %d: unknown label %r, skipping.", line_num, utt_label)
                continue

            segments: List[Segment] = []
            for tok in tokens[3:]:
                if _SEGMENT_RE.match(tok):
                    try:
                        segments.append(_parse_segment(tok))
                    except ValueError as exc:
                        logger.warning("Line %d: skipping invalid segment %r — %s", line_num, tok, exc)

            utterances[uid] = Utterance(uid=uid, duration=duration, label=utt_label, segments=segments)

    return utterances


# ---------------------------------------------------------------------------
# UID utilities
# ---------------------------------------------------------------------------
def generation_model(uid: str) -> str:
    """Extract the TTS generation model identifier from a UID.

    Follows the LlamaPartialSpoof naming convention::

        <partition>-<model>_<speaker>_...

    Returns 'bonafide' when the UID represents genuine speech (i.e. the
    partition suffix is 'clean' or absent).

    Parameters
    ----------
    uid:
        Utterance identifier string.

    Raises
    ------
    TypeError
        If *uid* is not a string.
    ValueError
        If *uid* is blank.
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


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------
def duration_bin(duration_s: Union[int, float]) -> str:
    """Bin an audio duration into one of three named categories.

    Thresholds are derived from LlamaPartialSpoof duration statistics:
        - short: < 3s
        - medium: 3s - 8s
        - long: >8s

    Parameters
    ----------
    duration_s:
        Audio duration in seconds. Must be a non-negative int or float.

    Returns
    -------
    str
        One of 'short', 'medium', or 'long'.

    Raises
    ------
    TypeError
        If *duration_s* is not an int or float (booleans are rejected).
    ValueError
        If *duration_s* is negative.
    """
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise TypeError(f"duration_s must be int or float, got {type(duration_s).__name__!r}")
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")

    if duration_s < DURATION_SHORT_THRESHOLD:
        return "short"
    if duration_s < DURATION_LONG_THRESHOLD:
        return "medium"
    return "long"


def n_synth_segs_bin(n: int) -> str:
    """Bin a synthetic segment count into one of three named categories.

    Thresholds are derived from LlamaPartialSpoof segment statistics
    (median = 3, p75 = 5, p90 = 7, max = 20):
     - 0: Bonafide or fully synthetic (no splicing)
     - 1-3: Below-or-at-median partial synthetic
     - 3+: Above-median partial synthetic

    Parameters
    ----------
    n:
        Number of synthetic segments. Must be a non-negative int.

    Returns
    -------
    str
        One of '0', '1-3', or '3+'.

    Raises
    ------
    TypeError
        If *n* is not an int (booleans are rejected).
    ValueError
        If *n* is negative.
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
        "bonafide":         "bonafide",
        "synthetic":        "synthetic",
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
    """Decode WAV bytes into a mono float32 array.

    Parameters
    ----------
    data:
        Raw WAV file bytes.

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
    height: int = 52,
) -> None:
    """Render an interactive waveform player via st.components.v1.html.

    Encodes *audio* as a 16-bit WAV data URI and mounts a self-contained
    Web Audio API player with a seekable waveform canvas. No external
    dependencies or temporary files are used.

    Parameters
    ----------
    audio:
        Mono float32 waveform array.
    sr:
        Sample rate in Hz.
    label:
        Optional text shown top-left of the player.
    height:
        Canvas height in pixels (default 52).
    """
    import streamlit.components.v1 as components

    wav_bytes  = audio_to_bytes(audio, sr)
    b64        = base64.b64encode(wav_bytes).decode()
    data_uri   = f"data:audio/wav;base64,{b64}"

    label_html = (
        f'<span style="font-size:10px;color:#888;letter-spacing:0.06em;text-transform:uppercase;">'
        f'{label}</span>'
        if label else '<span></span>'
    )

    download_html = ""

    html = f"""
<div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;
            padding:10px 14px 8px;font-family:'Courier New',monospace;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:7px;min-height:18px;">
    {label_html}
    {download_html}
  </div>
  <canvas id="wc" height="{height}"
          style="width:100%;height:{height}px;display:block;cursor:pointer;border-radius:4px;">
  </canvas>
  <div style="display:flex;align-items:center;gap:10px;margin-top:6px;">
    <button id="pb"
            style="background:none;border:none;padding:0;cursor:pointer;
                   width:28px;height:28px;display:flex;align-items:center;
                   justify-content:center;border-radius:50%;transition:background 0.12s;"
            onmouseover="this.style.background='#f0f0f0';"
            onmouseout="this.style.background='none';">
    </button>
    <span id="tm" style="font-size:11px;color:#888;font-variant-numeric:tabular-nums;
                          letter-spacing:0.03em;">0:00 / 0:00</span>
    <span id="ld" style="font-size:10px;color:#bbb;margin-left:auto;">loading…</span>
  </div>
</div>

<script>
(function () {{
  const DATA_URI    = "{data_uri}";
  const ACCENT      = "#1a1a1a";
  const canvas      = document.getElementById("wc");
  const playBtn     = document.getElementById("pb");
  const timeEl      = document.getElementById("tm");
  const loadingEl   = document.getElementById("ld");

  let audioCtx      = null;
  let audioBuffer   = null;
  let sourceNode    = null;
  let startTime     = 0;
  let pauseOffset   = 0;
  let playing       = false;
  let rafId         = null;
  let hoverFrac     = null;

  // ── Icons ────────────────────────────────────────────────────────────────
  function setIcon(isPlaying) {{
    playBtn.innerHTML = isPlaying
      ? `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
           <rect x="2"   y="1.5" width="3.5" height="11" rx="1" fill="#333"/>
           <rect x="8.5" y="1.5" width="3.5" height="11" rx="1" fill="#333"/>
         </svg>`
      : `<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
           <path d="M3 1.5 L12 7 L3 12.5 Z" fill="#333"/>
         </svg>`;
  }}
  setIcon(false);

  // ── Time formatter ────────────────────────────────────────────────────────
  function fmt(s) {{
    const m = Math.floor(s / 60);
    return m + ":" + String(Math.floor(s % 60)).padStart(2, "0");
  }}

  function updateTime(currentS) {{
    const dur = audioBuffer ? audioBuffer.duration : 0;
    timeEl.textContent = fmt(currentS) + " / " + fmt(dur);
  }}

  // ── Waveform drawing ──────────────────────────────────────────────────────
  function currentFrac() {{
    if (!audioBuffer) return 0;
    if (!playing)     return pauseOffset / audioBuffer.duration;
    return Math.min(1, (audioCtx.currentTime - startTime) / audioBuffer.duration);
  }}

  function draw(overrideFrac) {{
    const dpr  = window.devicePixelRatio || 1;
    const W    = canvas.offsetWidth;
    const H    = {height};

    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {{
      canvas.width  = W * dpr;
      canvas.height = H * dpr;
    }}

    const ctx  = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = "#f8f8f8";
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, 4);
    ctx.fill();

    if (!audioBuffer) {{
      ctx.strokeStyle = "#ddd";
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      return;
    }}

    const frac    = overrideFrac !== undefined ? overrideFrac : currentFrac();
    const playedX = frac * W;
    const hoverX  = hoverFrac !== null ? hoverFrac * W : null;
    const data    = audioBuffer.getChannelData(0);
    const nBars   = Math.floor(W / 3);
    const step    = Math.max(1, Math.floor(data.length / nBars));
    const midY    = H / 2;

    for (let i = 0; i < nBars; i++) {{
      const x   = i * (W / nBars);
      let   sum = 0;
      const s0  = i * step;
      for (let j = s0; j < s0 + step && j < data.length; j++) {{
        sum += data[j] * data[j];
      }}
      const rms  = Math.sqrt(sum / step);
      const barH = Math.max(2, rms * H * 2.8);
      const bw   = Math.max(1, W / nBars - 1.2);

      if (hoverX !== null && x <= hoverX) {{
        ctx.fillStyle = ACCENT + "55";
      }} else if (x <= playedX) {{
        ctx.fillStyle = ACCENT;
      }} else {{
        ctx.fillStyle = "#d0d0d0";
      }}

      ctx.beginPath();
      ctx.roundRect(x, midY - barH / 2, bw, barH, 1.5);
      ctx.fill();
    }}

    // Playhead
    if (frac > 0) {{
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      ctx.moveTo(playedX, 0);
      ctx.lineTo(playedX, H);
      ctx.stroke();
    }}
  }}

  // ── Animation loop ────────────────────────────────────────────────────────
  function animLoop() {{
    if (!playing) return;
    const frac = currentFrac();
    draw(frac);
    updateTime(frac * audioBuffer.duration);
    rafId = requestAnimationFrame(animLoop);
  }}

  // ── Playback ──────────────────────────────────────────────────────────────
  function play() {{
    if (!audioBuffer || playing) return;
    if (audioCtx.state === "suspended") audioCtx.resume();
    sourceNode           = audioCtx.createBufferSource();
    sourceNode.buffer    = audioBuffer;
    sourceNode.connect(audioCtx.destination);
    sourceNode.onended   = () => {{ if (playing) onEnded(); }};
    sourceNode.start(0, pauseOffset);
    startTime = audioCtx.currentTime - pauseOffset;
    playing   = true;
    setIcon(true);
    animLoop();
  }}

  function pause() {{
    if (!playing) return;
    pauseOffset          = audioCtx.currentTime - startTime;
    sourceNode.onended   = null;
    sourceNode.stop();
    playing = false;
    cancelAnimationFrame(rafId);
    setIcon(false);
    draw();
  }}

  function onEnded() {{
    playing     = false;
    pauseOffset = 0;
    cancelAnimationFrame(rafId);
    setIcon(false);
    draw(0);
    updateTime(0);
  }}

  // ── Seek ──────────────────────────────────────────────────────────────────
  function seek(e) {{
    if (!audioBuffer) return;
    const rect    = canvas.getBoundingClientRect();
    const frac    = (e.clientX - rect.left) / rect.width;
    const seekTo  = frac * audioBuffer.duration;
    const wasPlay = playing;
    if (wasPlay) {{ sourceNode.onended = null; sourceNode.stop(); playing = false; }}
    pauseOffset = seekTo;
    draw(frac);
    updateTime(seekTo);
    if (wasPlay) play();
  }}

  // ── Event listeners ───────────────────────────────────────────────────────
  playBtn.addEventListener("click", () => {{ playing ? pause() : play(); }});
  canvas.addEventListener("click",      seek);
  canvas.addEventListener("mousemove",  e => {{
    const rect = canvas.getBoundingClientRect();
    hoverFrac  = (e.clientX - rect.left) / rect.width;
    draw();
  }});
  canvas.addEventListener("mouseleave", () => {{ hoverFrac = null; draw(); }});

  // ── Fetch and decode ──────────────────────────────────────────────────────
  (async function () {{
    try {{
      const resp   = await fetch(DATA_URI);
      const arrBuf = await resp.arrayBuffer();
      audioCtx     = new (window.AudioContext || window.webkitAudioContext)();
      audioBuffer  = await audioCtx.decodeAudioData(arrBuf);
      loadingEl.textContent = "";
      updateTime(0);
      draw();
    }} catch (err) {{
      loadingEl.textContent = "error";
      console.error("waveform_player: decode failed", err);
    }}
  }})();
}})();
</script>
"""
    components.html(html, height=height + 90)