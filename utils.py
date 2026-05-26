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