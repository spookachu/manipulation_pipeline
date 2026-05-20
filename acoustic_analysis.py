"""Forensic acoustic analysis module.

Measures
----
  F0 / Pitch       
  Jitter (local)  
  Shimmer (local)  
  Spectrogram      

References
-----
Boersma, P. & Weenink, D. (2024). Praat. https://www.praat.org/
Jadoul et al. (2018). Introducing Parselmouth. Journal of Phonetics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Praat defaults 
_F0_MIN_HZ  = 75.0    # pitch floor  (Hz)
_F0_MAX_HZ  = 600.0   # pitch ceiling (Hz)  
_N_FORMANTS = 4       # F1-F4


# --------------------------------------
# Dataclasses
# --------------------------------------
@dataclass
class PitchStats:
    mean:     Optional[float] = None
    sd:       Optional[float] = None
    min:      Optional[float] = None
    max:      Optional[float] = None
    range:    Optional[float] = None
    n_voiced: int = 0


@dataclass
class FormantStats:
    mean: Optional[float] = None
    sd:   Optional[float] = None
    n:    int = 0


@dataclass
class VoiceQualityStats:
    jitter_pct:  Optional[float] = None
    shimmer_pct: Optional[float] = None
    nhr:         Optional[float] = None


@dataclass
class AcousticFeatures:
    duration_s:    float = 0.0
    sample_rate:   int   = 0
    pitch:         PitchStats              = field(default_factory=PitchStats)
    formants:      Dict[str, FormantStats] = field(default_factory=dict)
    voice_quality: VoiceQualityStats       = field(default_factory=VoiceQualityStats)
    error:         Optional[str]           = None


# --------------------------------------
# Core extraction via parselmouth
# --------------------------------------
def extract_features(audio: np.ndarray, sr: int) -> AcousticFeatures:
    """Extract Praat-equivalent acoustic features from a waveform."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return AcousticFeatures(
            error="parselmouth not installed. Run: pip install praat-parselmouth"
        )

    try:
        duration_s = len(audio) / sr
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)

        # Pitch - try filtered autocorrelation, fall back to ac.
        try:
            pitch_obj = call(snd, "To Pitch (filtered autocorrelation)",
                             0.0,        # time step (0 = auto)
                             _F0_MIN_HZ, # pitch floor
                             _F0_MAX_HZ, # pitch ceiling
                             15,         # max candidates
                             "no",       # very accurate
                             0.03,       # attenuation at top
                             0.09,       # silence threshold
                             0.50,       # voicing threshold
                             0.055,      # octave cost
                             0.35,       # octave jump cost
                             0.14)       # voiced/unvoiced cost
        except Exception:
            pitch_obj = call(snd, "To Pitch (ac)",
                             0.0,        # time step (0 = auto)
                             _F0_MIN_HZ, # pitch floor
                             15,         # max candidates
                             "no",       # very accurate
                             0.09,       # silence threshold
                             0.50,       # voicing threshold
                             0.055,      # octave cost
                             0.35,       # octave jump cost
                             0.14,       # voiced/unvoiced cost
                             _F0_MAX_HZ) # pitch ceiling

        f0_values = pitch_obj.selected_array["frequency"]
        voiced    = f0_values[f0_values > 0]

        if len(voiced) >= 2:
            pitch = PitchStats(
                mean     = float(np.mean(voiced)),
                sd       = float(np.std(voiced)),
                min      = float(np.min(voiced)),
                max      = float(np.max(voiced)),
                range    = float(np.max(voiced) - np.min(voiced)),
                n_voiced = len(voiced),
            )
        else:
            pitch = PitchStats(n_voiced=len(voiced))

        # Formants.
        formant_obj = call(snd, "To Formant (burg)",
                           0.0,   # time step (auto)
                           5,     # max formants
                           5500,  # max formant Hz
                           0.025, # window length (s)
                           50)    # pre-emphasis from (Hz)

        n_frames = call(formant_obj, "Get number of frames")
        formants: Dict[str, FormantStats] = {}

        for fi in range(1, _N_FORMANTS + 1):
            vals = []
            for frame in range(1, n_frames + 1):
                v = call(formant_obj, "Get value at time",
                         fi, call(formant_obj, "Get time from frame number", frame),
                         "Hertz", "Linear")
                if v == v and v is not None:
                    vals.append(v)
            if vals:
                arr = np.array(vals)
                formants[f"F{fi}"] = FormantStats(
                    mean=float(np.mean(arr)),
                    sd=float(np.std(arr)),
                    n=len(vals),
                )
            else:
                formants[f"F{fi}"] = FormantStats()

        # Jitter and shimmer via PointProcess (periodic, cc).
        point_process = call(snd, "To PointProcess (periodic, cc)",
                             _F0_MIN_HZ, _F0_MAX_HZ)

        jitter_pct = None
        shimmer_pct = None
        try:
            jitter_pct = float(call(
                point_process, "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3,
            )) * 100.0

            shimmer_pct = float(call(
                [snd, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6,
            )) * 100.0
        except Exception as exc:
            logger.warning("Jitter/shimmer failed: %s", exc)

        return AcousticFeatures(
            duration_s    = duration_s,
            sample_rate   = sr,
            pitch         = pitch,
            formants      = formants,
            voice_quality = VoiceQualityStats(
                jitter_pct  = jitter_pct,
                shimmer_pct = shimmer_pct,
            ),
        )

    except Exception as exc:
        logger.exception("extract_features failed")
        return AcousticFeatures(error=str(exc))


# --------------------------------------
# Spectrogram 
# --------------------------------------
def _praat_spectrogram(snd, window_len: float = 0.005, fmax_hz: float = 5000.0):
    """Return a Praat Spectrogram object and matching Intensity object.
    """
    spec      = snd.to_spectrogram(window_length=window_len, maximum_frequency=fmax_hz)
    intensity = snd.to_intensity()
    return spec, intensity


# --------------------------------------
# Streamlit rendering
# --------------------------------------
def render_acoustic_tab(
    target_audio: np.ndarray,
    target_sr:    int,
    target_name:  str,
) -> None:
    """Render the Acoustic Analysis tab in Stage 1."""
    import streamlit as st
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    col_target, col_comparison = st.columns(2)
    cmp_file = st.file_uploader(
        "Upload comparison audio",
        type=["wav", "flac", "mp3"],
        key="acoust_cmp_upload",
        help="Reference recording from the suspected speaker (exemplar).",
    )

    with col_target:
        st.markdown("**Target sample**")
        st.caption(f"`{target_name}` - {len(target_audio) / target_sr:.2f} s")
        if st.button("Extract features", key="acoust_extract_target"):
            with st.spinner("Computing features..."):
                feats = extract_features(target_audio, target_sr)
            st.session_state["acoust_target_feats"] = feats

        feats_t: Optional[AcousticFeatures] = st.session_state.get("acoust_target_feats")
        if feats_t is not None:
            _render_features(feats_t)

    with col_comparison:
        st.markdown("**Comparison sample** (suspected same source)")

        if cmp_file is not None:
            from utils import load_wav_bytes
            cmp_audio, cmp_sr = load_wav_bytes(cmp_file.read())
            st.session_state["acoust_cmp_audio"] = cmp_audio
            st.session_state["acoust_cmp_sr"]    = cmp_sr
            st.session_state["acoust_cmp_name"]  = cmp_file.name
            st.session_state.pop("acoust_cmp_feats", None)

        cmp_audio = st.session_state.get("acoust_cmp_audio")
        cmp_sr    = st.session_state.get("acoust_cmp_sr")
        cmp_name  = st.session_state.get("acoust_cmp_name", "")

        if cmp_audio is not None:
            st.caption(f"`{cmp_name}` - {len(cmp_audio) / cmp_sr:.2f} s")
            if st.button("Extract features", key="acoust_extract_cmp"):
                with st.spinner("Running Praat..."):
                    feats_c = extract_features(cmp_audio, cmp_sr)
                st.session_state["acoust_cmp_feats"] = feats_c
            feats_c: Optional[AcousticFeatures] = st.session_state.get("acoust_cmp_feats")
            if feats_c is not None:
                _render_features(feats_c)
        else:
            st.info("Upload a comparison audio file.")

    # Spectrogram comparison.
    feats_t = st.session_state.get("acoust_target_feats")
    feats_c = st.session_state.get("acoust_cmp_feats")

    if feats_t is not None or feats_c is not None:
        st.divider()
        st.markdown("#### Spectrogram comparison")
        spec_cols = st.columns(2)
        with spec_cols[0]:
            if feats_t and feats_t.error is None:
                _render_spectrogram(target_audio, target_sr, target_name)
        with spec_cols[1]:
            cmp_audio_s = st.session_state.get("acoust_cmp_audio")
            cmp_sr_s    = st.session_state.get("acoust_cmp_sr")
            if feats_c and feats_c.error is None and cmp_audio_s is not None:
                _render_spectrogram(cmp_audio_s, cmp_sr_s, cmp_name)


# --------------------------------------
# Internal rendering helpers
# --------------------------------------
def _render_features(feats: AcousticFeatures) -> None:
    import streamlit as st

    if feats.error:
        st.error(f"Feature extraction failed: {feats.error}")
        return

    p  = feats.pitch
    vq = feats.voice_quality

    st.markdown("**Fundamental frequency (F0 / Pitch)**")
    if p.mean is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean F0 (Hz)",  f"{p.mean:.1f}")
        c2.metric("SD F0 (Hz)",    f"{p.sd:.1f}")
        c3.metric("Range (Hz)",    f"{p.range:.1f}")
        c1.metric("Min F0 (Hz)",   f"{p.min:.1f}")
        c2.metric("Max F0 (Hz)",   f"{p.max:.1f}")
        c3.metric("Voiced frames", str(p.n_voiced))
    else:
        st.warning("Insufficient voiced frames for F0 estimation.")

    st.markdown("**Formants F1-F4**")
    if feats.formants:
        cols = st.columns(_N_FORMANTS)
        for i, (fname, fstat) in enumerate(feats.formants.items()):
            with cols[i]:
                st.markdown(f"*{fname}*")
                if fstat.mean is not None:
                    st.metric("Mean (Hz)", f"{fstat.mean:.0f}")
                    st.metric("SD (Hz)",   f"{fstat.sd:.0f}")
                else:
                    st.caption("-")

    st.markdown("**Voice quality**")
    vq_cols = st.columns(2)
    vq_cols[0].metric(
        "Jitter (local) %",
        f"{vq.jitter_pct:.3f}" if vq.jitter_pct is not None else "-",
        help="Normal speech: < 1.04% (Praat reference threshold).",
    )
    vq_cols[1].metric(
        "Shimmer (local) %",
        f"{vq.shimmer_pct:.3f}" if vq.shimmer_pct is not None else "-",
        help="Normal speech: < 3.81% (Praat reference threshold).",
    )


def _render_spectrogram(audio: np.ndarray, sr: int, label: str) -> None:
    """Render a wideband spectrogram with intensity overlay.

    Follows the official parselmouth GitHub example (Jadoul et al. 2018):
      - afmhot colormap, 70 dB dynamic range
      - intensity overlaid on a twin y-axis (white outline + black line)
    """
    import parselmouth
    import matplotlib.pyplot as plt
    import streamlit as st

    duration_s = len(audio) / sr
    if duration_s > 30:
        st.caption("Spectrogram limited to first 30 s for performance.")
        audio = audio[:30 * sr]

    with st.spinner("Computing Praat spectrogram..."):
        try:
            snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
            spec, intensity = _praat_spectrogram(snd, window_len=0.005, fmax_hz=5000.0)
        except Exception as exc:
            st.warning(f"Praat spectrogram failed: {exc}")
            return

    dynamic_range = 70
    X, Y   = spec.x_grid(), spec.y_grid()
    sg_db  = 10 * np.log10(spec.values)

    fig, ax = plt.subplots(figsize=(5, 3))

    # Spectrogram.
    ax.pcolormesh(X, Y, sg_db,
                  vmin=sg_db.max() - dynamic_range,
                  cmap="afmhot", shading="auto")
    ax.set_ylim([spec.ymin, spec.ymax])
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", fontsize=8)
    ax.tick_params(labelsize=7)

    # Intensity overlay on twin axis.
    ax2 = ax.twinx()
    ax2.plot(intensity.xs(), intensity.values.T, linewidth=2, color="w")
    ax2.plot(intensity.xs(), intensity.values.T, linewidth=1, color="k")
    ax2.set_ylabel("Intensity (dB)", fontsize=7, color="#444")
    ax2.tick_params(labelsize=6)
    ax2.set_ylim(0)
    ax2.grid(False)

    ax.set_xlim([snd.xmin, snd.xmax])
    ax.set_title(f"Wideband spectrogram - {label}", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
