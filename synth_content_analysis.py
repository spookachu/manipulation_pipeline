"""Synthetic Content Analysis module for the Voice Deepfake Forensic Pipeline.

Contains three components:

  1. Detection      — run CM model(s) and display scores
  2. Annotation     — structured perceptual indicators (-2 to +2 Likert scale)
                      adapted from Bayerl et al. (2023) "Towards Robust Speech
                      Deepfake Detection via Human-Inspired Reasoning"
  3. Explainability — CoughLIME loudness decomposition overlaid on the waveform
                      Adapted from Wullenweber A, Akman A, Schuller BW. CoughLIME:
                      Sonified Explanations for the Predictions of COVID-19 Cough
                      Classifiers. Annu Int Conf IEEE Eng Med Biol Soc.
                      2022 Jul;2022:1342-1345. doi: 10.1109/EMBC48229.2022.9871291.
                      Original implementation: https://github.com/glam-imperial/CoughLIME
"""
import logging
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perceptual annotation schema
# Indicators adapted from:
#   Dvirniak, A., Kushnir, E., Tarasov, D., Iudin, A., Kiriukhin, O., 
#   Pautov, M., ... & Rogov, O. Y. (2026). Towards Robust Speech Deepfake 
#   Detection via Human-Inspired Reasoning. arXiv preprint arXiv:2603.10725.
# ---------------------------------------------------------------------------
SCALE_OPTIONS = [
    "-2 Very sure not present",
    "-1 Probably not present",
    " 0 Unsure",
    "+1 Probably present",
    "+2 Very sure present",
]
SCALE_VALUES = [-2, -1, 0, 1, 2]
_VAL_TO_OPT  = dict(zip(SCALE_VALUES, SCALE_OPTIONS))
_OPT_TO_VAL  = dict(zip(SCALE_OPTIONS, SCALE_VALUES))

ANNOTATION_INDICATORS: List[Tuple[str, str]] = [
    ("fluency",        "Lack of fluency or coherence"),
    ("pauses_unnat",   "Unnatural pauses"),
    ("pauses_uniform", "Uniform pauses between words"),
    ("intonation",     "Unusual intonation patterns"),
    ("style_var",      "Insufficient variation in speaking style"),
    ("stress",         "Incorrect stress in common words"),
    ("mispron",        "Mispronunciation of common words"),
    ("accent",         "Unusual or inconsistent accent"),
    ("voice_qual",     "Atypical voice characteristics"),
    ("rate",           "Excessively fast speech"),
    ("abbrev",         "Incorrect reading of abbreviations"),
    ("typo",           "Verbalisation of typographical errors"),
    ("repetition",     "Word-by-word repetition in cases of tautology"),
]

# ---------------------------------------------------------------------------
# Perceptual annotation schema
# Indicators adapted from:
#    I Hear, Therefore I Trust: A Socio-Technical Investigation of Humans as 
#    Synthetic Speech Detectors Lelia Erscoi (University of Eastern Finland); 
#    Tomi Kinnunen (University of Eastern Finland)
# ---------------------------------------------------------------------------
EVAL_QUESTIONS: List[Tuple[str, str]] = [
    ("voice_mechanical",   "The voice sounds mechanical."),
    ("voice_expressive",   "The voice sounds expressive."),
    ("voice_intelligible", "The voice is easy to understand."),
    ("audio_clean",        "The audio sounds clean."),
    ("voice_calm",         "The voice sounds calm."),
    ("eval_confident",     "I am confident in my evaluation."),
]
EVAL_OPTIONS = [
    "Completely disagree",
    "Disagree",
    "Unsure",
    "Agree",
    "Completely agree",
]


# ---------------------------------------------------------------------------
# LoudnessDecomposition
# Directly adapted from glam-imperial/CoughLIME loudness_decomposition.py
# Original authors: Wullenweber A, Akman A, Schuller BW (2022)
# https://github.com/glam-imperial/CoughLIME
# ---------------------------------------------------------------------------
class LoudnessDecomposition:
    """Decomposes an audio array into components at minima of the power curve.

    Directly adapted from CoughLIME (glam-imperial/CoughLIME,
    loudness_decomposition.py). Original authors: Wullenweber, Akman &
    Schuller (2022). Adaptations: removed cough-specific defaults, added
    type hints, replaced plt.show() calls.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        min_length: int = 0,
        threshold: float = 75.0,
    ) -> None:
        self.audio              = audio.astype(np.float64)
        self.sr                 = sr
        self.threshold          = threshold
        self.min_length         = min_length
        self.decomposition_type = "loudness"

        (
            self.num_components,
            self.components,
            self.indices_components,
            self.loudness,
        ) = self._initialize_components()
        self.fudged_components = self._initialize_fudged_components()

    def _compute_power_db(
        self, win_len_sec: float = 0.1, power_ref: float = 1e-12
    ) -> np.ndarray:
        """Signal power in dB.

        Notebook: C1/C1S3_Dynamics.ipynb — audiolabs-erlangen.de/FMP.
        Adapted from CoughLIME loudness_decomposition.py.
        """
        win_len  = max(1, round(win_len_sec * self.sr))
        win      = np.ones(win_len) / win_len
        power_db = 10 * np.log10(
            np.convolve(self.audio ** 2, win, mode="same") / power_ref
        )
        power_db[~np.isfinite(power_db)] = 0.0
        return np.abs(power_db)

    def _get_loudness_indices(self) -> Tuple[List[int], np.ndarray]:
        """Indices of power minima below threshold — adapted from CoughLIME."""
        import itertools
        from scipy.signal import argrelextrema

        loudness         = self._compute_power_db()
        loudness_rounded = np.around(loudness, decimals=-1)
        li = [
            [k, next(g)[0]]
            for k, g in itertools.groupby(
                enumerate(loudness_rounded), key=lambda x: x[1]
            )
        ]
        loudness_no_dups = [item[0] for item in li]
        indices          = [item[1] for item in li]

        minima    = np.array(argrelextrema(np.array(loudness_no_dups), np.less)).flatten()
        to_delete = []
        for i, m in enumerate(minima):
            if loudness[int(indices[int(m)])] > self.threshold:
                to_delete.append(i)
            elif (
                i < len(minima) - 1
                and int(indices[int(m + 1)]) - int(indices[int(m)]) < self.min_length
            ):
                to_delete.append(i)
        minima = np.delete(minima, to_delete)
        return [int(indices[int(k)]) for k in minima], loudness

    def _initialize_components(self):
        """Split audio at loudness minima — adapted from CoughLIME."""
        indices_min, loudness = self._get_loudness_indices()
        components = []
        previous   = 0
        if len(indices_min) == 0:
            components.append(self.audio)
        else:
            for idx in indices_min:
                components.append(self.audio[previous:idx])
                previous = idx
            components.append(self.audio[previous:])
        return len(components), components, indices_min, loudness

    def _initialize_fudged_components(self) -> List[np.ndarray]:
        """Zero-filled replacement components — adapted from CoughLIME."""
        return [np.zeros_like(c) for c in self.components]

    def get_number_components(self) -> int:
        return self.num_components

    def get_components_mask(self, mask: np.ndarray) -> np.ndarray:
        """Reconstruct audio from a boolean mask — adapted from CoughLIME."""
        return np.concatenate([
            self.components[i] if mask[i] else self.fudged_components[i]
            for i in range(self.num_components)
        ])

    def return_components(self, indices: List[int], loudness: bool = False):
        """Return audio for given component indices, zeros elsewhere."""
        mask      = np.zeros(self.num_components, dtype=bool)
        mask[indices] = True
        audio_out = self.get_components_mask(mask)
        if loudness:
            borders  = [0] + self.indices_components + [len(self.audio)]
            loud_out = np.concatenate([
                self.loudness[borders[i]:borders[i + 1]]
                if mask[i] else self.fudged_components[i]
                for i in range(self.num_components)
            ])
            return audio_out, loud_out
        return audio_out

    def return_mask_boundaries(
        self, positive_indices: List[int], negative_indices: List[int]
    ) -> np.ndarray:
        """1-D boundary mask (+1 positive, -1 negative) — adapted from CoughLIME."""
        mask    = np.zeros(len(self.audio), dtype=np.int8)
        borders = [0] + self.indices_components + [len(self.audio)]
        for i in range(len(borders) - 1):
            if i in positive_indices:
                mask[borders[i] + 1: borders[i + 1] - 1] = 1
            elif i in negative_indices:
                mask[borders[i] + 1: borders[i + 1] - 1] = -1
        return mask


# ---------------------------------------------------------------------------
# LimeCoughExplainer
# Adapted from glam-imperial/CoughLIME lime_cough.py
# Original authors: Wullenweber A, Akman A, Schuller BW (2022)
# ---------------------------------------------------------------------------
class LimeCoughExplainer:
    """Model-agnostic LIME explainer for audio using loudness decomposition.

    Adapted from CoughLIME (glam-imperial/CoughLIME, lime_cough.py).
    Original authors: Wullenweber, Akman & Schuller (2022).
    """

    def __init__(
        self,
        kernel_width: float = 0.25,
        random_state: Optional[int] = None,
        feature_selection: str = "auto",
    ) -> None:
        from sklearn.utils import check_random_state
        from lime import lime_base

        self.random_state      = check_random_state(random_state)
        self.feature_selection = feature_selection

        def _kernel(d: np.ndarray, kw: float) -> np.ndarray:
            return np.sqrt(np.exp(-(d ** 2) / kw ** 2))

        kernel_fn = partial(_kernel, kw=float(kernel_width))
        self.base = lime_base.LimeBase(
            kernel_fn, verbose=False, random_state=self.random_state
        )

    def explain_instance(
        self,
        decomposition: LoudnessDecomposition,
        classifier_fn: Callable,
        label: int = 1,
        num_samples: int = 64,
        batch_size: int = 16,
        distance_metric: str = "cosine",
    ) -> dict:
        """Run LIME and return explanation dict.

        classifier_fn must accept (batch, n_samples) and return (batch, n_classes).
        Adapted from CoughLIME lime_cough.py LimeCoughExplainer.explain_instance.
        """
        import sklearn.metrics

        n_features        = decomposition.get_number_components()
        data, predictions = self._data_labels(
            decomposition, classifier_fn, num_samples, batch_size
        )
        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        intercept, local_exp, score, local_pred = (
            self.base.explain_instance_with_data(
                data, predictions, distances, label, n_features,
                feature_selection=self.feature_selection,
            )
        )
        return {
            "intercept":  intercept,
            "local_exp":  local_exp,
            "score":      score,
            "local_pred": local_pred,
        }

    def _data_labels(
        self,
        decomposition: LoudnessDecomposition,
        classifier_fn: Callable,
        num_samples: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate perturbed neighbourhood and predictions.

        Adapted verbatim from CoughLIME lime_cough.py data_labels().
        """
        n_features = decomposition.get_number_components()
        data       = self.random_state.randint(
            0, 2, num_samples * n_features
        ).reshape((num_samples, n_features))
        data[0, :] = 1  # first row = original (all components present)

        all_labels:   List             = []
        batch_audios: List[np.ndarray] = []

        for row in data:
            batch_audios.append(decomposition.get_components_mask(row.astype(bool)))
            if len(batch_audios) == batch_size:
                all_labels.extend(classifier_fn(np.array(batch_audios)))
                batch_audios = []
        if batch_audios:
            all_labels.extend(classifier_fn(np.array(batch_audios)))

        return data, np.array(all_labels)


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------
def render_synthetic_content_tab(
    audio:       np.ndarray,
    sr:          int,
    name:        str,
    insp_models: List[str],
    insp_thr:    float = 0.5,
) -> None:
    """Render the Synthetic Content Analysis tab in Stage 1.

    Parameters
    ----------
    audio        : Mono float32 waveform.
    sr           : Sample rate in Hz.
    name         : File stem, used for display labels.
    insp_models  : List of model keys selected in the sidebar.
    insp_thr     : Detection threshold (default 0.5).
    """
    import json
    import uuid
    import datetime as _dt
    import pandas as pd
    import streamlit as st
    from detection import MODEL_SPECS, MODEL_SPECS as _SPECS, predict
    from utils import audio_to_bytes, waveform_player

    WINDOW_S = 0.5
    duration = float(len(audio) / sr)
    seg_key  = "insp_segments"
    flag_key = "insp_flags"
    st.session_state.setdefault(seg_key,  [])
    st.session_state.setdefault(flag_key, [])

    ann_col, det_col = st.columns([2, 3])

    # ------------------------------------------------------------------ #
    # 1. Annotation column
    # ------------------------------------------------------------------ #
    with ann_col:
        ann = st.session_state.get("insp_det_annotation", {})

        st.markdown("#### Perceptual indicators")
        st.caption(
            "Indicators adapted from: Bayerl et al. (2023) *Towards Robust Speech "
            "Deepfake Detection via Human-Inspired Reasoning*, Interspeech 2023."
        )

        perc_rows = []
        for key, label in ANNOTATION_INDICATORS:
            perc_rows.append({
                "_key":      key,
                "Indicator": label,
                "Rating":    _VAL_TO_OPT.get(ann.get(key, 0), " 0  Unsure"),
            })
        perc_df = pd.DataFrame(perc_rows)
        edited_perc = st.data_editor(
            perc_df.drop(columns=["_key"]),
            hide_index=True,
            use_container_width=True,
            disabled=["Indicator"],
            column_config={
                "Indicator": st.column_config.TextColumn(width="medium"),
                "Rating": st.column_config.SelectboxColumn(
                    "Rating",
                    options=SCALE_OPTIONS,
                    width="small",
                    required=True,
                ),
            },
            key="ann_likert_table",
        )
        for i, row in enumerate(perc_rows):
            ann[row["_key"]] = _OPT_TO_VAL.get(edited_perc.iloc[i]["Rating"], 0)

        st.divider()
        st.markdown("#### Subjective evaluation")

        eval_rows = []
        for ekey, elabel in EVAL_QUESTIONS:
            eval_rows.append({
                "_key":     ekey,
                "Question": elabel,
                "Rating":   ann.get(ekey, "Unsure"),
            })
        eval_df = pd.DataFrame(eval_rows)
        edited_eval = st.data_editor(
            eval_df.drop(columns=["_key"]),
            hide_index=True,
            use_container_width=True,
            disabled=["Question"],
            column_config={
                "Question": st.column_config.TextColumn(width="medium"),
                "Rating": st.column_config.SelectboxColumn(
                    "Rating",
                    options=EVAL_OPTIONS,
                    width="medium",
                    required=True,
                ),
            },
            key="ann_eval_table",
        )
        for i, row in enumerate(eval_rows):
            ann[row["_key"]] = edited_eval.iloc[i]["Rating"]

        st.divider()
        ann["notes"] = st.text_area(
            "Free-text observations",
            value=ann.get("notes", ""),
            height=80,
            placeholder=("Additional observations not captured above."),
            key="ann_notes",
        )

        if st.button("Save annotation", type="primary", key="ann_save"):
            st.session_state["insp_det_annotation"] = dict(ann)
            st.session_state["insp_det_annotation"]["segments"] = list(
                st.session_state[seg_key]
            )
            st.session_state["insp_det_annotation"]["flags"] = list(st.session_state[flag_key])
            st.success("Annotation saved.")

    # ------------------------------------------------------------------ #
    # 2. Detection column
    # ------------------------------------------------------------------ #
    with det_col:
        
        st.markdown("#### Mark suspicious timestamps")
        waveform_player(
            audio, sr,
            label=name,
            flags=st.session_state[flag_key],
        )

        # ---------------------------------------------------------- #
        # Flag marking
        # ---------------------------------------------------------- #
        flag_slider = st.slider(
            "Flag timestamp",
            0.0, duration,
            value=0.0,
            step=0.01,
            key="ann_flag_slider",
        )
        if st.button("Add flag", key="ann_add_flag"):
            st.session_state[flag_key].append({
                "id":        str(uuid.uuid4()),
                "time":      flag_slider,
                "timestamp": _dt.datetime.now().isoformat(),
            })
            st.rerun()

        # Delete lists
        if st.session_state[flag_key]:
            flags_to_delete = []
            st.markdown("**Flags**")
            cols_f = st.columns(len(st.session_state[flag_key]))
            for col_f, flag in zip(cols_f, st.session_state[flag_key]):
                with col_f:
                    st.markdown(
                        f'<span style="background:#ffd166;color:#333;border-radius:4px;'
                        f'padding:2px 6px;font-size:0.78em;">{flag["time"]:.2f}s</span>',
                        unsafe_allow_html=True,
                    )
                    if st.button("X", key=f"del_flag_{flag['id']}"):
                        flags_to_delete.append(flag["id"])
            if flags_to_delete:
                st.session_state[flag_key] = [
                    f for f in st.session_state[flag_key]
                    if f["id"] not in flags_to_delete
                ]
                st.rerun()

        # ---------------------------------------------------------- #
        # Detection
        # ---------------------------------------------------------- #
        st.divider()
        st.caption(
            "Run one or more countermeasure (CM) models on the target audio. ")
        if st.button("Run detection on target", type="primary", key="det_run"):
            if not insp_models:
                st.warning("Select at least one model in the sidebar.")
            else:
                with st.spinner("Running detection…"):
                    results = predict(
                        audio, sr,
                        model_keys=insp_models,
                        threshold=insp_thr,
                        max_duration_s=len(audio) / sr + 1,
                    )
                st.session_state.insp_det = {r.model_key: r for r in results}

        for mk, r in st.session_state.get("insp_det", {}).items():
            spec  = next((s for s in MODEL_SPECS if s.key == mk), None)
            dname = spec.display_name if spec else mk
            if r.error:
                st.error(f"{dname}: {r.error}")
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model",      dname)
            c2.metric("Synth prob", round(r.spoof_prob, 4))
            c3.metric("CM score",   round(r.cm_score,   4))
            pred_color = "#ff4f4f" if r.prediction == "synthetic" else "#3ddc84"
            c4.markdown(
                f"**Prediction:** <span style='color:{pred_color};"
                f"font-weight:bold;'>{r.prediction.upper()}</span>",
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------- #
        # Explainability — CoughLIME
        # ---------------------------------------------------------- #
        st.divider()

        model_options = list(st.session_state.get("insp_det", {}).keys())
        if not model_options:
            st.info("Run detection above to enable explainability.")
        else:
            exp_model_key = st.selectbox(
                "Model to explain",
                options=model_options,
                format_func=lambda k: next(
                    (s.display_name for s in _SPECS if s.key == k), k
                ),
                key="exp_model_select",
            )

            col_thr, col_ns, col_bs = st.columns(3)
            exp_threshold = col_thr.slider(
                "Loudness threshold (dB)",
                min_value=50, max_value=160, value=120, step=5,
                key="exp_threshold",
                help=("Power minima below this threshold define component boundaries. "),
            )
            exp_num_samples = col_ns.slider(
                "LIME samples",
                min_value=32, max_value=256, value=64, step=32,
                key="exp_num_samples",
                help="More samples -> more stable explanation but slower.",
            )
            exp_num_features = col_bs.slider(
                "Components to highlight",
                min_value=1, max_value=10, value=5, step=1,
                key="exp_num_features",
            )

            if st.button("Run LIME", type="primary", key="exp_run"):
                det_r = st.session_state.insp_det.get(exp_model_key)
                if det_r is None or det_r.error:
                    st.error("Detection result unavailable for this model.")
                else:
                    with st.spinner("Decomposing audio…"):
                        decomp = LoudnessDecomposition(
                            audio, sr, threshold=float(exp_threshold)
                        )

                    power = decomp.loudness
                    st.caption(
                        f"Loudness decomposition: {decomp.num_components} component(s). "
                        f"Power range: {power.min():.0f}-{power.max():.0f} dB."
                    )

                    if decomp.num_components < 2:
                        st.warning(
                            "Too few components — try raising the loudness threshold."
                        )
                    else:
                        from detection import predict as _predict

                        auto_samples = min(
                            exp_num_samples,
                            max(32, 6 * decomp.num_components),
                        )
                        if auto_samples < exp_num_samples:
                            st.caption(
                                f"Sample count capped at {auto_samples} "
                                f"(6X the {decomp.num_components} components found)."
                            )

                        def _classifier_fn(audio_batch: np.ndarray) -> np.ndarray:
                            out = []
                            for wav in audio_batch:
                                wav_f32 = wav.astype(np.float32)
                                res     = _predict(
                                    wav_f32, sr,
                                    model_keys=[exp_model_key],
                                    threshold=0.5,
                                    max_duration_s=len(wav_f32) / sr + 1,
                                )
                                p_syn = (
                                    res[0].spoof_prob
                                    if res and not res[0].error
                                    else 0.5
                                )
                                out.append([1 - p_syn, p_syn])
                            return np.array(out)

                        with st.spinner(
                            f"Running LIME with {auto_samples} samples "
                            f"on {decomp.num_components} components…"
                        ):
                            try:
                                explainer = LimeCoughExplainer(random_state=42)
                                result    = explainer.explain_instance(
                                    decomp,
                                    _classifier_fn,
                                    label=1,
                                    num_samples=auto_samples,
                                )
                                st.session_state["insp_lime_result"] = {
                                    "result":    result,
                                    "decomp":    decomp,
                                    "model_key": exp_model_key,
                                }
                            except Exception as exc:
                                st.error(f"LIME failed: {exc}")
                                logger.exception("LIME explainability error")

            lime_cache = st.session_state.get("insp_lime_result")
            if lime_cache:
                if lime_cache.get("model_key") != exp_model_key:
                    st.caption(
                        f"Showing explanation for **{lime_cache.get('model_key')}** ")

                score = lime_cache["result"].get("score")
                if score is not None:
                    st.metric(
                        "LIME local fidelity (R²)",
                        f"{score:.3f}",
                        delta="reliable" if score >= 0.5 else "low",
                        delta_color="normal" if score >= 0.5 else "inverse",
                        help=(
                            "R² of the local linear model on the perturbed neighbourhood. "
                            "Values below 0.5 indicate the explanation may be unreliable."
                        ),
                    )

                _render_lime_result(
                    lime_cache["result"],
                    lime_cache["decomp"],
                    audio, sr,
                    exp_num_features,
                )


# ---------------------------------------------------------------------------
# LIME result rendering
# ---------------------------------------------------------------------------
def _render_lime_result(
    result:       dict,
    decomp:       LoudnessDecomposition,
    audio:        np.ndarray,
    sr:           int,
    num_features: int,
) -> None:
    """Render the LIME explanation as a waveform heatmap and playable segments."""
    import pandas as pd
    import streamlit as st
    from matplotlib.patches import Patch
    from utils import audio_to_bytes

    local_exp = result["local_exp"]
    if not local_exp:
        st.warning("LIME returned an empty explanation.")
        return

    sorted_exp  = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)
    top_exp     = sorted_exp[:num_features]
    borders     = [0] + decomp.indices_components + [len(audio)]
    weight_map  = {idx: w for idx, w in top_exp}
    all_weights = [abs(w) for _, w in top_exp]
    max_w       = max(all_weights) if all_weights else 1.0
    if not np.isfinite(max_w) or max_w == 0.0:
        max_w = 1.0

    times = np.linspace(0, len(audio) / sr, num=len(audio))
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.plot(times, audio, color="#1a1a1a", linewidth=0.5, alpha=0.7, zorder=2)

    for i in range(decomp.num_components):
        t_start = borders[i]     / sr
        t_end   = borders[i + 1] / sr
        if i in weight_map:
            w     = weight_map[i]
            w_abs = abs(w) if np.isfinite(w) else 0.0
            alpha = float(np.clip(0.15 + 0.55 * w_abs / max_w, 0.0, 1.0))
            color = "#f21010" if w > 0 else "#50dc3d"
            ax.axvspan(t_start, t_end, facecolor=color, alpha=alpha, zorder=1)
            ax.text(
                (t_start + t_end) / 2,
                ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.9,
                str(i), fontsize=6, ha="center", va="top", color=color, zorder=3,
            )
        else:
            ax.axvspan(t_start, t_end, facecolor="#888888", alpha=0.05, zorder=1)

    ax.set_xlim(0, len(audio) / sr)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.legend(
        handles=[
            Patch(facecolor="#f21010", alpha=0.6, label="-> synthetic"),
            Patch(facecolor="#50dc3d", alpha=0.6, label="-> bonafide"),
            Patch(facecolor="#888888", alpha=0.2, label="not in top features"),
        ],
        fontsize=6, loc="upper right",
    )
    ax.set_title("CoughLIME: loudness component importance", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Weight table
    rows = []
    for rank, (idx, w) in enumerate(sorted_exp[:num_features], start=1):
        t_s = borders[idx]     / sr
        t_e = borders[idx + 1] / sr
        rows.append({
            "Rank":      rank,
            "Component": idx,
            "Start (s)": f"{t_s:.3f}",
            "End (s)":   f"{t_e:.3f}",
            "Weight":    round(w, 4),
            "Direction": "-> synthetic" if w > 0 else "-> bonafide",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Listenable segments
    st.markdown("**Listen to important components**")
    st.caption(
        "Each segment is isolated with surrounding components zeroed. "
        "Red = pushes toward synthetic, green = pushes toward bonafide."
    )
    cols = st.columns(min(num_features, 3))
    for rank, (idx, w) in enumerate(sorted_exp[:num_features]):
        col       = cols[rank % len(cols)]
        seg       = decomp.return_components([idx])
        color     = "#f21010" if w > 0 else "#50dc3d"
        direction = "-> synthetic" if w > 0 else "-> bonafide"
        t_s       = borders[idx]     / sr
        t_e       = borders[idx + 1] / sr
        col.markdown(
            f'<span style="background:{color};color:#fff;border-radius:3px;'
            f'padding:2px 6px;font-size:0.8em;">#{rank + 1} comp {idx}</span> '
            f'<span style="font-size:0.75em;color:#888;">'
            f'{t_s:.2f}-{t_e:.2f}s w={w:+.3f} {direction}</span>',
            unsafe_allow_html=True,
        )
        col.audio(audio_to_bytes(seg.astype(np.float32), sr), format="audio/wav")