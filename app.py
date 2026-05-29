"""Voice Deepfake Forensic Pipeline — Streamlit application.

Stages
------
Stage 0 - Database   : Dataset selection, label loading, BM score caching,
                       background model visualisation.
Stage 1 - Inspection : Target audio upload, acoustic/auditory analysis,
                       deepfake detection.
Stage 2 - Analysis   : LR framework, M-BM scoring, hypothesis shift assessment.
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

matplotlib.use("Agg")

from visualizations import (
    LABEL_COLORS,
    cat_color,
    pipeline_badge,
    kde_plot,
    bm_kde_sidebyside,
    bm_kde_overlay,
    per_dataset_kde,
    combo_shift_kde,
    neighbourhood_composition,
)

from background_model import (
    compute_likelihood_ratio,
    verbal_lr_strength
)
from config import (
    apply_path_overrides,
    _clean_path,
    load_path_overrides_from_session,
    _PLACEHOLDER_PREFIX,
)
from datasets import (
    DATASET_REGISTRY,
    get_dataset,
)
from detection import (
    CACHE_BATCH_SIZE,
    MODEL_SPECS,
    cache_status,
    load_cache,
    pipeline_cache_key,
    predict,
    score_and_cache,
    compute_eer,
    compute_cllr,
)
from manipulations import (
    CATEGORY_LABELS,
    MANIPULATION_CATEGORIES,
    MANIPULATIONS,
    MANIPULATIONS_BY_CATEGORY,
    ManipulationStep,
    PARAM_PRESETS,
    PARAM_SCHEMA,
    PRESET_PIPELINES,
    REGION_MODE_LABELS,
    REGION_MODES,
    apply_pipeline,
)
from session import init_state
from utils import (
    audio_to_bytes,
    load_wav_bytes,
    waveform_player,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Voice Deepfake Forensics", layout="wide")
init_state()
load_path_overrides_from_session()

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

WINDOW_S = 0.5  # minimum audio window for detection confidence warning

MODEL_DISPLAY = {s.key: s.display_name for s in MODEL_SPECS}
LABEL_CLASSES = ["bonafide", "synthetic", "partial synthetic"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def get_wav_index(ds, selected_key: str) -> dict:
    key = f"wav_index_{selected_key}"
    if key not in st.session_state:
        with st.spinner("Indexing audio files…"):
            st.session_state[key] = {
                p.stem: p for p in ds.audio_path().rglob(f"*.{ds.audio_ext}")
            }
    return st.session_state[key]


def mbm_cache_path(dataset_key, model_key, cache_key) -> Path:
    return CACHE_DIR / f"{dataset_key}_{model_key}_mbm_{cache_key}.json"


def mbm_partial_path(dataset_key, model_key, cache_key) -> Path:
    return CACHE_DIR / f"{dataset_key}_{model_key}_mbm_{cache_key}.partial.json"


def load_mbm_cache_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {uid: v for uid, v in data.get("scores", {}).items() if isinstance(v, dict)}
    except Exception:
        return {}


def agg_scores(db_data: dict, model_key: str) -> dict:
    """Aggregate per-model scores into {uid: {cm, label}} using canonical labels."""
    return {
        uid: {"cm": e["cm"], "label": e["gt"]}
        for uid, entries in db_data.items()
        for e in entries
        if e.get("model") == model_key
    }


def build_plot_and_lr_dicts(agg_data: dict) -> tuple:
    """Split aggregated scores into KDE plot dict and LR dict."""
    plot_dict = {"bonafide": [], "synthetic": []}
    lr_dict   = {"bonafide": [], "synthetic": [], "partial synthetic": []}
    for d in agg_data.values():
        cm, lbl = d["cm"], d["label"]
        lr_dict.setdefault(lbl, []).append(cm)
        if lbl == "bonafide":
            plot_dict["bonafide"].append(cm)
        else:
            plot_dict["synthetic"].append(cm)
    return plot_dict, lr_dict

def compute_lr_scores(nat_cm: list, syn_cm: list) -> tuple:
    """Compute per-utterance LR values from CM scores using KDE-fitted densities.

    Returns (lr_values, labels) ready for compute_cllr, or (None, None)
    if either distribution has insufficient samples.
    """
    from background_model import _fit_kde, _log_pdf_kde
    kde_bonaf = _fit_kde(nat_cm)
    kde_synth = _fit_kde(syn_cm)
    if kde_bonaf is None or kde_synth is None:
        return None, None

    all_cms = nat_cm + syn_cm
    labels  = [0] * len(nat_cm) + [1] * len(syn_cm)
    lrs     = []
    for cm in all_cms:
        log_hd = _log_pdf_kde(kde_bonaf, cm)
        log_hp = _log_pdf_kde(kde_synth, cm)
        lr     = float(np.exp(np.clip(log_hp - log_hd, -500, 500)))
        lrs.append(max(lr, 1e-300)) 
    return np.array(lrs), np.array(labels, dtype=np.int32)

def pct_syn_bucket(pct: float) -> str:
    if pct == 0.0:  return "0% — bonafide"
    if pct < 50.0:  return "1-49%"
    if pct < 100.0: return "50-99%"
    return "100% — fully synthetic"


def fmt_pct(v, decimals=1) -> str:
    return "—" if v is None else f"{v * 100:.{decimals}f}%"


def colour_delta(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return f"background-color: {'#c8f7c5' if val < 0 else '#f7c5c5'}"


def neighbourhood(agg_data: dict, target: float, hw: float) -> dict:
    lo, hi = target - hw, target + hw
    nat_in, syn_in, nat_all, syn_all = [], [], [], []
    for d in agg_data.values():
        cm, lbl = d["cm"], d["label"]
        if lbl == "bonafide":
            nat_all.append(cm)
            if lo <= cm <= hi:
                nat_in.append(cm)
        else:
            syn_all.append(cm)
            if lo <= cm <= hi:
                syn_in.append(cm)
    n_total_in = len(nat_in) + len(syn_in)
    return {
        "n_nat_in":   len(nat_in),
        "n_syn_in":   len(syn_in),
        "n_total_in": n_total_in,
        "pct_syn":    (len(syn_in) / n_total_in * 100) if n_total_in > 0 else float("nan"),
        "nat_all": nat_all, "syn_all": syn_all,
        "nat_in":  nat_in,  "syn_in":  syn_in,
        "lo": lo, "hi": hi,
    }


def stat_parity_table(bk_rows, group_key_fn, group_label, order=None, show_class_counts=False):
    buckets = defaultdict(list)
    for r in bk_rows:
        buckets[group_key_fn(r)].append(r)
    keys = [k for k in (order or sorted(buckets.keys())) if k in buckets]
    all_preds     = [r["pred"] for r in bk_rows]
    p_syn_global  = float(np.mean(all_preds)) if all_preds else None
    rows_out = []
    for k in keys:
        rows_k = buckets[k]
        if len(rows_k) < 2:
            continue
        preds  = np.array([r["pred"] for r in rows_k], dtype=np.int32)
        gts    = np.array([r["gt"]   for r in rows_k], dtype=np.int32)
        acc    = float(np.mean(preds == gts))
        p_syn  = float(np.mean(preds == 1))
        sp_gap = (p_syn - p_syn_global) if p_syn_global is not None else None
        row    = {group_label: k, "N": len(rows_k)}
        if show_class_counts:
            row["N bonafide"]  = int(np.sum(gts == 0))
            row["N synthetic"] = int(np.sum(gts == 1))
        row["Accuracy"]        = fmt_pct(acc)
        row["P(pred=synth)"]   = fmt_pct(p_syn)
        row["SP gap"]          = f"{sp_gap * 100:+.1f}%" if sp_gap is not None else "—"
        rows_out.append(row)
    if rows_out:
        st.dataframe(pd.DataFrame(rows_out), hide_index=True, use_container_width=True)
        if p_syn_global is not None:
            st.caption(
                f"**Statistical parity**: P(pred=synthetic) should be equal across "
                f"groups for a fair detector. Global rate = {p_syn_global * 100:.1f}%. "
                f"SP gap = slice rate - global rate (+: over-flagged, -: under-flagged). "
                f"Threshold: cm ≥ 0."
            )
    else:
        st.caption("Not enough data.")


def run_mbm_scoring(pipeline_steps, cache_key, model_keys_to_run, selected_key, ds, utterances, an_models):
    """Score all BM UIDs through pipeline_steps and cache results."""
    wav_index = get_wav_index(ds, selected_key)

    bm_uids = []
    for spec in MODEL_SPECS:
        bm_path = CACHE_DIR / f"{selected_key}_{spec.key}.json"
        if bm_path.exists():
            try:
                bm_uids = sorted(json.loads(bm_path.read_text()).get("scores", {}).keys())
                if bm_uids:
                    break
            except Exception:
                pass
    if not bm_uids:
        bm_uids = sorted(st.session_state.db_cm_scores.keys())

    for model_key in model_keys_to_run:
        spec         = next(s for s in MODEL_SPECS if s.key == model_key)
        cache_path   = mbm_cache_path(selected_key, model_key, cache_key)
        partial_path = mbm_partial_path(selected_key, model_key, cache_key)

        if load_mbm_cache_file(cache_path):
            st.caption(f"{spec.display_name} [{cache_key}] — already cached, skipping.")
            continue

        existing = load_mbm_cache_file(partial_path)
        to_score = [uid for uid in bm_uids if uid in wav_index and uid not in existing]

        if not to_score:
            st.caption(f"{spec.display_name} [{cache_key}] — nothing new to score.")
            continue

        st.caption(
            f"{spec.display_name} [{cache_key}] — "
            f"scoring {len(to_score)} files ({len(existing)} already cached)."
        )
        prog = st.progress(0)
        stat = st.empty()

        scores_out = dict(existing)
        parameters = {
            "scoring":        "utterance_level",
            "max_duration_s": 30.0,
            "threshold":      0.5,
            "cm_convention":  "logit_synthetic - logit_bonafide",
            "cache_key":      cache_key,
            "pipeline_steps": [
                {"name": s.name, "params": s.params,
                 "region_mode": s.region_mode, "boundary_pad_s": s.boundary_pad_s}
                for s in pipeline_steps
            ],
        }

        for i, uid in enumerate(to_score):
            try:
                import io as _io
                with open(str(wav_index[uid]), "rb") as _fh:
                    _buf = _io.BytesIO(_fh.read())
                raw_audio, raw_sr = sf.read(_buf, dtype="float32", always_2d=False)
                if raw_audio.ndim > 1:
                    raw_audio = raw_audio.mean(axis=1)
            except Exception as exc:
                logger.warning("MBM: cannot read %s: %s", uid, exc)
                prog.progress((i + 1) / len(to_score))
                continue
            
            utt     = utterances.get(uid)
            gt      = utt.effective_label if utt else "unknown"
            uid_idx = bm_uids.index(uid)

            try:
                manip_audio = apply_pipeline(raw_audio, raw_sr, pipeline_steps, utterance=utt, uid_index=uid_idx)
            except Exception as exc:
                logger.warning("MBM pipeline error %s: %s", uid, exc)
                prog.progress((i + 1) / len(to_score))
                continue

            try:
                results = predict(manip_audio, raw_sr, model_keys=[model_key], threshold=0.5, max_duration_s=30.0)
                for r in results:
                    if not r.error:
                        scores_out[uid] = {"cm": r.cm_score, "gt": gt, "split": "train"}
            except Exception as exc:
                logger.warning("MBM inference error %s: %s", uid, exc)

            if (i + 1) % CACHE_BATCH_SIZE == 0 or (i + 1) == len(to_score):
                partial_path.write_text(json.dumps({
                    "dataset_key":        selected_key,
                    "model_key":          model_key,
                    "model_display_name": spec.display_name,
                    "scored_at":          datetime.now().isoformat(timespec="seconds"),
                    "parameters":         parameters,
                    "scores":             scores_out,
                }, indent=2))

            prog.progress((i + 1) / len(to_score))
            stat.caption(f"{spec.display_name} [{cache_key}] — {i + 1}/{len(to_score)} scored")

        if partial_path.exists():
            partial_path.rename(cache_path)
        stat.success(f"{spec.display_name} [{cache_key}] — {len(scores_out)} utterances cached.")


# ---------------------------------------------------------------------------
# Shared UI components
# ---------------------------------------------------------------------------
def pipeline_sidebar(stage_key: str, pipeline_state_key: str, title: str = "Manipulation pipeline") -> None:
    st.markdown(f"### {title}")
    chosen_cat = st.selectbox(
        "Category", MANIPULATION_CATEGORIES,
        format_func=lambda c: CATEGORY_LABELS[c],
        key=f"{stage_key}_cat",
    )
    names  = MANIPULATIONS_BY_CATEGORY.get(chosen_cat, [])
    chosen = st.selectbox("Manipulation", ["- select -"] + names, key=f"{stage_key}_manip")

    region_mode  = "whole"
    manual_start = 0.0
    manual_end   = 0.0
    boundary_pad = 0.15
    collected_params: dict = {}

    if chosen and chosen != "- select -":
        st.caption(f"_{MANIPULATIONS[chosen]['description']}_")

        for p in PARAM_SCHEMA.get(chosen, []):
            pk = f"{stage_key}_p_{chosen}_{p['key']}"
            pt = p["type"]
            if pt in ("float", "int"):
                cast = float if pt == "float" else int
                collected_params[p["key"]] = st.slider(
                    p["label"], cast(p["min"]), cast(p["max"]),
                    cast(p["default"]), cast(p["step"]),
                    help=p.get("help", ""), key=pk,
                )
            elif pt == "select":
                opts = p["options"]
                collected_params[p["key"]] = st.selectbox(
                    p["label"], options=opts,
                    index=opts.index(p["default"]) if p["default"] in opts else 0,
                    help=p.get("help", ""), key=pk,
                )
            elif pt == "seed":
                collected_params[p["key"]] = int(st.number_input(
                    p["label"], 0, 2**31 - 1, int(p["default"]), 1,
                    help=p.get("help", ""), key=pk,
                ))

        region_mode = st.selectbox(
            "Region", REGION_MODES,
            format_func=lambda m: REGION_MODE_LABELS.get(m, m),
            key=f"{stage_key}_rmode",
        )
        if region_mode == "manual":
            manual_start = st.number_input("Start (s)", 0.0, value=0.0, step=0.1, key=f"{stage_key}_ms")
            manual_end   = st.number_input("End (s)",   0.0, value=1.0, step=0.1, key=f"{stage_key}_me")
        elif region_mode == "splice_boundaries":
            boundary_pad = st.slider("Boundary pad (s)", 0.05, 1.0, 0.15, step=0.05, key=f"{stage_key}_bp")

        if st.button("Add step", width="stretch", key=f"{stage_key}_add"):
            region   = (manual_start, manual_end) if region_mode == "manual" else None
            presets  = PARAM_PRESETS.get(chosen, {})
            resolved = {}
            for k, v in collected_params.items():
                if k in presets or v in presets:
                    resolved.update(presets.get(v, {v: v}))
                else:
                    resolved[k] = v
            if not resolved:
                resolved = dict(collected_params)
            st.session_state[pipeline_state_key].append(
                ManipulationStep(
                    name=chosen, params=resolved, region=region,
                    region_mode=region_mode, boundary_pad_s=boundary_pad,
                )
            )
            st.rerun()

    st.divider()
    pipeline = st.session_state[pipeline_state_key]
    if pipeline:
        for i, step in enumerate(pipeline):
            c1, c2 = st.columns([5, 1])
            mode_s   = "" if step.region_mode == "whole" else f" [{step.region_mode}]"
            params_s = (" · " + ", ".join(f"{k}={v}" for k, v in step.params.items())) if step.params else ""
            c1.markdown(
                f'<span style="background:{cat_color(step.name)};color:#fff;'
                f'border-radius:3px;padding:1px 4px;font-size:0.76em;">{i + 1}</span> '
                f'**{step.name}**{mode_s}'
                f'<br><span style="font-size:0.72em;color:#aaa;">{params_s}</span>',
                unsafe_allow_html=True,
            )
            if c2.button("✕", key=f"{stage_key}_del_{i}"):
                pipeline.pop(i)
                st.rerun()
        if st.button("Clear pipeline", width="stretch", key=f"{stage_key}_clear"):
            st.session_state[pipeline_state_key] = []
            st.rerun()
    else:
        st.caption("No steps added.")


def render_cache_status_table(dataset_key: str, model_filter: list = None, expected_n: int = None) -> tuple:
    rows = cache_status(dataset_key, CACHE_DIR, expected_n=expected_n)
    if model_filter is not None:
        rows = [r for r in rows if r["model_key"] in model_filter]
    display_rows = [
        {
            "Model":     r["display_name"],
            "Status":    {"complete": "✅", "partial": "⏳ Partial"}.get(r["status"], "❌"),
            "Files":     r["n_files"] if r["n_files"] else "—",
            "Scored at": r["scored_at"],
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True)
    missing_keys = [r["model_key"] for r in rows if not r["is_complete"]]
    return rows, missing_keys


def stratified_sample(scoreable: list, utterances: dict, pct: int, seed: int,
                      strat_spec=None) -> list:
    """Return a stratified pct% sample of scoreable UIDs.

    Parameters
    ----------
    scoreable:
        List of UIDs to sample from.
    utterances:
        Dict mapping UID -> Utterance.
    pct:
        Target percentage of *scoreable* to return (1-100).
    seed:
        Random seed for reproducibility.
    strat_spec:
        A :class:`datasets.StratificationSpec` instance that defines the
        stratum key for this dataset. 
    """
    flat_strata: dict = {}
    for uid in scoreable:
        utt = utterances.get(uid)
        if utt is None:
            continue
        key = strat_spec.stratum_key(utt) if strat_spec is not None else (utt.effective_label,)
        flat_strata.setdefault(key, []).append(uid)

    if not flat_strata:
        return []

    n_target    = round(len(scoreable) * pct / 100)
    sorted_keys = sorted(flat_strata.keys())
    rng         = random.Random(seed)

    shuffled = {k: sorted(flat_strata[k]) for k in sorted_keys}
    for g in shuffled.values():
        rng.shuffle(g)

    lo, hi = 0, max(len(v) for v in flat_strata.values())
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(min(mid, len(shuffled[k])) for k in sorted_keys) >= n_target:
            hi = mid
        else:
            lo = mid + 1
    n_per_stratum = lo

    selected = {k: shuffled[k][:n_per_stratum] for k in sorted_keys}

    extras = n_target - sum(len(v) for v in selected.values())
    for k in sorted(sorted_keys, key=lambda k: -len(flat_strata[k])):
        if extras <= 0:
            break
        already   = len(selected[k])
        available = len(shuffled[k]) - already
        if available <= 0:
            continue
        take = min(available, extras)
        selected[k].extend(shuffled[k][already: already + take])
        extras -= take

    return [uid for k in sorted_keys for uid in selected[k]]


# ---------------------------------------------------------------------------
# Stage 0 — Database
# ---------------------------------------------------------------------------
def render_database() -> None:
    st.title("Stage 0 - Database")
    tab_ds, tab_bm = st.tabs(["Datasets", "Background Model"])

    with tab_ds:
        st.subheader("Select datasets")
        ds_options = [ds.key for ds in DATASET_REGISTRY]
        ds_labels  = {ds.key: ds.display_name for ds in DATASET_REGISTRY}

        if "selected_datasets" not in st.session_state:
            st.session_state.selected_datasets = (
                [st.session_state.selected_dataset]
                if st.session_state.selected_dataset in ds_options
                else ds_options[:1]
            )

        selected_keys = st.multiselect(
            "Datasets",
            options=ds_options,
            default=st.session_state.selected_datasets,
            format_func=lambda k: ds_labels[k],
            key="ds_selector",
        )
        if not selected_keys:
            st.info("Select at least one dataset.")
            return

        if selected_keys != st.session_state.selected_datasets:
            st.session_state.selected_datasets = selected_keys
            st.session_state.selected_dataset  = selected_keys[-1]
            st.session_state.audio_store       = {}
            st.session_state.db_cm_scores      = {}
            st.session_state.utterances        = {}
            st.session_state.cache_status      = {}
            st.rerun()
        elif not st.session_state.selected_dataset:
            st.session_state.selected_dataset = selected_keys[-1]

        selected_specs = [get_dataset(k) for k in selected_keys]
        st.session_state["_bm_selected_keys"] = selected_keys 

        def _ds_paths_configured(spec) -> bool:
            overrides   = st.session_state.get("path_overrides", {})
            audio_dir   = overrides.get(spec.key, {}).get("audio_dir",   spec.audio_dir)
            label_files = overrides.get(spec.key, {}).get("label_files", spec.label_files)
            return (
                not audio_dir.startswith(_PLACEHOLDER_PREFIX)
                and not any(lf.startswith(_PLACEHOLDER_PREFIX) for lf in label_files)
            )

        unconfigured = [s for s in selected_specs if not _ds_paths_configured(s)]
        if unconfigured:
            names = ", ".join(s.display_name for s in unconfigured)
            st.warning(
                f"**Paths not configured for: {names}.** "
                "Enter your local paths below. "
                "They will be used for this session only — "
                "edit `config.py` to make them permanent.",
                icon="📂",
            )
            with st.form("path_setup_form"):
                form_values = {}
                for spec in unconfigured:
                    st.markdown(f"**{spec.display_name}**")
                    audio_val = st.text_input(
                        "Audio directory",
                        placeholder=spec.audio_dir,
                        key=f"path_audio_{spec.key}",
                    )
                    label_vals = []
                    for i, lf in enumerate(spec.label_files):
                        label_vals.append(st.text_input(
                            f"Label file{' ' + str(i+1) if len(spec.label_files) > 1 else ''}",
                            placeholder=lf,
                            key=f"path_label_{spec.key}_{i}",
                        ))
                    form_values[spec.key] = {"audio_dir": audio_val, "label_files": label_vals}
                st.markdown("**Soundscape (ISD London)**")
                soundscape_val = st.text_input(
                    "Soundscape directory",
                    placeholder=r"path/to/ISD/WAV_London_1/Audio",
                    help="Required for the Soundscape Mix manipulation. Download from zenodo.org/records/10672568.",
                    key="path_soundscape",
                )
                submitted = st.form_submit_button("Apply paths", type="primary")

            if submitted:
                missing = []
                for spec in unconfigured:
                    vals = form_values[spec.key]
                    if not vals["audio_dir"].strip():
                        missing.append(f"{spec.display_name} audio directory")
                    for i, lf in enumerate(vals["label_files"]):
                        if not lf.strip():
                            missing.append(f"{spec.display_name} label file {i+1}")
                if not soundscape_val.strip():
                    missing.append("Soundscape directory")
                if missing:
                    st.error("Please fill in: " + ", ".join(missing))
                else:
                    overrides = dict(st.session_state.get("path_overrides", {}))
                    for spec in unconfigured:
                        overrides[spec.key] = {
                            "audio_dir":   _clean_path(form_values[spec.key]["audio_dir"]),
                            "label_files": [_clean_path(lf) for lf in form_values[spec.key]["label_files"]],
                        }
                    overrides["__soundscape__"] = _clean_path(soundscape_val)
                    apply_path_overrides(overrides)
                    st.success("Paths applied for this session.")
                    st.rerun()
            return

        # Dataset info 
        all_utterances: dict = {}
        for ds in selected_specs:
            with st.expander(f"**{ds.display_name}**", expanded=len(selected_specs) == 1):
                col_i1, col_i2 = st.columns([3, 1])
                col_i1.caption(ds.description)
                col_i1.markdown(f"**License:** {ds.license}")
                col_i1.markdown(f"**Citation:** {ds.citation}")
                if ds.exists():
                    col_i2.success("Paths OK")
                else:
                    col_i2.error("Path not found")
                    st.error(f"Audio dir or label file not found for {ds.display_name}.")
                    continue

            cache_key = f"utterances_{ds.key}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = ds.load_utterances()
            all_utterances.update(st.session_state[cache_key])

        st.session_state.utterances = all_utterances
        utterances = all_utterances

        if not utterances:
            st.warning("No utterances loaded from label files.")
            return

        n_bonafide  = sum(1 for u in utterances.values() if u.effective_label == "bonafide")
        n_synthetic = sum(1 for u in utterances.values() if u.effective_label == "synthetic")
        n_partial   = sum(1 for u in utterances.values() if u.effective_label == "partial synthetic")

        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        col_l1.metric("Total utterances", len(utterances))
        col_l2.metric("Bonafide",          n_bonafide)
        col_l3.metric("Synthetic",         n_synthetic)
        col_l4.metric("Partial synthetic", n_partial)

        # Per-dataset summary panels
        for ds in selected_specs:
            ds_utts = st.session_state.get(f"utterances_{ds.key}", {})
            if not ds_utts:
                continue
            spec = ds.strat_spec
            if spec is None:
                continue
            for strat_field in spec.fields:
                values = sorted({strat_field.extractor(u) for u in ds_utts.values()
                                 if strat_field.extractor(u) not in ("bonafide", "unknown")})
                if values:
                    with st.expander(f"{ds.display_name} — {strat_field.name} ({len(values)})"):
                        st.write(", ".join(values))

        st.divider()
        st.subheader("Precomputed scores")
        st.caption("Scores are cached per dataset per model in `cache/`.")

        # Scoring and cache status per dataset.
        all_missing_keys = set()
        for ds in selected_specs:
            st.markdown(f"**{ds.display_name}**")
            ds_utterances = st.session_state.get(f"utterances_{ds.key}", {})
            expected_n = None
            if ds_utterances:
                try:
                    train_uids_check, _ = ds.compute_bm_split(ds_utterances, CACHE_DIR)
                    wav_index_check = get_wav_index(ds, ds.key)
                    expected_n = sum(1 for uid in train_uids_check if uid in wav_index_check)
                except Exception:
                    pass
            _, missing_keys = render_cache_status_table(ds.key, expected_n=expected_n)
            all_missing_keys.update(missing_keys)

        col_btn1, col_btn2 = st.columns(2)
        compute_missing = col_btn1.button(
            f"Compute missing ({len(all_missing_keys)} model{'s' if len(all_missing_keys) != 1 else ''})",
            type="primary", disabled=len(all_missing_keys) == 0,
        )
        recompute_all = col_btn2.button("Recompute all", disabled=len(MODEL_SPECS) == 0)

        models_to_run = []
        if compute_missing:
            models_to_run = list(all_missing_keys)
        elif recompute_all:
            models_to_run = [s.key for s in MODEL_SPECS]

        with st.expander("Sampling options for cache scoring"):
            col_sp, col_ss = st.columns(2)
            cache_sample_pct  = col_sp.slider("% of train files to score", 1, 100, 10, key="cache_sample_pct")
            cache_sample_seed = col_ss.number_input("Seed", 0, value=42, key="cache_sample_seed")

        if models_to_run:
            for ds in selected_specs:
                ds_utterances = st.session_state.get(f"utterances_{ds.key}", {})
                if not ds_utterances:
                    continue
                st.markdown(f"**Scoring: {ds.display_name}**")
                train_uids, test_uids = ds.compute_bm_split(ds_utterances, CACHE_DIR)
                uid_splits = {uid: "train" for uid in train_uids}
                uid_splits.update({uid: "test" for uid in test_uids})

                wav_index = get_wav_index(ds, ds.key)
                scoreable = [uid for uid in train_uids if uid in wav_index]

                if cache_sample_pct < 100:
                    scoreable = stratified_sample(scoreable, ds_utterances, cache_sample_pct, int(cache_sample_seed),
                                                  strat_spec=ds.strat_spec)

                for model_key in models_to_run:
                    spec     = next(s for s in MODEL_SPECS if s.key == model_key)
                    existing = {}
                    for candidate in (
                        CACHE_DIR / f"{ds.key}_{model_key}.json",
                        CACHE_DIR / f"{ds.key}_{model_key}.partial.json",
                    ):
                        if candidate.exists():
                            try:
                                existing = json.loads(candidate.read_text()).get("scores", {})
                            except Exception:
                                existing = {}
                            break

                    scoreable_set  = set(scoreable)
                    valid_existing = {uid: v for uid, v in existing.items() if uid in scoreable_set}
                    to_score       = [uid for uid in scoreable if uid not in valid_existing]

                    if not to_score:
                        st.caption(f"{spec.display_name} / {ds.display_name} — nothing new to score.")
                        continue

                    st.caption(
                        f"{spec.display_name} — scoring {len(to_score)} / {len(scoreable)} files "
                        f"({len(valid_existing)} already cached)."
                    )
                    prog_sc = st.progress(0)
                    stat_sc = st.empty()

                    score_and_cache(
                        uids=to_score, wav_index=wav_index, utterances=ds_utterances,
                        dataset_key=ds.key, model_key=model_key, cache_dir=CACHE_DIR,
                        existing_scores=valid_existing,
                        uid_splits=uid_splits,
                        progress_cb=lambda done, total: (
                            prog_sc.progress(done / total),
                            stat_sc.caption(f"{spec.display_name} — {done}/{total} files scored"),
                        ),
                    )
                    stat_sc.success(f"{spec.display_name} / {ds.display_name} — scoring complete.")

            st.session_state.cache_status = {}
            st.rerun()

        # Pool cm_scores across all selected datasets.
        pooled_scores = {}
        for ds in selected_specs:
            pooled_scores.update(load_cache(ds.key, CACHE_DIR))
        st.session_state.db_cm_scores = pooled_scores

    with tab_bm:
        if not st.session_state.db_cm_scores:
            st.warning("No scored files found. Use the Datasets tab to compute scores.")
            return

        _bm_keys   = st.session_state.get("_bm_selected_keys", [])
        if not _bm_keys:
            st.info("Select datasets in the Datasets tab first.")
            return
        selected_specs = [get_dataset(k) for k in _bm_keys]

        bm_models = st.multiselect(
            "Models", [s.key for s in MODEL_SPECS],
            default=[s.key for s in MODEL_SPECS],
            format_func=MODEL_DISPLAY.get,
            key="bm_models",
        )
        active_specs  = [s for s in MODEL_SPECS if s.key in bm_models] if bm_models else MODEL_SPECS
        model_eer_cache = {}

        for spec in active_specs:
            nat_cm, syn_cm = [], []
            file_scores    = {}

            uid_to_ds: dict = {}
            for ds_candidate in selected_specs:
                for cuid in st.session_state.get(f"utterances_{ds_candidate.key}", {}):
                    uid_to_ds[cuid] = ds_candidate       

            per_ds_nat: dict = {ds_s.key: [] for ds_s in selected_specs}
            per_ds_syn: dict = {ds_s.key: [] for ds_s in selected_specs}

            for uid, entries in st.session_state.db_cm_scores.items():
                for e in entries:
                    if e.get("model") != spec.key:
                        continue
                    gt, cm = e["gt"], e["cm"]
                    ds_spec = uid_to_ds.get(uid)          
                    ds_key  = ds_spec.key if ds_spec else None 
                    if gt == "bonafide":
                        nat_cm.append(cm)
                        file_scores[uid] = {"cm": cm, "label": "bonafide"}
                        if ds_key and ds_key in per_ds_nat:
                            per_ds_nat[ds_key].append(cm)
                    elif gt in ("synthetic", "partial synthetic"):
                        syn_cm.append(cm)
                        file_scores[uid] = {"cm": cm, "label": "synthetic"}
                        if ds_key and ds_key in per_ds_syn:
                            per_ds_syn[ds_key].append(cm)

            st.markdown(f"**{spec.display_name}**")
            st.caption(
                "One CM score per utterance (utterance-level inference). "
                "Partial synthetic utterances are included in the synthetic distribution."
            )

            eer_info = None
            if len(nat_cm) >= 2 and len(syn_cm) >= 2:
                try:
                    sc      = np.array(nat_cm + syn_cm, dtype=np.float64)
                    lb      = np.array([0] * len(nat_cm) + [1] * len(syn_cm), dtype=np.int32)
                    eer_val, eer_tau, _, _ = compute_eer(sc, lb)
                    tau_d    = float(np.clip(eer_tau, sc.min(), sc.max()))
                    eer_info = (eer_val, tau_d, eer_tau)
                    model_eer_cache[spec.key] = (eer_val, eer_tau, file_scores)
                except Exception:
                    model_eer_cache[spec.key] = (None, None, file_scores)

            plot_dict = {}
            if len(nat_cm) >= 2: plot_dict["bonafide"]  = nat_cm
            if len(syn_cm) >= 2: plot_dict["synthetic"] = syn_cm

            all_sc = [v for lst in plot_dict.values() for v in lst]
            if not all_sc:
                st.caption("No scores available for this model.")
                st.divider()
                continue

            fig = kde_plot(plot_dict, title=f"Background Model — {spec.display_name}")
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            cllr_val = None
            lr_values, lr_labels = compute_lr_scores(nat_cm, syn_cm)
            if lr_values is not None:
                try:
                    cllr_val   = compute_cllr(lr_values, lr_labels)
                except Exception:
                    pass

            n_cols     = len(plot_dict) + (1 if eer_info else 0) + (1 if cllr_val is not None else 0)
            count_cols = st.columns(n_cols)
 
            for i, (lbl, scores) in enumerate(plot_dict.items()):
                count_cols[i].metric(lbl, len(scores))
            col_idx = len(plot_dict)
 
            if eer_info:
                count_cols[col_idx].metric("EER (utterance-level)", f"{eer_val * 100:.1f}%")
                col_idx += 1
 
            if cllr_val is not None:
                count_cols[col_idx].metric(
                    "C_llr",
                    f"{cllr_val:.4f}",
                    help="Log-likelihood-ratio cost. 0 = perfect, 1 = no better than chance, >1 = miscalibrated.",
                )

            with st.expander("Performance & fairness breakdown", expanded=False):
                threshold = 0.0
                utt_store = st.session_state.utterances

                per_ds_rows: dict = {ds_s.key: [] for ds_s in selected_specs}

                for uid, entries in st.session_state.db_cm_scores.items():
                    for e in entries:
                        if e.get("model") != spec.key:
                            continue
                        utt = utt_store.get(uid)
                        if utt is None:
                            continue
                        ds_for_uid = uid_to_ds.get(uid)
                        if ds_for_uid is None:
                            continue
                        n_segs     = len(utt.segments)
                        n_syn_segs = sum(1 for s in utt.segments if s.label != "bonafide")
                        pct_syn    = (
                            0.0     if utt.effective_label == "bonafide"
                            else 100.0 if n_segs == 0
                            else n_syn_segs / n_segs * 100.0
                        )
                        cm   = float(e["cm"])
                        pred = 1 if cm >= threshold else 0
                        gt   = 0 if utt.effective_label == "bonafide" else 1
                        row  = {
                            "uid":     uid,
                            "cm":      cm,
                            "pred":    pred,
                            "gt":      gt,
                            "gt_fine": utt.effective_label,
                            "pct_syn": pct_syn,
                        }
                        if ds_for_uid.strat_spec is not None:
                            for strat_field in ds_for_uid.strat_spec.fields:
                                row[strat_field.name] = strat_field.extractor(utt)
                        per_ds_rows[ds_for_uid.key].append(row)

                pct_order = ["0% — bonafide", "1-49%", "50-99%", "100% — fully synthetic"]

                for ds_s in selected_specs:
                    bk_rows = per_ds_rows[ds_s.key]
                    if not bk_rows:
                        st.caption(f"{ds_s.display_name}: no scored utterances found.")
                        continue

                    st.markdown(f"**{ds_s.display_name}**")

                    ds_field_names: list = []
                    if ds_s.strat_spec is not None:
                        for strat_field in ds_s.strat_spec.fields:
                            ds_field_names.append(strat_field.name)

                    tab_labels = ["% Synthetic"] + ds_field_names
                    tabs = st.tabs(tab_labels)

                    with tabs[0]:
                        st.caption("Statistical parity by fraction of utterance that is synthetic.")
                        stat_parity_table(bk_rows, lambda r: pct_syn_bucket(r["pct_syn"]),
                                          "% Synthetic", order=pct_order)

                    for tab_widget, field_name in zip(tabs[1:], ds_field_names):
                        with tab_widget:
                            st.caption(f"Statistical parity by {field_name} ({ds_s.display_name}).")
                            field_rows = [r for r in bk_rows if field_name in r]
                            if field_rows:
                                stat_parity_table(field_rows,
                                                  lambda r, fn=field_name: r.get(fn, "—"),
                                                  field_name)
                            else:
                                st.caption("No data for this field.")

                    if len(selected_specs) > 1:
                        st.divider()
            
            # Per-dataset score breakdown (only meaningful when >1 dataset loaded).
            if len(selected_specs) > 1:
                with st.expander("Per-dataset score breakdown", expanded=False):
                    ds_cols = st.columns(len(selected_specs))
                    for col, ds_s in zip(ds_cols, selected_specs):
                        n_nat = len(per_ds_nat.get(ds_s.key, []))
                        n_syn = len(per_ds_syn.get(ds_s.key, []))
                        with col:
                            st.markdown(f"**{ds_s.display_name}**")
                            st.metric("Bonafide",  n_nat)
                            st.metric("Synthetic", n_syn)
                            if n_nat >= 2 and n_syn >= 2:
                                try:
                                    sc_ds = np.array(
                                        per_ds_nat[ds_s.key] + per_ds_syn[ds_s.key],
                                        dtype=np.float64,
                                    )
                                    lb_ds = np.array(
                                        [0] * n_nat + [1] * n_syn,
                                        dtype=np.int32,
                                    )
                                    eer_ds, _, _, _ = compute_eer(sc_ds, lb_ds)
                                    st.metric("EER", f"{eer_ds * 100:.1f}%")
                                except Exception:
                                    st.caption("EER unavailable.")
                            else:
                                st.caption("Insufficient data for EER.")

                    # KDE overlay per dataset.
                    fig_ds = per_dataset_kde(selected_specs, per_ds_nat, per_ds_syn)
                    if fig_ds:
                        st.pyplot(fig_ds, width="stretch")
                        plt.close(fig_ds)

# ---------------------------------------------------------------------------
# Stage 1 — Inspection
# ---------------------------------------------------------------------------
def render_inspection() -> None:
    with st.sidebar:
        st.markdown("### Suspected manipulations")
        pipeline_sidebar("insp", "insp_pipeline", title="Manipulation pipeline")
        st.divider()
        st.markdown("### Detection settings")
        insp_models = st.multiselect(
            "Models", [s.key for s in MODEL_SPECS],
            default=[s.key for s in MODEL_SPECS],
            format_func=MODEL_DISPLAY.get,
            key="insp_models",
        )

    st.title("Stage 1 - Inspection")
    st.caption("Upload the target audio. Inspect, listen, and annotate suspected manipulations.")

    f = st.file_uploader("Upload target audio", type=["wav", "flac", "mp3"], key="insp_up")
    if f is not None and f.name != st.session_state.get("_insp_last_filename", ""):
        audio, sr = load_wav_bytes(f.read())
        st.session_state.insp_audio           = audio
        st.session_state.insp_sr              = sr
        st.session_state.insp_name            = Path(f.name).stem
        st.session_state._insp_last_filename  = f.name
        st.session_state.insp_det             = {}
        st.session_state.insp_windowed        = {}
        st.session_state.insp_acoustic        = {}
        st.session_state.insp_auditory        = {}
        st.session_state.insp_transcript      = None
        st.session_state.insp_lime            = {}
        st.session_state.insp_lime_result     = {}
        st.session_state.insp_det_annotation  = {}

    audio = st.session_state.insp_audio
    if audio is None:
        st.info("Upload a target audio file to begin.")
        return

    sr          = st.session_state.insp_sr
    name        = st.session_state.insp_name
    insp_models = st.session_state.get("insp_models", [s.key for s in MODEL_SPECS])
    insp_thr    = 0.5  # UI threshold; principled threshold set in Stage 2.

    col1, col2, col3 = st.columns(3)
    col1.metric("Duration (s)", round(len(audio) / sr, 2))
    col2.metric("Sample rate", sr)
    col3.metric("File", name)
    waveform_player(audio, sr, label="Target audio")
    st.markdown(
        "**Hypothesis pipeline:** " + pipeline_badge(st.session_state.insp_pipeline),
        unsafe_allow_html=True,
    )

    tab_acoust, tab_aud, tab_det = st.tabs(["Acoustic Analysis", "Auditory Analysis", "Synthetic Content Analysis"])

    with tab_aud:
        st.caption("Record auditory observations heard during listening.")
        aud = st.session_state.insp_auditory

        def aud_section(title, key, options, multiselect=True):
            st.markdown(f"**{title}**")
            if multiselect:
                val = st.multiselect(
                    title, options=options, default=aud.get(key, []),
                    label_visibility="collapsed", key=f"aud_{key}",
                )
            else:
                all_opts = ["(not assessed)"] + options
                val = st.selectbox(
                    title, options=all_opts,
                    index=all_opts.index(aud.get(key, "(not assessed)")),
                    label_visibility="collapsed", key=f"aud_{key}",
                )
            aud[key] = val

        col_a, col_b = st.columns(2)

        with col_a:
            aud_section(
                "Voice quality", "voice_quality",
                ["Breathy", "Creaky", "Harsh",
                 "Falsetto", "Tense", "Whispery"]
            )
            aud_section(
                "Speaking rate", "speaking_rate",
                ["Very slow", "Slow", "Average", "Fast", "Very fast", "Variable / irregular"],
                multiselect=False,
            )
            aud_section(
                "Prosodic features", "prosody",
                ["Monotone pitch", "Wide pitch range", "Narrow pitch range",
                 "Rising intonation", "Falling intonation",
                 "Stress-timed rhythm", "Syllable-timed rhythm", "Irregular rhythm"],
            )
            aud_section(
                "Pitch level impression", "pitch_level",
                ["Very low", "Low", "Mid", "High", "Very high"],
                multiselect=False,
            )

        with col_b:
            aud_section(
                "Suspected intraspeaker variability factors", "variability_factors",
                ["Apparent emotional state change", "Apparent fatigue / drowsiness",
                 "Register shift (formal -> casual)", "Loudness change",
                 "Tempo change mid-utterance",
                 "Apparent disguise attempt (pitch raising/lowering)",
                 "Apparent disguise attempt (accent modification)",
                 "Possible read speech (vs. spontaneous)",
                 "Possible scripted / rehearsed delivery", "Possible stress reaction"],
            )
            aud_section(
                "Transmission / recording conditions", "transmission",
                ["Telephone", "Room acoustics / reverb audible", "Background noise (stationary)",
                 "Background noise (non-stationary)", "Clipping / overload distortion",
                 "Re-recording artefact suspected", "ENF hum audible"],
            )
            aud_section(
                "Idiosyncratic features", "idiosyncratic",
                ["Audible lip smacks / clicks", "Habitual filled pauses (uh, um)",
                 "Distinctive laugh / cough", "Non-native accent (L2 features)"],
            )

        st.divider()
        st.markdown("**Free-text observations**")
        aud["notes"] = st.text_area(
            "Additional observations", value=aud.get("notes", ""), height=100,
            placeholder=(
                "Record any observations not captured above."
            ),
            key="aud_notes", label_visibility="collapsed",
        )

        if st.button("Save auditory annotation", type="primary", key="aud_save"):
            st.session_state.insp_auditory = dict(aud)
            st.success("Auditory annotation saved.")

    with tab_acoust:
        from acoustic_analysis import render_acoustic_tab
        render_acoustic_tab(audio, sr, name)

    with tab_det:
        from synth_content_analysis import render_synthetic_content_tab
        render_synthetic_content_tab(audio, sr, name, insp_models, insp_thr)      


# ---------------------------------------------------------------------------
# Stage 2 — Analysis
# ---------------------------------------------------------------------------
def render_analysis() -> None:
    with st.sidebar:
        insp_pipe = st.session_state.insp_pipeline
        if insp_pipe:
            st.markdown("**Stage 1 pipeline:**")
            st.markdown(pipeline_badge(insp_pipe), unsafe_allow_html=True)
            if st.button("Copy to M-BM pipeline"):
                st.session_state.an_mbm_pipeline = list(insp_pipe)
                st.session_state.an_preset_key   = None
                st.rerun()

        st.markdown("### M-BM Pipeline")

        preset_options = ["— custom (use builder below) —"] + [
            f"{info['display']}"
            for info in PRESET_PIPELINES.values()
        ]
        preset_keys         = [None] + list(PRESET_PIPELINES.keys())
        current_preset_key  = st.session_state.get("an_preset_key", None)
        current_index       = preset_keys.index(current_preset_key) if current_preset_key in preset_keys else 0

        selected_option = st.selectbox(
            "Preset pipeline", options=preset_options, index=current_index,
            key="an_preset_selector",
            help=(
                "Select a named preset to load it immediately. "
                "Δ syn is the mean CM shift on synthetic files from "
                "mini scoring — more negative = better at fooling the detector."
            ),
        )

        selected_preset_key = preset_keys[preset_options.index(selected_option)]

        if selected_preset_key != current_preset_key:
            if selected_preset_key is not None:
                st.session_state.an_mbm_pipeline = list(PRESET_PIPELINES[selected_preset_key]["steps"])
                st.session_state.an_preset_key   = selected_preset_key
            else:
                st.session_state.an_preset_key = None
            st.rerun()

        if selected_preset_key is not None:
            st.success(f"Loaded: **{selected_preset_key}**")
            st.caption(PRESET_PIPELINES[selected_preset_key]["description"])
        else:
            st.caption("No preset selected — build a custom pipeline below.")

        with st.expander("Custom pipeline builder", expanded=False):
            pipeline_sidebar("an", "an_mbm_pipeline", title="")


    st.title("Stage 2 - Analysis")

    audio = st.session_state.insp_audio
    if audio is None:
        st.warning("Upload a target in Stage 1 first.")
        return

    sr           = st.session_state.insp_sr
    insp_det     = st.session_state.insp_det
    an_models    = st.session_state.get("insp_models", [s.key for s in MODEL_SPECS])
    mbm_pipeline = st.session_state.an_mbm_pipeline
    utterances   = st.session_state.utterances

    if not st.session_state.db_cm_scores:
        st.warning("No BM scores found. Go to Stage 0 -> Background Model tab to load scores.")
        return
    if not st.session_state.selected_dataset:
        st.warning("No dataset selected. Go to Stage 0 first.")
        return

    ds           = get_dataset(st.session_state.selected_dataset)
    selected_key = st.session_state.selected_dataset
    selected_keys_an  = st.session_state.get("selected_datasets", [selected_key])
    selected_specs_an = [get_dataset(k) for k in selected_keys_an]
    pipe_hash    = pipeline_cache_key(mbm_pipeline)

    st.markdown("**M-BM pipeline:** " + pipeline_badge(mbm_pipeline), unsafe_allow_html=True)
    if not insp_det:
        st.info("Run detection in Stage 1 first to enable target overlay on distributions.")

    _uses_soundscape = any(s.name == "Soundscape Mix" for s in mbm_pipeline)
    if _uses_soundscape:
        import config as _cfg
        if _cfg.SOUNDSCAPE_DIR.startswith("path/to"):
            st.warning(
                "**Soundscape Mix is in your pipeline but the ISD path is not set.** "
                "Enter the path to the WAV_London_1/Audio folder below. "
                "Download from [Zenodo](https://zenodo.org/records/10672568) if needed.",
                icon="🔊",
            )
            _snd_input = st.text_input(
                "Soundscape directory (ISD London)",
                placeholder=r"path/to/ISD/WAV_London_1/Audio",
                key="an_soundscape_dir_input",
            )
            if st.button("Apply soundscape path", key="an_soundscape_apply"):
                _snd_clean = _snd_input.strip()
                if len(_snd_clean) >= 2 and _snd_clean[0] in ('"', "'") and _snd_clean[-1] == _snd_clean[0]:
                    _snd_clean = _snd_clean[1:-1]
                if not _snd_clean:
                    st.error("Please enter a path.")
                else:
                    _cfg.SOUNDSCAPE_DIR = _snd_clean
                    _overrides = st.session_state.get("path_overrides", {})
                    _overrides["__soundscape__"] = _snd_clean
                    st.session_state["path_overrides"] = _overrides
                    st.success("Soundscape path applied.")
                    st.rerun()
            st.stop()

    main_tab, preview_tab, miniscoring_tab = st.tabs(["M-BM Analysis", "Preview Manipulations", "Run Scoring"])
    
    with main_tab:
        # M-BM cache status and scoring
        st.subheader("M-BM Score Cache")

        current_preset_key = st.session_state.get("an_preset_key", None)
        active_cache_key   = current_preset_key if current_preset_key is not None else pipeline_cache_key(mbm_pipeline)

        st.caption(
            f"Cache key: `{active_cache_key}` — "
            f"{'preset pipeline' if current_preset_key else 'custom pipeline (hash)'}."
        )

        mbm_cached     = {}
        mbm_cache_rows = []
        missing_models = []

        for ds_an in selected_specs_an:
            for spec in MODEL_SPECS:
                if spec.key not in an_models:
                    continue
                cache_path = mbm_cache_path(ds_an.key, spec.key, active_cache_key)
                scores     = load_mbm_cache_file(cache_path)
                is_complete = bool(scores)
                if is_complete:
                    scored_at = json.loads(cache_path.read_text()).get("scored_at", "unknown")
                    mbm_cached[spec.key] = scores
                    mbm_cache_rows.append({
                        "Dataset": ds_an.display_name,
                        "Model":     spec.display_name,
                        "Status":    "✅",
                        "Files":     len(scores),
                        "Scored at": scored_at,
                        "_ds_key":   ds_an.key,
                        "_key":      spec.key,
                        "_complete": True,
                    })
                else:
                    mbm_cache_rows.append({
                        "Dataset": ds_an.display_name,
                        "Model":     spec.display_name,
                        "Status":    "❌",
                        "Files":     None,
                        "Scored at": "—",
                        "_ds_key":   ds_an.key,
                        "_key":      spec.key,
                        "_complete": False,
                    })
                    missing_models.append((ds_an.key, spec.key))

        st.dataframe(
            pd.DataFrame(mbm_cache_rows).drop(columns=["_ds_key", "_key", "_complete"]),
            use_container_width=True, hide_index=True,
        )

        col_single, col_all, col_clr = st.columns([2, 2, 1])

        run_active = col_single.button(
            f"Score active pipeline  [{active_cache_key}]"
            + (f"  ({len(missing_models)} missing)" if missing_models else "  ✅"),
            type="primary", disabled=len(missing_models) == 0, key="mbm_run_active",
        )

        presets_missing = [
            (pkey, ds_an.key, spec.key)
            for pkey in PRESET_PIPELINES
            for ds_an in selected_specs_an
            for spec in MODEL_SPECS
            if spec.key in an_models
            and not load_mbm_cache_file(mbm_cache_path(ds_an.key, spec.key, pkey))
        ]
        flagged_presets = list(dict.fromkeys(pkey for pkey, _, _ in presets_missing))

        run_all_presets = col_all.button(
            f"Score all presets  ({len(flagged_presets)} incomplete)",
            disabled=len(flagged_presets) == 0, key="mbm_run_all_presets",
        )

        if col_clr.button("Clear cache", key="mbm_clr"):
            for ds_an in selected_specs_an:
                for spec in MODEL_SPECS:
                    for path in [mbm_cache_path(ds_an.key, spec.key, active_cache_key),
                                mbm_partial_path(ds_an.key, spec.key, active_cache_key)]:
                        if path.exists():
                            path.unlink()
            st.session_state.an_lr_results = {}
            st.rerun()

        if run_active and missing_models:
            if not mbm_pipeline:
                st.warning("No pipeline steps defined.")
            else:
                for ds_key_missing, model_key_missing in missing_models:
                    ds_missing = get_dataset(ds_key_missing)
                    if not ds_missing.exists():
                        st.error(f"Dataset path not found: {ds_missing.display_name}")
                        continue
                    utts_missing = st.session_state.get(f"utterances_{ds_key_missing}", {})
                    run_mbm_scoring(
                        mbm_pipeline, active_cache_key,
                        [model_key_missing], ds_key_missing,
                        ds_missing, utts_missing, an_models,
                    )
                st.rerun()

        if run_all_presets:
            for pkey, pinfo in PRESET_PIPELINES.items():
                for ds_an in selected_specs_an:
                    models_needed = [
                        spec.key for spec in MODEL_SPECS
                        if spec.key in an_models
                        and not load_mbm_cache_file(mbm_cache_path(ds_an.key, spec.key, pkey))
                    ]
                    if not models_needed:
                        continue
                    if not ds_an.exists():
                        st.error(f"Dataset path not found: {ds_an.display_name}")
                        continue
                    utts_an = st.session_state.get(f"utterances_{ds_an.key}", {})
                    st.markdown(f"**{pkey}** — {pinfo['display']} ({ds_an.display_name})")
                    run_mbm_scoring(pinfo["steps"], pkey, models_needed, ds_an.key, ds_an, utts_an, an_models)
            st.rerun()

        # Load all cached presets for downstream analysis
        for pkey in PRESET_PIPELINES:
            for ds_an in selected_specs_an:
                for spec in MODEL_SPECS:
                    if spec.key not in an_models:
                        continue
                    scores = load_mbm_cache_file(mbm_cache_path(ds_an.key, spec.key, pkey))
                    if scores:
                        mbm_cached[(pkey, spec.key)] = scores

        if not mbm_cached:
            st.info("No M-BM scores cached yet. Select a preset or build a custom pipeline, then click Score.")
            return

        pre  = st.session_state.db_cm_scores
        post = {}
        for model_key, scores in mbm_cached.items():
            for uid, entry in scores.items():
                if isinstance(entry, dict):
                    post.setdefault(uid, []).append({"model": model_key, "cm": entry["cm"], "gt": entry["gt"]})

        empty_lr = {"bonafide": [], "synthetic": [], "partial synthetic": []}

        for spec in MODEL_SPECS:
            if spec.key not in an_models or spec.key not in mbm_cached:
                continue

            st.subheader(spec.display_name)

            agg_pre  = agg_scores(pre,  spec.key)
            agg_post = agg_scores(post, spec.key)

            bm_by,  bm_scores  = build_plot_and_lr_dicts(agg_pre)  if agg_pre  else ({}, empty_lr)
            mbm_by, mbm_scores = build_plot_and_lr_dicts(agg_post) if agg_post else ({}, empty_lr)
            st.session_state.setdefault("an_bm_scores",  {})[spec.key] = bm_scores
            st.session_state.setdefault("an_mbm_scores", {})[spec.key] = mbm_scores

            det_r     = insp_det.get(spec.key)
            target_cm = det_r.cm_score if (det_r and not det_r.error) else None

            # Side-by-side BM vs M-BM KDE
            fig = bm_kde_sidebyside(bm_by, mbm_by, target_cm, pipe_hash)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            # Overlay plot
            fig2 = bm_kde_overlay(bm_by, mbm_by, target_cm)
            if fig2:
                st.pyplot(fig2, width="stretch")
                plt.close(fig2)

            lr_col, neighbour_col =st.columns([3,3])
            with lr_col:
                # LR assessment
                if target_cm is not None:
                    import math
                    lr_result = compute_likelihood_ratio(target_cm, bm_scores, mbm_scores)
                    st.session_state.an_lr_results[spec.key] = lr_result

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Target CM",         round(target_cm, 4))
                    mc2.metric("LR (authenticity)", f"{lr_result.lr_authenticity:.2e}"
                            if lr_result.lr_authenticity is not None else "N/A")
                    mc3.metric("LR (manipulation)", f"{lr_result.lr_manipulation:.2e}"
                            if lr_result.lr_manipulation is not None else "N/A")

                    strength_str = verbal_lr_strength(lr_result.lr_authenticity)
                    if lr_result.lr_authenticity is not None and not math.isnan(float(lr_result.lr_authenticity or 0)):
                        if lr_result.lr_authenticity >= 1.0:
                            st.success(f"**Evidence strength:** {strength_str}")
                        elif lr_result.lr_authenticity >= 0.1:
                            st.info(f"**Evidence strength:** {strength_str}")
                        else:
                            st.warning(f"**Evidence strength:** {strength_str}")
                    else:
                        st.info(f"**Evidence strength:** {strength_str}")
            
                    from background_model import evaluate_synthetic_evasion

                    evasion = evaluate_synthetic_evasion(bm_scores, mbm_scores)
                    st.markdown("### Synthetic Evasion",
                                help = "Tests whether the manipulation pipeline shifts synthetic CM scores "
                        "toward the bonafide distribution, i.e. whether the processing "
                        "makes synthetic speech harder to detect. A negative mean shift "
                        "and increased overlap indicate the pipeline is consistent with "
                        "having been applied to obfuscate synthetic content.")

                    if evasion.error:
                        st.warning(f"Evasion analysis unavailable: {evasion.error}")
                    else:
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            m_col1.metric(
                            "Mean CM shift (synthetic)",
                            f"{evasion.mean_shift:+.3f}" if evasion.mean_shift is not None else "N/A",
                            help="Mean M-BM synthetic CM - mean BM synthetic CM. "
                                "Negative = shifted toward bonafide = harder to detect.",
                            )
                            #m_col1.metric(
                            #    "Evading BM (%)",
                            #    f"{evasion.pct_evading_bm:.1f}%" if evasion.pct_evading_bm is not None else "N/A",
                            #    help="% of BM synthetic scores below CM = 0 (misclassified as bonafide).",
                            #)
                        with m_col2:
                            m_col2.metric(
                                "Distribution overlap increase",
                                f"{evasion.overlap_increase:+.4f}" if evasion.overlap_increase is not None else "N/A",
                                help="Bhattacharyya coefficient increase between bonafide and "
                                    "synthetic distributions. Positive = more confused.",
                            )
                            #m_col2.metric(
                            #    "Evading M-BM (%)",
                            #    f"{evasion.pct_evading_mbm:.1f}%" if evasion.pct_evading_mbm is not None else "N/A",
                            #    help="% of M-BM synthetic scores below CM = 0.",
                            #)

                        if evasion.mean_shift is not None and evasion.mean_shift < 0:
                            st.success(f"**{evasion.evasion_verdict}**")
                        elif "not supported" in evasion.evasion_verdict.lower():
                            st.warning(f"**{evasion.evasion_verdict}**")
                        else:
                            st.info(f"**{evasion.evasion_verdict}**")

                else:
                    st.info("Run detection on target in Stage 1 to enable LR assessment.")
                    
            with neighbour_col:
                if target_cm is not None and agg_pre and agg_post:
                    bm_nat = np.array([d["cm"] for d in agg_pre.values() if d["label"] == "bonafide"])
                    sd_w = float(np.std(bm_nat)) if len(bm_nat) > 1 else 1
                    nb_bm  = neighbourhood(agg_pre,  target_cm, sd_w)
                    nb_mbm = neighbourhood(agg_post, target_cm, sd_w)
                    fig = neighbourhood_composition(nb_bm, nb_mbm, sd_w)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    with preview_tab:
        # Pipeline preview
        st.markdown("🔊 Pipeline preview",
            help = "One file per label class is loaded from the database and the current pipeline "
                    "is applied. Listen to originals and processed versions to sanity-check the "
                    "manipulation before running it on the full database.")

        if not mbm_pipeline:
            st.info("Add manipulation steps in the sidebar to enable preview.")
        else:
            merged_wav_index: dict = {}
            uid_to_preview_ds: dict = {}
            unavailable_ds = []
            for ds_an in selected_specs_an:
                if not ds_an.exists():
                    unavailable_ds.append(ds_an.display_name)
                    continue
                ds_wav = get_wav_index(ds_an, ds_an.key)
                for uid, path in ds_wav.items():
                    merged_wav_index[uid] = path
                    uid_to_preview_ds[uid] = ds_an

            if unavailable_ds:
                st.warning(f"Audio path not found for: {', '.join(unavailable_ds)}. ")
            else:
                preview_uids = {}
                rng_prev     = random.Random(42)
                for label in LABEL_CLASSES:
                    candidates = [
                        uid for uid, utt in utterances.items()
                        if utt.effective_label == label and uid in merged_wav_index
                    ]
                    if candidates:
                        preview_uids[label] = rng_prev.choice(candidates)

                if not preview_uids:
                    st.warning("No audio files found on disk. Check paths in Stage 0.")
                else:
                    st.markdown("**Select preview files** (one per class):")
                    selected_preview = {}
                    rng_pool = random.Random(99) 
                    
                    for col, (label, default_uid) in zip(st.columns(len(preview_uids)), preview_uids.items()):
                        per_ds_candidates: dict = {}
                        for ds_an in selected_specs_an:
                            ds_uids = [
                                uid for uid in st.session_state.get(f"utterances_{ds_an.key}", {})
                                if uid in merged_wav_index
                                and utterances.get(uid) is not None
                                and utterances[uid].effective_label == label
                            ]
                            if ds_uids:
                                per_ds_candidates[ds_an.key] = ds_uids

                        pool: list = []
                        if per_ds_candidates:
                            ds_keys_cycle = list(per_ds_candidates.keys())
                            shuffled = {k: sorted(v) for k, v in per_ds_candidates.items()}
                            for v in shuffled.values():
                                rng_pool.shuffle(v)
                            i = 0
                            while len(pool) < 20:
                                ds_k = ds_keys_cycle[i % len(ds_keys_cycle)]
                                remaining = shuffled[ds_k]
                                if remaining:
                                    pool.append(remaining.pop(0))
                                i += 1
                                if all(len(v) == 0 for v in shuffled.values()):
                                    break

                        if not pool:
                            col.caption(f"No {label} files on disk.")
                            continue

                        if default_uid not in pool:
                            pool.insert(0, default_uid)

                        def fmt_option(uid):
                            src = uid_to_preview_ds.get(uid)
                            return f"{uid}  ({src.display_name})" if src else uid

                        default_idx = pool.index(default_uid)
                        selected_uid = col.selectbox(
                            label.capitalize(),
                            options=pool,
                            index=default_idx,
                            format_func=fmt_option,
                            key=f"an_prev_{label}",
                        )
                        selected_preview[label] = selected_uid
                        
                    if st.button("Apply pipeline to preview files", key="an_prev_run"):
                        st.session_state["an_preview_results"] = {}
                        for label, uid in selected_preview.items():
                            audio_path = merged_wav_index.get(uid)
                            if audio_path is None:
                                st.session_state["an_preview_results"][label] = {
                                    "uid": uid, "error": "File not found in any dataset index."
                                }
                                continue
                            try:
                                try:
                                    import io as _io
                                    with open(str(audio_path), "rb") as _fh:
                                        _buf = _io.BytesIO(_fh.read())
                                    raw_audio, raw_sr = sf.read(_buf, dtype="float32", always_2d=False)
                                    if raw_audio.ndim > 1:
                                        raw_audio = raw_audio.mean(axis=1)
                                except Exception as _sf_exc:
                                    raise RuntimeError(
                                        f"Could not decode {audio_path.name}: {_sf_exc}. "
                                        f"The file may be corrupted or truncated."
                                    ) from _sf_exc
                                manip_audio = apply_pipeline(raw_audio, raw_sr, mbm_pipeline, utterance=utterances.get(uid))
                                st.session_state["an_preview_results"][label] = {
                                    "uid": uid, "original": (raw_audio, raw_sr), "processed": (manip_audio, raw_sr),
                                }
                            except Exception as exc:
                                st.session_state["an_preview_results"][label] = {"uid": uid, "error": str(exc)}

                    preview_results = st.session_state.get("an_preview_results", {})
                    if preview_results:
                        st.divider()
                        for label, result in preview_results.items():
                            color    = LABEL_COLORS.get(label, "#888")
                            src_ds   = uid_to_preview_ds.get(result["uid"])
                            src_name = f" · {src_ds.display_name}" if src_ds else ""
                            st.markdown(
                                f'<span style="background:{color};color:#fff;border-radius:3px;'
                                f'padding:2px 8px;font-size:0.85em;">{label}</span> '
                                f'<span style="font-size:0.8em;color:#aaa;">{result["uid"]}{src_name}</span>',
                                unsafe_allow_html=True,
                            )
                            if "error" in result:
                                st.error(f"Pipeline error: {result['error']}")
                                continue
                            orig_audio, orig_sr = result["original"]
                            proc_audio, proc_sr = result["processed"]
                            col_orig, col_proc  = st.columns(2)
                            with col_orig:
                                waveform_player(orig_audio, orig_sr, label="Original")
                            with col_proc:
                                waveform_player(proc_audio, proc_sr, label="Processed")
                            dl1, dl2 = st.columns(2)
                            dl1.download_button(
                                "Download original", data=audio_to_bytes(orig_audio, orig_sr),
                                file_name=f"{result['uid']}_original.wav", mime="audio/wav",
                                key=f"an_dl_orig_{label}",
                            )
                            dl2.download_button(
                                "Download processed", data=audio_to_bytes(proc_audio, proc_sr),
                                file_name=f"{result['uid']}_processed.wav", mime="audio/wav",
                                key=f"an_dl_proc_{label}",
                            )
                    st.divider()

    with miniscoring_tab:
        st.markdown("📊 Mini scoring — rank all manipulation combinations", 
            help= "Applies every non-empty subset of the manipulations to a small stratified sample "
                    "and re-scores with the detection model. Results are ranked by mean CM shift on "
                    "synthetic files; the most negative Δ synthetic means the combination best fools "
                    "the detector. Use this to pick which pipelines to run as full MBM.")

        if not st.session_state.db_cm_scores:
            st.warning("No BM scores loaded. Complete Stage 0 first.")
        elif not ds.exists():
            st.error("Dataset audio path not found. Check paths in `config.py`.")
        else:
            col_n, col_seed = st.columns(2)
            mini_n_per_class = col_n.slider("Files per label class", 5, 50, 20, step=5, key="mini_n_per_class")
            mini_seed        = col_seed.number_input("Sample seed", 0, value=42, step=1, key="mini_seed")

            n_combos = 2 ** len(MANIPULATIONS) - 1
            st.caption(f"{len(MANIPULATIONS)} manipulations -> {n_combos} combinations.")

            if st.button("Run all combinations", type="primary", key="mini_run"):
                wav_index = get_wav_index(ds, selected_key)

                by_label = {lbl: [] for lbl in LABEL_CLASSES}
                for uid, entries in st.session_state.db_cm_scores.items():
                    if uid not in wav_index:
                        continue
                    utt = utterances.get(uid)
                    if utt and utt.effective_label in by_label:
                        by_label[utt.effective_label].append(uid)

                rng_sample = random.Random(int(mini_seed))
                sample_uids = []
                for lbl in LABEL_CLASSES:
                    pool = sorted(by_label[lbl])
                    rng_sample.shuffle(pool)
                    for uid in pool[:mini_n_per_class]:
                        sample_uids.append((uid, lbl, pool.index(uid)))

                if not sample_uids:
                    st.warning("No scoreable UIDs found in the BM cache that are also on disk.")
                else:
                    st.info(f"Loading {len(sample_uids)} audio files into memory…")
                    audio_cache = {}
                    for uid, lbl, uid_idx in sample_uids:
                        try:
                            raw_audio, raw_sr = sf.read(str(wav_index[uid]), dtype="float32", always_2d=False)
                            if raw_audio.ndim > 1:
                                raw_audio = raw_audio.mean(axis=1)
                            audio_cache[uid] = (raw_audio, raw_sr)
                        except Exception as exc:
                            logger.warning("mini scoring: cannot read %s: %s", uid, exc)

                    model_key   = an_models[0]
                    original_cm = {}
                    for uid, lbl, _ in sample_uids:
                        for entry in st.session_state.db_cm_scores.get(uid, []):
                            if entry.get("model") == model_key:
                                original_cm[uid] = entry["cm"]
                                break

                    manip_names = list(MANIPULATIONS.keys())
                    all_combos  = [
                        tuple(manip_names[i] for i in range(len(manip_names)) if mask & (1 << i))
                        for mask in range(1, 2 ** len(manip_names))
                    ]

                    n_total = len(sample_uids) * len(all_combos)
                    prog    = st.progress(0)
                    stat    = st.empty()
                    done    = 0
                    combo_results = {}

                    for combo in all_combos:
                        combo_key = " + ".join(combo)
                        combo_results[combo_key] = {}
                        steps = [
                            ManipulationStep(name=name, params=dict(MANIPULATIONS[name]["params"]))
                            for name in combo
                        ]

                        for uid, lbl, uid_idx in sample_uids:
                            before_cm = original_cm.get(uid)
                            if before_cm is None or uid not in audio_cache:
                                done += 1
                                prog.progress(done / n_total)
                                continue

                            raw_audio, raw_sr = audio_cache[uid]
                            utt = utterances.get(uid)

                            try:
                                manip_audio = apply_pipeline(raw_audio, raw_sr, steps, utterance=utt, uid_index=uid_idx)
                            except Exception as exc:
                                logger.warning("mini combo error %s / %s: %s", uid, combo_key, exc)
                                done += 1
                                prog.progress(done / n_total)
                                continue

                            try:
                                det_results = predict(
                                    manip_audio, raw_sr, model_keys=[model_key],
                                    threshold=0.5, max_duration_s=len(manip_audio) / raw_sr + 1,
                                )
                                after_cm = next((r.cm_score for r in det_results if not r.error), None)
                            except Exception as exc:
                                logger.warning("mini inference error %s / %s: %s", uid, combo_key, exc)
                                after_cm = None

                            if after_cm is not None:
                                combo_results[combo_key][uid] = {"before": before_cm, "after": after_cm, "label": lbl}

                            done += 1
                            prog.progress(done / n_total)

                        stat.caption(f"Combo {all_combos.index(combo) + 1}/{len(all_combos)}: {combo_key}")

                    st.session_state["mini_combo_results"] = combo_results
                    prog.progress(1.0)
                    stat.success(f"Done — {len(all_combos)} combinations scored on {len(sample_uids)} files.")

            combo_results = st.session_state.get("mini_combo_results", {})
            if combo_results:
                st.divider()

                summary_rows = []
                for combo_key, uid_dict in combo_results.items():
                    row = {"Pipeline": combo_key, "N steps": len(combo_key.split(" + "))}
                    for lbl, col_label in [
                        ("bonafide", "Δ bonafide"), ("synthetic", "Δ synthetic"), ("partial synthetic", "Δ partial"),
                    ]:
                        shifts = [v["after"] - v["before"] for v in uid_dict.values() if v["label"] == lbl]
                        row[col_label] = round(float(np.mean(shifts)), 3) if shifts else None
                    summary_rows.append(row)

                df_summary = (
                    pd.DataFrame(summary_rows)
                    .sort_values("Δ synthetic", ascending=True, na_position="last")
                    .reset_index(drop=True)
                )
                df_summary.index += 1

                st.markdown("#### All combinations ranked by Δ synthetic")
                st.caption(
                    "Sorted by mean CM shift on synthetic files after manipulation. "
                    "Most negative Δ synthetic = manipulation best fools the detector. "
                    "Δ bonafide close to 0 = bonafide files are not badly affected. "
                    "Use the filter below to narrow by number of steps."
                )

                max_steps    = df_summary["N steps"].max()
                filter_steps = st.multiselect(
                    "Show combinations with N steps",
                    options=list(range(1, max_steps + 1)),
                    default=list(range(1, max_steps + 1)),
                    key="mini_filter_steps",
                )
                df_show = df_summary[df_summary["N steps"].isin(filter_steps)] if filter_steps else df_summary

                st.dataframe(
                    df_show.style.applymap(colour_delta, subset=["Δ synthetic", "Δ partial"]),
                    use_container_width=True,
                )

                st.markdown("#### Top 5 combinations (most negative Δ synthetic)")
                st.dataframe(
                    df_summary.head(5)[["Pipeline", "N steps", "Δ bonafide", "Δ synthetic", "Δ partial"]],
                    use_container_width=True, hide_index=False,
                )

                st.markdown("#### CM score shift — top 3 combinations")
                st.caption("Solid = before manipulation. Dashed = after. Good manipulation: red dashed shifts left toward green.")

                top3_keys = df_summary.head(3)["Pipeline"].tolist()
                for combo_key in top3_keys:
                    uid_dict = combo_results.get(combo_key, {})
                    fig = combo_shift_kde(uid_dict, combo_key)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                st.divider()
                st.markdown(
                    "**Next step:** take the top-ranked combinations from the table above, "
                    "add those steps to the M-BM pipeline in the sidebar, and run the full M-BM Score Cache below."
                )

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Voice Deepfake Forensics")
    page = st.radio(
        "Page",
        ["Stage 0 - Database", "Stage 1 - Inspection", "Stage 2 - Analysis", "Stage 3 - Report"],
        label_visibility="collapsed",
    )
    st.divider()

if   page == "Stage 0 - Database":   render_database()
elif page == "Stage 1 - Inspection": render_inspection()
elif page == "Stage 2 - Analysis":   render_analysis()
elif page == "Stage 3 - Report":
    from report import render_report_section
    render_report_section()