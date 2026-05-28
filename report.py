"""Forensic report generation for the Voice Deepfake Forensic Pipeline.

Collects session state from all three stages and renders a plain-text
report following the ENFSI Guideline for Evaluative Reporting (2015).
"""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PipelineStepRecord:
    """Manipulation steps applied to M-BM."""
    name:   str
    params: Dict


@dataclass
class DetectionRecord:
    """The countermeasure model loaded and its performance."""
    model_key:    str
    model_display: str
    cm_score:     Optional[float]
    error:        Optional[str]


@dataclass
class EvasionRecord:
    """Synthetic evasion analysis results from Stage 2."""
    mean_shift:           Optional[float] = None
    mean_shift_se:        Optional[float] = None
    overlap_bm:           Optional[float] = None
    overlap_mbm:          Optional[float] = None
    overlap_increase:     Optional[float] = None
    pct_evading_bm:       Optional[float] = None
    pct_evading_mbm:      Optional[float] = None
    pct_evading_increase: Optional[float] = None
    pct_evading_se:       Optional[float] = None
    evasion_verdict:      str             = ""
    error:                Optional[str]   = None


@dataclass
class LRRecord:
    """
    Record of the LR calculations for sample authenticity 
    and manipulation influence.
    """
    model_key:            str
    model_display:        str
    target_cm:            Optional[float]
    lr_authenticity:      Optional[float]
    log_lr_authenticity:  Optional[float]
    lr_manipulation:      Optional[float]
    log_lr_manipulation:  Optional[float]
    evasion:              Optional[EvasionRecord] = None


@dataclass
class AcousticRecord:
    """Record of the acoustic features extracted via Parselmouth."""
    duration_s:  Optional[float]
    f0_mean:     Optional[float]
    f0_sd:       Optional[float]
    f0_range:    Optional[float]
    jitter_pct:  Optional[float]
    shimmer_pct: Optional[float]


@dataclass
class AuditoryRecord:
    """Record of the auditory features as noted manually after critical listening."""
    voice_quality:       List[str] = field(default_factory=list)
    speaking_rate:       str       = "(not assessed)"
    prosody:             List[str] = field(default_factory=list)
    pitch_level:         str       = "(not assessed)"
    variability_factors: List[str] = field(default_factory=list)
    transmission:        List[str] = field(default_factory=list)
    idiosyncratic:       List[str] = field(default_factory=list)
    notes:               str       = ""


@dataclass
class SubjectiveIndicatorRecord:
    """Record of perceptual and quality markers noted after critical listening."""
    key:   str
    label: str
    value: str


@dataclass
class LimeComponentRecord:
    """Record of CoughLIME run information."""
    rank:            int
    component_index: int
    start_s:         float
    end_s:           float
    weight:          float
    direction:       str


@dataclass
class SynthContentRecord:
    """Record of evaluations around target sample authenticity."""
    perceptual_indicators: List[SubjectiveIndicatorRecord] = field(default_factory=list)
    subjective_eval:       List[SubjectiveIndicatorRecord] = field(default_factory=list)
    annotation_notes:      str                             = ""
    flags:                 List[float]                     = field(default_factory=list)
    time_s:                float                           = 0
    lime_model_key:        Optional[str]                   = None
    lime_r2:               Optional[float]                 = None
    lime_components:       List[LimeComponentRecord]       = field(default_factory=list)


@dataclass
class ReportData:
    """Complete record data to be included in report."""
    # Provenance
    generated_at:       str            = ""
    analyst_name:       str            = ""
    analyst_confidence: Optional[int]  = None
    analyst_notes:      str            = ""
    analyst_motivation: str            = ""
    # Target
    target_filename:    str            = ""
    target_duration_s:  Optional[float] = None
    target_sample_rate: Optional[int]   = None
    # Background model
    dataset_name:       str = ""
    dataset_key:        str = ""
    bm_n_bonafide:      int = 0
    bm_n_synthetic:     int = 0
    bm_cllr:            Dict[str, float] = field(default_factory=dict)
    # Stage 1
    acoustic:           Optional[AcousticRecord]     = None
    auditory:           Optional[AuditoryRecord]     = None
    synth_content:      Optional[SynthContentRecord] = None
    detection:          List[DetectionRecord]         = field(default_factory=list)
    # Stage 2
    mbm_pipeline:       List[PipelineStepRecord]     = field(default_factory=list)
    lr_results:         List[LRRecord]               = field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema mirrors
# ---------------------------------------------------------------------------
_ANNOTATION_INDICATORS: List[tuple] = [
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

_EVAL_QUESTIONS: List[tuple] = [
    ("voice_mechanical",   "The voice sounds mechanical."),
    ("voice_expressive",   "The voice sounds expressive."),
    ("voice_intelligible", "The voice is easy to understand."),
    ("audio_clean",        "The audio sounds clean."),
    ("voice_calm",         "The voice sounds calm."),
    ("eval_confident",     "I am confident in my evaluation."),
]

_LIKERT_LABELS: Dict[int, str] = {
    -2: "Very sure not present",
    -1: "Probably not present",
     0: "Unsure",
     1: "Probably present",
     2: "Very sure present",
}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect_report_data() -> ReportData:
    """
    Read current Streamlit session state and return a ReportData snapshot.
    """
    import streamlit as st
    from background_model import evaluate_synthetic_evasion
    from detection import MODEL_SPECS
    from datasets import DATASET_REGISTRY

    ss  = st.session_state
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = ReportData(generated_at=now)

    # Analyst inputs
    data.analyst_name       = ss.get("_report_analyst_name",       "")
    data.analyst_confidence = ss.get("_report_analyst_confidence", None)
    data.analyst_notes      = ss.get("analyst_notes",              "")
    data.analyst_motivation = ss.get("_report_analyst_motivation", "")

    #  Target audio
    data.target_filename    = ss.get("insp_name", "")
    data.target_sample_rate = ss.get("insp_sr")
    audio: Optional[np.ndarray] = ss.get("insp_audio")
    if audio is not None and data.target_sample_rate:
        data.target_duration_s = len(audio) / data.target_sample_rate

    #  Dataset / BM
    selected_key = ss.get("selected_dataset")
    if selected_key:
        spec = next((s for s in DATASET_REGISTRY if s.key == selected_key), None)
        if spec:
            data.dataset_name = spec.display_name
            data.dataset_key  = spec.key

        db_scores: dict = ss.get("db_cm_scores", {})
        bonafide_uids  = set()
        synthetic_uids = set()
        for uid, entries in db_scores.items():
            for entry in entries:
                if entry.get("gt") == "bonafide":
                    bonafide_uids.add(uid)
                elif entry.get("gt") in ("synthetic", "partial synthetic"):
                    synthetic_uids.add(uid)
        data.bm_n_bonafide  = len(bonafide_uids)
        data.bm_n_synthetic = len(synthetic_uids)

    data.bm_cllr = ss.get("bm_cllr", {})

    # Acoustic features
    raw_acoustic = ss.get("acoust_target_feats")
    if raw_acoustic is not None and raw_acoustic.error is None:
        p  = raw_acoustic.pitch
        vq = raw_acoustic.voice_quality
        data.acoustic = AcousticRecord(
            duration_s  = raw_acoustic.duration_s,
            f0_mean     = p.mean,
            f0_sd       = p.sd,
            f0_range    = p.range,
            jitter_pct  = vq.jitter_pct,
            shimmer_pct = vq.shimmer_pct,
        )

    # Auditory analysis
    raw_aud = ss.get("insp_auditory", {})
    _not_assessed = {"(not assessed)", ""}
    if any(
        (isinstance(v, list) and v)
        or (isinstance(v, str) and v not in _not_assessed)
        for v in raw_aud.values()
    ):
        data.auditory = AuditoryRecord(
            voice_quality       = raw_aud.get("voice_quality",       []),
            speaking_rate       = raw_aud.get("speaking_rate",       "(not assessed)"),
            prosody             = raw_aud.get("prosody",             []),
            pitch_level         = raw_aud.get("pitch_level",         "(not assessed)"),
            variability_factors = raw_aud.get("variability_factors", []),
            transmission        = raw_aud.get("transmission",        []),
            idiosyncratic       = raw_aud.get("idiosyncratic",       []),
            notes               = raw_aud.get("notes",               ""),
        )

    # Synthetic content analysis
    raw_ann   = ss.get("insp_det_annotation", {})
    raw_flags = ss.get("insp_flags", [])

    perceptual = [
        SubjectiveIndicatorRecord(key=k, label=l, value=raw_ann.get(k, 0))
        for k, l in _ANNOTATION_INDICATORS
    ]
    subjective = [
        SubjectiveIndicatorRecord(key=k, label=l, value=raw_ann.get(k, "Unsure"))
        for k, l in _EVAL_QUESTIONS
    ]
    flags = [SynthContentRecord(time_s=float(f["time"])) for f in raw_flags if "time" in f]

    lime_model_key: Optional[str]               = None
    lime_r2:        Optional[float]             = None
    lime_components: List[LimeComponentRecord]  = []
    lime_cache = ss.get("insp_lime_result")
    if lime_cache:
        lime_model_key = lime_cache.get("model_key")
        result         = lime_cache.get("result", {})
        lime_r2        = result.get("score")
        decomp         = lime_cache.get("decomp")
        local_exp      = result.get("local_exp", [])
        if decomp is not None and local_exp:
            borders    = [0] + decomp.indices_components + [len(decomp.audio)]
            sorted_exp = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)
            for rank, (idx, weight) in enumerate(sorted_exp, start=1):
                lime_components.append(LimeComponentRecord(
                    rank            = rank,
                    component_index = idx,
                    start_s         = round(borders[idx]     / decomp.sr, 3),
                    end_s           = round(borders[idx + 1] / decomp.sr, 3),
                    weight          = round(float(weight), 4),
                    direction       = "-> synthetic" if weight > 0 else "-> bonafide",
                ))

    if (
        any(r.value != 0 for r in perceptual)
        or any(r.value != "Unsure" for r in subjective)
        or flags or lime_components
        or raw_ann.get("notes", "").strip()
    ):
        data.synth_content = SynthContentRecord(
            perceptual_indicators = perceptual,
            subjective_eval       = subjective,
            annotation_notes      = raw_ann.get("notes", ""),
            flags                 = flags,
            lime_model_key        = lime_model_key,
            lime_r2               = lime_r2,
            lime_components       = lime_components,
        )

    # Detection results
    model_display = {s.key: s.display_name for s in MODEL_SPECS}
    insp_det: dict = ss.get("insp_det", {})
    for key, det in insp_det.items():
        data.detection.append(DetectionRecord(
            model_key     = key,
            model_display = model_display.get(key, key),
            cm_score      = det.cm_score if not det.error else None,
            error         = det.error,
        ))

    # M-BM pipeline
    for step in ss.get("an_mbm_pipeline", []):
        data.mbm_pipeline.append(PipelineStepRecord(
            name   = step.name,
            params = dict(step.params) if hasattr(step, "params") else {},
        ))

    #  LR results + evasion
    from background_model import evaluate_synthetic_evasion

    an_lr:     dict = ss.get("an_lr_results",  {})
    bm_scores_cache:  dict = ss.get("an_bm_scores",  {})
    mbm_scores_cache: dict = ss.get("an_mbm_scores", {})

    for key, lr_result in an_lr.items():
        det       = insp_det.get(key)
        target_cm = det.cm_score if (det and not det.error) else None

        evasion_record: Optional[EvasionRecord] = None
        bm_s  = bm_scores_cache.get(key)
        mbm_s = mbm_scores_cache.get(key)
        if bm_s is not None and mbm_s is not None:
            try:
                ev = evaluate_synthetic_evasion(bm_s, mbm_s)
                evasion_record = EvasionRecord(
                    mean_shift           = ev.mean_shift,
                    mean_shift_se        = ev.mean_shift_se,
                    overlap_bm           = ev.overlap_bm,
                    overlap_mbm          = ev.overlap_mbm,
                    overlap_increase     = ev.overlap_increase,
                    pct_evading_bm       = ev.pct_evading_bm,
                    pct_evading_mbm      = ev.pct_evading_mbm,
                    pct_evading_increase = ev.pct_evading_increase,
                    pct_evading_se       = ev.pct_evading_se,
                    evasion_verdict      = ev.evasion_verdict,
                    error                = ev.error,
                )
            except Exception as exc:
                evasion_record = EvasionRecord(error=str(exc))

        data.lr_results.append(LRRecord(
            model_key           = key,
            model_display       = model_display.get(key, key),
            target_cm           = target_cm,
            lr_authenticity     = lr_result.lr_authenticity,
            log_lr_authenticity = lr_result.log_lr_authenticity,
            lr_manipulation     = lr_result.lr_manipulation,
            log_lr_manipulation = lr_result.log_lr_manipulation,
            evasion             = evasion_record,
        ))

    return data


# ---------------------------------------------------------------------------
# Text rendering helpers
# ---------------------------------------------------------------------------
_SEP_MAJOR = "=" * 100
_INDENT    = "  "

def _fmt_float(value: Optional[float], decimals: int = 4,
    fallback: str = "N/A") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.{decimals}f}"


def _fmt_sci(value: Optional[float], fallback: str = "N/A") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.4e}"


def _fmt_list(items: List[str], fallback: str = "(none)") -> str:
    return ", ".join(items) if items else fallback


def _wrap(text: str, width: int = 100, indent: str = "") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


# ---------------------------------------------------------------------------
# Report text builder
# ---------------------------------------------------------------------------
def build_report_text(data: ReportData) -> str:
    """Convert a ReportData instance into a plain-text report."""
    from background_model import verbal_lr_strength, CONFIDENCE_LEVELS

    lines: List[str] = []

    def section(title: str) -> None:
        lines.extend(["", _SEP_MAJOR, f"  {title}", _SEP_MAJOR])

    def subsection(title: str) -> None:
        lines.extend(["", f"--- {title} " + "-" * max(0, 66 - len(title))])

    def row(label: str, value: str) -> None:
        lines.append(f"{_INDENT}{label:<36}{value}")

    #  Header
    lines.extend([_SEP_MAJOR, "  SYNTHETIC SPEECH FORENSIC ANALYSIS REPORT", _SEP_MAJOR])
    row("Generated at (UTC):", data.generated_at)
    row("Analyst:",            data.analyst_name or "(not specified)")
    row("Target file:",        data.target_filename or "-")
    if data.target_duration_s  is not None:
        row("Duration:", f"{data.target_duration_s:.2f} s")
    if data.target_sample_rate is not None:
        row("Sample rate:", f"{data.target_sample_rate} Hz")

    #  Disclaimer
    section("DISCLAIMER")
    lines.append(_wrap(
        "This report is produced for research and investigative support.",
        indent=_INDENT,
    ))

    #  1. Background model
    section("1. BACKGROUND MODEL")
    row("Dataset:",                  data.dataset_name or "-")
    row("BM bonafide utterances:",   str(data.bm_n_bonafide))
    row("BM synthetic utterances:",  str(data.bm_n_synthetic))
    if data.bm_cllr:
        lines.append("")
        for model_key, cllr_val in data.bm_cllr.items():
            row(f"  Cllr ({model_key}):", f"{cllr_val:.4f}")
        lines.append(_wrap(
            "Cllr (log-likelihood-ratio cost): 0 = perfect calibration, "
            "1 = no better than chance, >1 = miscalibrated. ",
            indent=_INDENT,
        ))

    #  2. Acoustic analysis
    section("2. ACOUSTIC ANALYSIS")
    if data.acoustic is None:
        lines.append(f"{_INDENT}No acoustic features extracted.")
    else:
        ac = data.acoustic
        subsection("Fundamental Frequency (F0)")
        row("Mean F0:", f"{_fmt_float(ac.f0_mean, 1)} Hz")
        row("SD F0:",   f"{_fmt_float(ac.f0_sd,   1)} Hz")
        row("Range F0:",f"{_fmt_float(ac.f0_range, 1)} Hz")
        subsection("Voice Quality")
        row("Jitter (local) %:",  _fmt_float(ac.jitter_pct,  3))
        row("Shimmer (local) %:", _fmt_float(ac.shimmer_pct, 3))
        lines.append(_wrap(
            "Calculated using Parselmouth, a Python Praat implementation.",
            indent=_INDENT,
        ))

    #  3. Auditory analysis
    section("3. AUDITORY ANALYSIS")
    if data.auditory is None:
        lines.append(f"{_INDENT}No auditory observations recorded.")
    else:
        aud = data.auditory
        row("Voice quality:",          _fmt_list(aud.voice_quality))
        row("Speaking rate:",          aud.speaking_rate)
        row("Prosodic features:",      _fmt_list(aud.prosody))
        row("Pitch level:",            aud.pitch_level)
        row("Variability factors:",    _fmt_list(aud.variability_factors))
        row("Transmission/recording:", _fmt_list(aud.transmission))
        row("Idiosyncratic features:", _fmt_list(aud.idiosyncratic))
        if aud.notes.strip():
            subsection("Free-text Observations")
            for para in aud.notes.strip().splitlines():
                lines.append(_wrap(para, indent=_INDENT) if para.strip() else "")

    #  4. Synthetic content analysis
    section("4. SYNTHETIC CONTENT ANALYSIS (STAGE 1)")
    if data.synth_content is None:
        lines.append(f"{_INDENT}No annotations recorded.")
    else:
        sc = data.synth_content

        subsection("Perceptual Indicators")
        informative = [r for r in sc.perceptual_indicators if r.value != 0]
        if informative:
            for rec in informative:
                row(f"  {rec.label}:", _LIKERT_LABELS.get(rec.value, str(rec.value)))
        else:
            lines.append(f"{_INDENT}All indicators rated Unsure.")

        subsection("Subjective Evaluation")
        non_default = [r for r in sc.subjective_eval if r.value != "Unsure"]
        if non_default:
            for rec in non_default:
                row(f"  {rec.label}:", rec.value)
        else:
            lines.append(f"{_INDENT}All items rated Unsure.")

        subsection("Suspicious Timestamps")
        if sc.flags:
            for i, flag in enumerate(sc.flags, start=1):
                row(f"  Flag {i}:", f"{flag.time_s:.3f} s")
        else:
            lines.append(f"{_INDENT}No timestamps flagged.")

        subsection("CoughLIME Explainability")
        if sc.lime_model_key is None:
            lines.append(f"{_INDENT}LIME was not run.")
        else:
            from detection import MODEL_SPECS
            display = next(
                (s.display_name for s in MODEL_SPECS if s.key == sc.lime_model_key),
                sc.lime_model_key,
            )
            row("Model:", display)
            row("Local fidelity R²:", _fmt_float(sc.lime_r2, 3))
            if sc.lime_r2 is not None and sc.lime_r2 < 0.5:
                lines.append(f"{_INDENT}Note: R² < 0.5 - explanation may be unreliable.")
            if sc.lime_components:
                lines.append(
                    f"{_INDENT}{'Rank':<6}{'Comp':<6}{'Start (s)':<12}"
                    f"{'End (s)':<12}{'Weight':<10}Direction"
                )
                lines.append(f"{_INDENT}" + "-" * 56)
                for comp in sc.lime_components:
                    lines.append(
                        f"{_INDENT}{comp.rank:<6}{comp.component_index:<6}"
                        f"{comp.start_s:<12.3f}{comp.end_s:<12.3f}"
                        f"{comp.weight:<+10.4f}{comp.direction}"
                    )

        if sc.annotation_notes.strip():
            subsection("Annotation Notes")
            for para in sc.annotation_notes.strip().splitlines():
                lines.append(_wrap(para, indent=_INDENT) if para.strip() else "")

    #  5. Detection scores
    section("5. DEEPFAKE DETECTION SCORES")
    lines.append(_wrap(
        "CM score = logit[P(synthetic)] - logit[P(bonafide)]. "
        "Higher = stronger evidence of synthetic speech. "
        "CM = 0 is the equal-probability boundary.",
        indent=_INDENT,
    ))
    lines.append("")
    if not data.detection:
        lines.append(f"{_INDENT}No detection results available.")
    else:
        for det in data.detection:
            subsection(det.model_display)
            if det.error:
                row("Status:", f"ERROR - {det.error}")
            else:
                row("CM score:", _fmt_float(det.cm_score, 4))

    #  6. Manipulation pipeline
    section("6. MANIPULATION PIPELINE (M-BM)")
    if not data.mbm_pipeline:
        lines.append(f"{_INDENT}No manipulation steps applied.")
    else:
        for i, step in enumerate(data.mbm_pipeline, start=1):
            lines.append(f"{_INDENT}{i}. {step.name}")
            for k, v in step.params.items():
                lines.append(f"{_INDENT}   {k}: {v}")

    #  7. LR assessment
    section("7. LIKELIHOOD RATIO ASSESSMENT")
    lines.append(_wrap(
        "LR = P(E|Hp) / P(E|Hd). The LR is prior-free and expresses "
        "evidential weight only. Follows the ENFSI (2015) guideline.",
        indent=_INDENT,
    ))

    if not data.lr_results:
        lines.append(f"\n{_INDENT}No LR results available.")
    else:
        for rec in data.lr_results:
            subsection(f"Model: {rec.model_display}")
            row("Target CM:", _fmt_float(rec.target_cm, 4))

            # Authenticity
            lines.append(f"\n{_INDENT}[ Authenticity: is the target synthetic? ]")
            row("  LR (authenticity):",     _fmt_sci(rec.lr_authenticity))
            row("  log LR:",               _fmt_float(rec.log_lr_authenticity, 4))
            row("  Evidence strength:",    verbal_lr_strength(rec.lr_authenticity))

            cllr = data.bm_cllr.get(rec.model_key)
            if cllr is not None:
                row("  System Cllr:",      f"{cllr:.4f}")

            # Evasion / manipulation
            lines.append(f"\n{_INDENT}[ Processing consistency: does the pipeline evade the detector? ]")
            row("  LR (manipulation):",    _fmt_sci(rec.lr_manipulation))
            row("  log LR:",               _fmt_float(rec.log_lr_manipulation, 4))

            if rec.evasion is None or rec.evasion.error:
                err = rec.evasion.error if rec.evasion else "not computed"
                lines.append(f"{_INDENT}  Synthetic evasion: {err}")
            else:
                ev = rec.evasion
                row("  Mean CM shift (synthetic):",
                    f"{_fmt_float(ev.mean_shift, 3)}"
                    + (f"  (SE {_fmt_float(ev.mean_shift_se, 3)})" if ev.mean_shift_se else ""))
                row("  Overlap increase (BC):",
                    _fmt_float(ev.overlap_increase, 4))
                row("  Evading BM -> M-BM (%):",
                    f"{_fmt_float(ev.pct_evading_bm, 1)}% -> "
                    f"{_fmt_float(ev.pct_evading_mbm, 1)}%"
                    + (f"  (SE ≈ {_fmt_float(ev.pct_evading_se, 1)}%)" if ev.pct_evading_se else ""))
                row("  Verdict:", ev.evasion_verdict)

    #  8. Analyst opionion
    section("8. ANALYST NOTES")
    conf = data.analyst_confidence
    if conf is not None:
        row("Confidence:", CONFIDENCE_LEVELS.get(conf, str(conf)))
    else:
        row("Analyst confidence:", "(not recorded)")

    if data.analyst_notes.strip():
        for para in data.analyst_notes.strip().splitlines():
            lines.append(_wrap(para, indent=_INDENT) if para.strip() else "")
    else:
        lines.append(f"{_INDENT}(none)")
        
    row("FINAL MOTIVATION", "")
    if data.analyst_motivation.strip():
        for para in data.analyst_motivation.strip().splitlines():
            lines.append(_wrap(para, indent=_INDENT) if para.strip() else "")
    else:
        lines.append(f"{_INDENT}(none)")

    lines.append("")
    lines.append(_wrap(
        "The analyst verdict is an expert opinion based on the totality of evidence.",
        indent=_INDENT,
    ))

    lines.extend(["", _SEP_MAJOR, "  END OF REPORT", _SEP_MAJOR, ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit entry point
# ---------------------------------------------------------------------------
def render_report_section() -> None:
    """Render the Stage 3 report section."""
    import streamlit as st
    from background_model import CONFIDENCE_LEVELS

    st.markdown("## Forensic Report")
    st.caption(
        "Compiles all data from into a plain-text report. "
        "Complete at least Stage 1 detection before generating."
    )

    #  Analyst name
    analyst_name = st.text_input(
        "Analyst name",
        value=st.session_state.get("_report_analyst_name", ""),
        placeholder="e.g. Dr. J. Smith",
        key="report_analyst_name_input",
    )
    st.session_state["_report_analyst_name"] = analyst_name

    #  Analyst confidence
    st.markdown("**Analyst confidence**")
    st.caption(
        "C1-C2: high selectivity for synthetic. "
        "C3-C5: inconclusive. "
        "C6-C7: high selectivity for genuine. "
        "Expert opinion only - does not enter any calculation."
    )
    confidence_options = list(CONFIDENCE_LEVELS.keys())
    confidence_labels  = list(CONFIDENCE_LEVELS.values())
    current_conf       = st.session_state.get("_report_analyst_confidence", 4)
    if current_conf not in confidence_options:
        current_conf = 4

    selected_conf_label = st.radio(
        "Analyst confidence",
        options=confidence_labels,
        index=confidence_options.index(current_conf),
        horizontal=False,
        label_visibility="collapsed",
        key="report_confidence_radio",
    )
    selected_conf = confidence_options[confidence_labels.index(selected_conf_label)]
    st.session_state["_report_analyst_confidence"] = selected_conf

    # Free final reasoning 
    analyst_motivation = st.text_area(
            "Final Motivation",
            value=st.session_state.get("_report_analyst_motivation", ""),
            height=80,
            placeholder=("Write down all aditional motivation and reasoning for your decision."),
            key="ann_motivation",
    )
    st.session_state["_report_analyst_motivation"] = analyst_motivation
        
    #  Generate
    if st.button("Generate report", key="report_generate"):
        with st.spinner("Collecting session data…"):
            report_data = collect_report_data()
        with st.spinner("Rendering…"):
            report_text = build_report_text(report_data)
        st.session_state["_report_text"] = report_text
        st.session_state["_report_data"] = report_data

    report_text: Optional[str] = st.session_state.get("_report_text")
    if report_text:
        target_name = st.session_state.get("_report_data", ReportData()).target_filename
        stem      = target_name.rsplit(".", 1)[0].replace(" ", "_") if target_name else "target"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        st.download_button(
            label     = "Download report (.txt)",
            data      = report_text,
            file_name = f"forensic_report_{stem}_{timestamp}.txt",
            mime      = "text/plain",
            key       = "report_download",
        )
       