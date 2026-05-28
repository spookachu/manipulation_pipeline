"""Likelihood Ratio framework for forensic synthetic speech assessment.

The LR framework used here follows the forensic science standard:

    LR = P(E | Hp) / P(E | Hd)

where:
    E  - the observed countermeasure (CM) score for the target utterance
    Hp - prosecution hypothesis: the target utterance is synthetic
    Hd - defence hypothesis:     the target utterance is bonafide

A KDE is fitted to background-model CM scores under each hypothesis.
The LR is then the ratio of the two KDE-evaluated likelihoods at the
target CM score.

Two background models are supported:
    BM  - unmanipulated background model scores
    MBM - manipulated background model scores (applied post-processing steps)

References
----------
ENFSI Guideline for Evaluative Reporting in Forensic Science (2015).
    European Network of Forensic Science Institutes.
Perlin, M. W. (2022). How to make better forensic decisions.
    PNAS, 119(38). doi:10.1073/pnas.2206567119.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_EXP: float = np.log(np.finfo(np.float64).max)
_CM_DECISION_THRESHOLD: float = 0.0 # uncalibrated

# ---------------------------------------------------------------------------
# KDE fitting and evaluation
# ---------------------------------------------------------------------------
def _fit_kde(scores: list, bandwidth: str = "scott") -> Optional[gaussian_kde]:
    """Fit a Gaussian KDE to scores.
    
    Parameters
    ----------
    scores:
        Raw CM score list. May contain non-finite values.
    bandwidth:
        KDE bandwidth selection method. 'scott' (default) 

    Returns
    -------
    gaussian_kde or None
    """
    arr = np.array(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        logger.debug("_fit_kde: only %d finite sample(s) - cannot fit KDE.", len(arr))
        return None

    try:
        return gaussian_kde(arr, bw_method=bandwidth)
    except Exception as exc:
        logger.warning("KDE fitting failed: %s", exc)
        return None


def _log_pdf_kde(kde: gaussian_kde, x: float) -> float:
    """Evaluate log p(x) for a fitted KDE at scalar *x*.

    Parameters
    ----------
    kde:
        A fitted scipy.stats.gaussian_kde object.
    x:
        The point at which to evaluate the log-density.

    Returns
    -------
    float
        Log probability density at *x*.
    """
    return float(kde.logpdf(np.array([x]))[0])


def _lr_from_log_lr(log_lr: float) -> float:
    """Convert a log-LR to a linear LR with overflow protection.

    Returns
    -------
    float
        inf  if log_lr overflows float64 upward,
        0.0  if log_lr overflows float64 downward,
        exp(log_lr) otherwise.
    """
    if log_lr > _MAX_EXP:
        return float("inf")
    if log_lr < -_MAX_EXP:
        return 0.0
    return float(np.exp(log_lr))


def _bhattacharyya(
    kde_a:    gaussian_kde,
    kde_b:    gaussian_kde,
    x_min:    float,
    x_max:    float,
    n_points: int = 500,
) -> float:
    """Bhattacharyya coefficient between two KDEs over [x_min, x_max].

    BC = integral( sqrt( p(x) * q(x) ) ) dx

    Values in [0, 1]: 0 = no overlap, 1 = identical distributions.
    Used as a distributional similarity measure for evasion analysis.

    Parameters
    ----------
    kde_a, kde_b : fitted KDE objects.
    x_min, x_max : integration range (set to data min/max + margin).
    n_points     : number of quadrature points.

    Returns
    -------
    float in [0, 1].
    """
    xs = np.linspace(x_min, x_max, n_points)
    dx = xs[1] - xs[0]
    pa = np.clip(kde_a(xs), 0.0, None)
    pb = np.clip(kde_b(xs), 0.0, None)
    return min(float(np.sum(np.sqrt(pa * pb)) * dx), 1.0)


def _mean_shift_se(a: np.ndarray, b: np.ndarray) -> float:
    """Standard error of the difference of two independent means.

    SE = sqrt( Var(a)/n_a + Var(b)/n_b )

    Used to express the synthetic mean CM shift relative to its
    sampling variability, avoiding arbitrary magnitude thresholds.
    """
    return float(np.sqrt(
        np.var(a, ddof=1) / len(a) +
        np.var(b, ddof=1) / len(b)
    ))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class LRResult:
    """Output of compute_likelihood_ratio.

    Two distinct LR values are computed and must be interpreted differently.

    LR_authenticity
        Hd: KDE of bonafide CM scores (unprocessed).
        Hp: KDE of synthetic CM scores (unprocessed).
        Answers: is the target synthetic or bonafide?
        ENFSI verbal scale applies here.

    LR_manipulation
        Hd: KDE of processed bonafide CM scores (M-BM).
        Hp: KDE of processed synthetic CM scores (M-BM).
        Answers: evidential weight conditioned on the manipulation
        hypothesis. Do not apply ENFSI verbal labels to this value.
        Used as an input to SyntheticEvasionResult only.

    Fields
    ------
    target_cm             : CM score evaluated.
    pdf_hd_bm             : p(E | Hd) from the unmanipulated BM.
    pdf_hd_mbm            : p(E | Hd) from the M-BM.
    pdf_hp_authenticity   : p(E | Hp) from the unmanipulated BM.
    pdf_hp_manipulation   : p(E | Hp) from the M-BM.
    lr_authenticity       : LR for authenticity question (BM).
    lr_manipulation       : LR conditioned on manipulation hypothesis.
    log_lr_authenticity   : log(LR_authenticity).
    log_lr_manipulation   : log(LR_manipulation).
    log_hd_shift          : log(pdf_hd_mbm) - log(pdf_hd_bm).
    log_hp_shift          : log(pdf_hp_mbm) - log(pdf_hp_bm).
    error                 : Error message, or None on success.
    """
    target_cm:            float
    pdf_hd_bm:            Optional[float] = None
    pdf_hd_mbm:           Optional[float] = None
    pdf_hp_authenticity:  Optional[float] = None
    pdf_hp_manipulation:  Optional[float] = None
    lr_authenticity:      Optional[float] = None
    lr_manipulation:      Optional[float] = None
    log_lr_authenticity:  Optional[float] = None
    log_lr_manipulation:  Optional[float] = None
    log_hd_shift:         Optional[float] = None
    log_hp_shift:         Optional[float] = None
    error:                Optional[str]   = None


@dataclass
class SyntheticEvasionResult:
    """Whether the M-BM pipeline shifts synthetic scores toward bonafide.

    mean_shift / mean_shift_se
        Mean CM shift of the synthetic distribution, expressed as a
        z-score (shift / SE) relative to sampling variability.
        SE is the standard error of the difference of two independent
        means. A large |z| indicates the shift exceeds what would be
        expected from sampling noise alone.

    overlap_increase
        Change in Bhattacharyya coefficient between bonafide and
        synthetic distributions. Reported without a fixed threshold
        since no standard SE formula exists for BC; direction and
        magnitude are reported for analyst interpretation.

    pct_evading_increase / pct_evading_se
        Absolute risk increase in the proportion of synthetic scores
        below the detection threshold, with SE = sqrt(p(1-p)/n) for
        each proportion. Allows the analyst to judge whether the
        increase is meaningful relative to sampling variability.

    Fields
    ------
    mean_bm_synthetic     : Mean CM of BM synthetic scores.
    mean_mbm_synthetic    : Mean CM of M-BM synthetic scores.
    mean_shift            : mean_mbm - mean_bm (negative = evasion).
    mean_shift_se         : Standard error of mean_shift.
    mean_bm_bonafide      : Mean CM of BM bonafide scores (reference).
    overlap_bm            : Bhattacharyya coefficient, BM bonafide/synthetic.
    overlap_mbm           : Bhattacharyya coefficient, BM bonafide/M-BM synthetic.
    overlap_increase      : overlap_mbm - overlap_bm.
    pct_evading_bm        : % BM synthetic scores below cm_threshold.
    pct_evading_mbm       : % M-BM synthetic scores below cm_threshold.
    pct_evading_increase  : pct_evading_mbm - pct_evading_bm.
    pct_evading_se        : SE of the absolute risk increase.
    evasion_verdict       : Evidence summary string.
    error                 : Set if computation failed.
    """
    mean_bm_synthetic:    Optional[float] = None
    mean_mbm_synthetic:   Optional[float] = None
    mean_shift:           Optional[float] = None
    mean_shift_se:        Optional[float] = None
    mean_bm_bonafide:     Optional[float] = None
    overlap_bm:           Optional[float] = None
    overlap_mbm:          Optional[float] = None
    overlap_increase:     Optional[float] = None
    pct_evading_bm:       Optional[float] = None
    pct_evading_mbm:      Optional[float] = None
    pct_evading_increase: Optional[float] = None
    pct_evading_se:       Optional[float] = None
    evasion_verdict:      str             = ""
    error:                Optional[str]   = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def _compute_hp_fields(
    kde_hp:   gaussian_kde,
    log_hd:   float,
    x:        float,
) -> tuple:
    """Compute pdf, log-LR, and LR for one Hp KDE at point *x*.

    Returns
    -------
    (pdf_hp, log_lr, lr) : tuple of float
    """
    log_hp = _log_pdf_kde(kde_hp, x)
    pdf_hp = float(np.exp(np.clip(log_hp, -_MAX_EXP, _MAX_EXP)))
    log_lr = log_hp - log_hd
    lr     = _lr_from_log_lr(log_lr)
    return pdf_hp, log_lr, lr


def compute_likelihood_ratio(
    target_cm:  float,
    bm_scores:  dict,
    mbm_scores: dict,
) -> LRResult:
    """Compute LR_authenticity and LR_manipulation for target_cm.

    Parameters
    ----------
    target_cm:
        CM score of the target utterance.
    bm_scores:
        Dict with keys 'bonafide', 'synthetic', 'partial synthetic'
        mapping to CM score lists from the unmanipulated BM.
    mbm_scores:
        Same structure for the manipulated background model.

    Returns
    -------
    LRResult
    """
    synth_bm     = bm_scores.get("synthetic",  []) + bm_scores.get("partial synthetic", [])
    synth_mbm    = mbm_scores.get("synthetic", []) + mbm_scores.get("partial synthetic", [])
    bonafide_bm  = bm_scores.get("bonafide", [])
    bonafide_mbm = mbm_scores.get("bonafide", [])

    kde_hd_bm  = _fit_kde(bonafide_bm)
    kde_hd_mbm = _fit_kde(bonafide_mbm)
    kde_hp_bm  = _fit_kde(synth_bm)
    kde_hp_mbm = _fit_kde(synth_mbm)

    if kde_hd_bm is None:
        return LRResult(
            target_cm = float(target_cm),
            error     = "Insufficient bonafide BM samples to model Hd.",
        )

    x = float(target_cm)

    log_hd_bm  = _log_pdf_kde(kde_hd_bm, x)
    log_hd_mbm = _log_pdf_kde(kde_hd_mbm, x) if kde_hd_mbm is not None else log_hd_bm

    pdf_hd_bm  = float(np.exp(np.clip(log_hd_bm,  -_MAX_EXP, _MAX_EXP)))
    pdf_hd_mbm = float(np.exp(np.clip(log_hd_mbm, -_MAX_EXP, _MAX_EXP)))
    log_hd_shift = log_hd_mbm - log_hd_bm

    result = LRResult(
        target_cm         = x,
        pdf_hd_bm         = pdf_hd_bm,
        pdf_hd_mbm        = pdf_hd_mbm,
        log_hd_shift      = log_hd_shift,
    )

    if kde_hp_bm is not None:
        result.pdf_hp_authenticity, result.log_lr_authenticity, result.lr_authenticity = (
            _compute_hp_fields(kde_hp_bm, log_hd_bm, x)
        )

    if kde_hp_mbm is not None:
        result.pdf_hp_manipulation, result.log_lr_manipulation, result.lr_manipulation = (
            _compute_hp_fields(kde_hp_mbm, log_hd_mbm, x)
        )

    if kde_hp_bm is not None and kde_hp_mbm is not None:
        result.log_hp_shift = _log_pdf_kde(kde_hp_mbm, x) - _log_pdf_kde(kde_hp_bm, x)

    return result


# ---------------------------------------------------------------------------
# Synthetic evasion analysis
# ---------------------------------------------------------------------------
def _evasion_verdict(
    mean_shift:       float,
    mean_shift_se:    float,
    overlap_increase: Optional[float],
    pct_evading_bm:   float,
    pct_evading_mbm:  float,
    n_bm:             int,
    n_mbm:            int,
) -> str:
    """Return a directional evasion verdict.

    Mean shift z-score (shift / SE) expressesthe shift relative to sampling variability. 
    """
    z = (mean_shift / mean_shift_se) if mean_shift_se > 0 else None

    if mean_shift < 0:
        strength = (
            "strong"    if z is not None and abs(z) >= 10
            else "moderate" if z is not None and abs(z) >= 3
            else "weak"
        )
        return f"Evasion supported ({strength} effect, z = {z:.1f})"
    else:
        return f"Evasion not supported (z = {z:.1f})" if z is not None else "Evasion not supported"
    

def evaluate_synthetic_evasion(
    bm_scores:    dict,
    mbm_scores:   dict,
    cm_threshold: float = _CM_DECISION_THRESHOLD,
) -> SyntheticEvasionResult:
    """Measure whether the M-BM pipeline shifts synthetic scores toward bonafide.

    Three complementary measures are computed; all expressed relative to
    sampling variability rather than fixed thresholds (ENFSI, 2015).

    Parameters
    ----------
    bm_scores:
        Dict with keys 'bonafide', 'synthetic', 'partial synthetic'.
    mbm_scores:
        Same structure for the manipulated background model.
    cm_threshold:
        Decision boundary for evading-percentage computation.
        Default _CM_DECISION_THRESHOLD (0.0) - the natural operating
        point of the logit CM convention.

    Returns
    -------
    SyntheticEvasionResult
    """
    synth_bm  = np.array(
        bm_scores.get("synthetic",  []) + bm_scores.get("partial synthetic", []),
        dtype=np.float64,
    )
    synth_mbm = np.array(
        mbm_scores.get("synthetic", []) + mbm_scores.get("partial synthetic", []),
        dtype=np.float64,
    )
    bonafide_bm = np.array(bm_scores.get("bonafide", []), dtype=np.float64)

    synth_bm    = synth_bm[np.isfinite(synth_bm)]
    synth_mbm   = synth_mbm[np.isfinite(synth_mbm)]
    bonafide_bm = bonafide_bm[np.isfinite(bonafide_bm)]

    if len(synth_bm) < 2:
        return SyntheticEvasionResult(error="Insufficient BM synthetic scores.")
    if len(synth_mbm) < 2:
        return SyntheticEvasionResult(error="Insufficient M-BM synthetic scores.")
    if len(bonafide_bm) < 2:
        return SyntheticEvasionResult(error="Insufficient BM bonafide scores.")

    mean_bm   = float(np.mean(synth_bm))
    mean_mbm  = float(np.mean(synth_mbm))
    shift     = mean_mbm - mean_bm
    shift_se  = _mean_shift_se(synth_bm, synth_mbm)

    pct_evading_bm  = float(np.mean(synth_bm  < cm_threshold) * 100.0)
    pct_evading_mbm = float(np.mean(synth_mbm < cm_threshold) * 100.0)
    pct_increase    = pct_evading_mbm - pct_evading_bm

    p_bm   = pct_evading_bm  / 100.0
    p_mbm  = pct_evading_mbm / 100.0
    ari_se = float(np.sqrt(
        p_bm  * (1.0 - p_bm)  / max(len(synth_bm),  1) +
        p_mbm * (1.0 - p_mbm) / max(len(synth_mbm), 1)
    )) * 100.0

    all_scores = np.concatenate([synth_bm, synth_mbm, bonafide_bm])
    x_min      = float(all_scores.min()) - 1.0
    x_max      = float(all_scores.max()) + 1.0

    kde_synth_bm  = _fit_kde(list(synth_bm))
    kde_synth_mbm = _fit_kde(list(synth_mbm))
    kde_bonafide  = _fit_kde(list(bonafide_bm))

    overlap_bm       = None
    overlap_mbm      = None
    overlap_increase = None

    if kde_synth_bm is not None and kde_bonafide is not None:
        overlap_bm = _bhattacharyya(kde_bonafide, kde_synth_bm,  x_min, x_max)
    if kde_synth_mbm is not None and kde_bonafide is not None:
        overlap_mbm = _bhattacharyya(kde_bonafide, kde_synth_mbm, x_min, x_max)
    if overlap_bm is not None and overlap_mbm is not None:
        overlap_increase = overlap_mbm - overlap_bm

    verdict = _evasion_verdict(
        mean_shift       = shift,
        mean_shift_se    = shift_se,
        overlap_increase = overlap_increase,
        pct_evading_bm   = pct_evading_bm,
        pct_evading_mbm  = pct_evading_mbm,
        n_bm             = len(synth_bm),
        n_mbm            = len(synth_mbm),
    )

    return SyntheticEvasionResult(
        mean_bm_synthetic    = mean_bm,
        mean_mbm_synthetic   = mean_mbm,
        mean_shift           = shift,
        mean_shift_se        = shift_se,
        mean_bm_bonafide     = float(np.mean(bonafide_bm)),
        overlap_bm           = overlap_bm,
        overlap_mbm          = overlap_mbm,
        overlap_increase     = overlap_increase,
        pct_evading_bm       = pct_evading_bm,
        pct_evading_mbm      = pct_evading_mbm,
        pct_evading_increase = pct_increase,
        pct_evading_se       = ari_se,
        evasion_verdict      = verdict,
    )


# ---------------------------------------------------------------------------
# ENFSI (2015) LR verbal scale
# ---------------------------------------------------------------------------
# Source: ENFSI Guideline for Evaluative Reporting in Forensic Science (2015)
LR_VERBAL_SCALE: list = [
    # (lower_inclusive, upper_exclusive, verbal_label, supported_hypothesis)
    (10_000, float("inf"), "very strong support",         "Hp"),
    (1_000,  10_000,       "strong support",              "Hp"),
    (100,    1_000,        "moderately strong support",   "Hp"),
    (10,     100,          "moderate support",            "Hp"),
    (1,      10,           "slight or limited support",   "Hp"),
    (0.1,    1,            "slight or limited support",   "Hd"),
    (0.01,   0.1,          "moderate support",            "Hd"),
    (0.001,  0.01,         "moderately strong support",   "Hd"),
    (0.0001, 0.001,        "strong support",              "Hd"),
    (0,      0.0001,       "very strong support",         "Hd"),
]


def verbal_lr_strength(lr: Optional[float]) -> str:
    """Return an ENFSI (2015) verbal strength string for a given LR.

    Parameters
    ----------
    lr:
        Likelihood ratio. None, nan, and negative values return
        'not computable'.

    Returns
    -------
    str
        e.g. 'moderately strong support for Hp (synthetic)'
    """
    if lr is None or (isinstance(lr, float) and (math.isnan(lr) or lr < 0)):
        return "not computable"
    if math.isinf(lr) and lr > 0:
        return "very strong support for Hp (synthetic)"
    if lr == 0.0:
        return "very strong support for Hd (bonafide)"
    for lo, hi, label, hyp in LR_VERBAL_SCALE:
        if lo <= lr < hi:
            hyp_label = "Hp (synthetic)" if hyp == "Hp" else "Hd (bonafide)"
            return f"{label} for {hyp_label}"
    return "not computable"


# ---------------------------------------------------------------------------
# Perlin (2022) confidence-criterion scale
# ---------------------------------------------------------------------------
# Source: Perlin, M.W. (2022). How to make better forensic decisions.
# PNAS, 119(38). doi:10.1073/pnas.2206567119.
CONFIDENCE_LEVELS: dict = {
    1: "C1 - Highly confident: SYNTHETIC",
    2: "C2 - Confident: SYNTHETIC",
    3: "C3 - Leaning synthetic",
    4: "C4 - Inconclusive",
    5: "C5 - Leaning bonafide",
    6: "C6 - Confident: BONAFIDE",
    7: "C7 - Highly confident: BONAFIDE",
}

CONFIDENCE_DECISION_REGION: dict = {
    1: "Synthetic identified (high selectivity)",
    2: "Synthetic identified",
    3: "Inconclusive - leaning synthetic",
    4: "Inconclusive",
    5: "Inconclusive - leaning bonafide",
    6: "Bonafide (synthetic eliminated)",
    7: "Bonafide (synthetic eliminated, high selectivity)",
}