"""Likelihood Ratio framework for forensic synthetic speech assessment.

The LR framework used here follows the forensic science standard:

    LR = P(E | Hp) / P(E | Hd)

where:
    E  — the observed countermeasure (CM) score for the target utterance
    Hp — prosecution hypothesis: the target utterance is synthetic
    Hd — defence hypothesis:     the target utterance is genuine

A KDE is fitted to background-model CM scores under each hypothesis.
The LR is then the ratio of the two KDE-evaluated likelihoods at the
target CM score.

Two background models are supported:
    BM  — unmanipulated background model scores
    MBM — manipulated background model scores (applied post-processing steps)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_EXP = np.log(np.finfo(np.float64).max)

# ---------------------------------------------------------------------------
# KDE fitting and evaluation
# ---------------------------------------------------------------------------
def _fit_kde(scores: list, bandwidth: str = "scott") -> Optional[gaussian_kde]:
    """Fit a Gaussian KDE to *scores*.

    Non-finite values (NaN, ±inf) are removed before fitting. Returns
    None if fewer than two finite samples remain, or if fitting fails.

    Parameters
    ----------
    scores:
        Raw CM score list. May contain non-finite values.
    bandwidth:
        KDE bandwidth selection method passed to scipy.stats.gaussian_kde.
        Supported values: 'scott' (default), 'silverman', or a
        positive float for a fixed normalised bandwidth.

    Returns
    -------
    gaussian_kde or None
    """
    arr = np.array(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        logger.debug("_fit_kde: only %d finite sample(s) — cannot fit KDE.", len(arr))
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


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class LRResult:
    """Output of compute_likelihood_ratio.

    All pdf and LR fields are None when the corresponding KDE could not
    be evaluated (e.g. insufficient samples).

    Fields
    ------
    target_cm   : The CM score that was evaluated.
    pdf_hd      : p(E | Hd) — likelihood under the defence hypothesis.
    pdf_hp_bm   : p(E | Hp) from the unmanipulated background model.
    pdf_hp_mbm  : p(E | Hp) from the manipulated background model.
    lr_bm       : LR against the unmanipulated background model.
    lr_mbm      : LR against the manipulated background model.
    log_lr_bm   : log(LR) against the unmanipulated background model.
    log_lr_mbm  : log(LR) against the manipulated background model.
    error       : Human-readable error message, or None on success.
    """
    target_cm:  float
    pdf_hd:     Optional[float] = None
    pdf_hp_bm:  Optional[float] = None
    pdf_hp_mbm: Optional[float] = None
    lr_bm:      Optional[float] = None
    lr_mbm:     Optional[float] = None
    log_lr_bm:  Optional[float] = None
    log_lr_mbm: Optional[float] = None
    error:      Optional[str]   = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def _compute_hp_fields(
    kde_hp:   gaussian_kde,
    log_hd:   float,
    x:        float,
) -> tuple:
    """Compute pdf, log-LR, and LR for one Hp KDE at point *x*.

    Parameters
    ----------
    kde_hp:
        KDE fitted to synthetic CM scores under Hp.
    log_hd:
        Log-likelihood under Hd at *x* (denominator of the LR).
    x:
        The target CM score.

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
    """Compute LR for *target_cm* against BM and M-BM score distributions.

    Parameters
    ----------
    target_cm:
        The CM score of the target utterance to evaluate.
    bm_scores:
        Dict with keys 'bonafide', 'synthetic', 'partial synthetic'
        mapping to lists of CM scores from the unmanipulated background model.
    mbm_scores:
        Same structure as *bm_scores* for the manipulated background model.

    Returns
    -------
    LRResult
        All LR fields are populated where sufficient data exists.
        LRResult.error is set (and other fields left None) if the Hd
        KDE cannot be fitted.
    """
    synth_bm  = bm_scores.get("synthetic",  []) + bm_scores.get("partial synthetic", [])
    synth_mbm = mbm_scores.get("synthetic", []) + mbm_scores.get("partial synthetic", [])

    bonafide_bm  = bm_scores.get("bonafide",  [])
    bonafide_mbm = mbm_scores.get("bonafide", [])

    kde_hd_bm  = _fit_kde(bonafide_bm)
    kde_hd_mbm = _fit_kde(bonafide_mbm)

    kde_hp_bm  = _fit_kde(synth_bm)
    kde_hp_mbm = _fit_kde(synth_mbm)

    if kde_hd_bm is None:
        return LRResult(
            target_cm = float(target_cm),
            error     = "Insufficient bonafide samples to model Hd.",
        )

    x = float(target_cm)
    log_hd_bm  = _log_pdf_kde(kde_hd_bm,  x)
    log_hd_mbm = _log_pdf_kde(kde_hd_mbm, x)

    result = LRResult(
        target_cm = x,
        pdf_hd    = float(np.exp(log_hd_bm)),
    )

    if kde_hp_bm is not None:
        result.pdf_hp_bm, result.log_lr_bm, result.lr_bm = (
            _compute_hp_fields(kde_hp_bm, log_hd_bm, x)
        )

    if kde_hp_mbm is not None:
        result.pdf_hp_mbm, result.log_lr_mbm, result.lr_mbm = (
            _compute_hp_fields(kde_hp_mbm, log_hd_mbm, x)
        )

    return result


# ---------------------------------------------------------------------------
# Posterior and decision threshold
# ---------------------------------------------------------------------------
def compute_posterior(lr: Optional[float], prior_prob: float) -> float:
    """Compute the posterior probability P(Hp | E) via Bayes' theorem.

        P(Hp | E) = LR · prior / (LR · prior + (1 - prior))

    Parameters
    ----------
    lr:
        Likelihood ratio. None and nan return nan.
        inf is handled correctly: P(Hp | E) = 1.0.
    prior_prob:
        Prior probability of the prosecution hypothesis. Must be in (0, 1).

    Returns
    -------
    float
        Posterior probability in [0, 1], or nan on invalid input.
    """
    if lr is None or (isinstance(lr, float) and np.isnan(lr)):
        return float("nan")

    prior = float(prior_prob)
    if not (0.0 < prior < 1.0):
        return float("nan")

    if np.isinf(lr):
        return 1.0

    numerator   = lr * prior
    denominator = numerator + (1.0 - prior)

    return float(numerator / denominator)


def compute_decision_threshold(
    cost_fp:    float,
    cost_fn:    float,
    prior_prob: float,
) -> float:
    """Compute the Bayes-optimal LR decision threshold.

    Declare synthetic when LR > τ, where:

        τ = (C_FP · (1 - prior)) / (C_FN · prior)

    Parameters
    ----------
    cost_fp:
        Cost of a false positive (declaring genuine speech synthetic).
    cost_fn:
        Cost of a false negative (declaring synthetic speech genuine).
    prior_prob:
        Prior probability that the target is synthetic. Must be in (0, 1).

    Returns
    -------
    float
        Decision threshold τ, or nan for invalid inputs.
        Returns 0.0 when cost_fn == 0 (any LR exceeds the threshold).
    """
    prior = float(prior_prob)
    if not (0.0 < prior < 1.0):
        return float("nan")

    c_fn = float(cost_fn)
    if c_fn == 0.0:
        return 0.0

    return float(cost_fp) * (1.0 - prior) / (c_fn * prior)


# ---------------------------------------------------------------------------
# Hypothesis shift evaluation
# ---------------------------------------------------------------------------
def evaluate_hypothesis_shift(
    lr_bm:      Optional[float],
    lr_mbm:     Optional[float],
    log_lr_bm:  Optional[float] = None,
    log_lr_mbm: Optional[float] = None,
) -> str:
    """Evaluate whether the manipulation pipeline strengthens evidence for Hp.

    Compares LR_BM (unmanipulated) with LR_MBM (manipulated). A higher
    LR_MBM indicates the manipulation pipeline increases the synthetic
    evidence — the hypothesis shift is "Supported".

    Resolution order
    ----------------
    1. If either LR is None or NaN -> Inconclusive (missing data).
    2. If both LRs are finite -> compare directly.
    3. If both log-LRs are finite -> compare in log space (avoids overflow).
    4. If one LR is infinite and the other finite -> direction is unambiguous.
    5. Both infinite -> Inconclusive (cannot rank).

    Parameters
    ----------
    lr_bm:
        LR against the unmanipulated background model.
    lr_mbm:
        LR against the manipulated background model.
    log_lr_bm:
        log(LR_BM), used as fallback when LRs overflow.
    log_lr_mbm:
        log(LR_MBM), used as fallback when LRs overflow.

    Returns
    -------
    str
        One of: 'Supported', 'Not Supported', 'Neutral',
        or 'Inconclusive (<reason>)'.
    """
    # Step 1: missing or NaN inputs.
    if lr_bm is None or lr_mbm is None:
        return "Inconclusive (Missing data)"
    if np.isnan(float(lr_bm)) or np.isnan(float(lr_mbm)):
        return "Inconclusive (Missing data)"

    bm_finite  = np.isfinite(lr_bm)
    mbm_finite = np.isfinite(lr_mbm)

    # Step 2: both finite — direct comparison.
    if bm_finite and mbm_finite:
        if lr_mbm > lr_bm:
            return "Supported"
        if lr_mbm < lr_bm:
            return "Not Supported"
        return "Neutral"

    # Step 3: at least one LR overflows — fall back to log space.
    if (
        log_lr_bm  is not None and np.isfinite(float(log_lr_bm))
        and log_lr_mbm is not None and np.isfinite(float(log_lr_mbm))
    ):
        if log_lr_mbm > log_lr_bm:
            return "Supported"
        if log_lr_mbm < log_lr_bm:
            return "Not Supported"
        return "Neutral"

    # Step 4: one side infinite, the other finite — direction is clear.
    if not bm_finite and mbm_finite:
        return "Not Supported"   # LR_BM = inf, LR_MBM finite -> MBM is weaker
    if bm_finite and not mbm_finite:
        return "Supported"       # LR_MBM = inf, LR_BM finite -> MBM is stronger

    # Step 5: both infinite and log-LRs unavailable or also infinite.
    return (
        "Inconclusive (Both LRs overflow — "
        "provide log-LR values or collect more overlapping scores)"
    )