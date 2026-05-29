"""visualizations.py - centralised plotting and UI colour constants for the Voice Deepfake Forensic Pipeline.

All figures follow a shared IEEE-style aesthetic:
  - Times New Roman / serif body font
  - 8 pt axis labels, 7 pt tick labels, 8 pt legend
  - No top/right spines
  - Tight layout with consistent pad=0.3

UI helpers
----------
    cat_color(name)         -> hex colour string for a manipulation step name
    pipeline_badge(steps)   -> HTML badge string for a pipeline

Plot functions
--------------
    kde_plot(scores_by_label, suspect_cm, title, threshold, ax)
    bm_kde_sidebyside(bm_by, mbm_by, target_cm, pipe_hash)
    bm_kde_overlay(bm_by, mbm_by, target_cm)
    per_dataset_kde(selected_specs, per_ds_nat, per_ds_syn)
    combo_shift_kde(uid_dict, combo_key)
    neighbourhood_composition(nb_bm, nb_mbm, sd_w, target_cm)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Shared aesthetic constants
# ---------------------------------------------------------------------------
LABEL_COLORS: Dict[str, str] = {
    "bonafide":          "#b0f75e",
    "synthetic":         "#b51212",
    "partial synthetic": "#f39c12",
}

CAT_COLORS: Dict[str, str] = {
    "Signal Degradation":     "#ff9e17",
    "Environment Simulation": "#5d17ff",
}

_FONT_FAMILY = "serif"
_AX_LABEL_FS = 8
_TICK_FS     = 7
_LEGEND_FS   = 8
_TITLE_FS    = 10
_SPINE_COLOR = "#333333"
_LEGEND_FACE = "#f8f8f8"

_TARGET_COLOR = "#111111"
_TARGET_LW    = 1.5
_TARGET_LS    = "-"

_THRESH_COLOR = "#d4a017"
_THRESH_LW    = 1.2
_THRESH_LS    = "--"

plt.rcParams["font.family"] = _FONT_FAMILY


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def cat_color(name: str) -> str:
    """Return the badge colour for a manipulation name."""
    from manipulations import MANIPULATIONS
    return CAT_COLORS.get(MANIPULATIONS.get(name, {}).get("category", ""), "#888")


def pipeline_badge(steps: list) -> str:
    """Return an HTML badge string for a list of ManipulationStep objects."""
    if not steps:
        return "<em>none</em>"
    return " &rarr; ".join(
        f'<span style="background:{cat_color(s.name)};color:#fff;border-radius:3px;'
        f'padding:1px 5px;font-size:0.78em;">{s.name}</span>'
        for s in steps
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _sign(x: int) -> str:
    return f"+{x}" if x >= 0 else str(x)


def _ds_color(i: int) -> str:
    """Return a distinct colour for dataset index i, sampled from tab10."""
    cmap = matplotlib.colormaps["tab10"]
    r, g, b, _ = cmap(i % cmap.N)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _new_fig(figsize: Tuple[float, float]) -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=figsize)


def _apply_style(ax: plt.Axes) -> None:
    """Apply shared style to an Axes."""
    ax.tick_params(labelsize=_TICK_FS, colors="#000000", width=0.6)
    ax.set_xlabel(ax.get_xlabel(), fontsize=_AX_LABEL_FS, fontfamily=_FONT_FAMILY)
    ax.set_ylabel(ax.get_ylabel(), fontsize=_AX_LABEL_FS, fontfamily=_FONT_FAMILY)
    ax.title.set_fontsize(_TITLE_FS)
    ax.title.set_fontfamily(_FONT_FAMILY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_edgecolor(_SPINE_COLOR)
        ax.spines[sp].set_linewidth(0.8)


def _ieee_legend(ax: plt.Axes, **kwargs) -> None:
    """Add a consistently styled legend."""
    ax.legend(
        fontsize=_LEGEND_FS,
        facecolor=_LEGEND_FACE,
        edgecolor=_SPINE_COLOR,
        labelcolor="#000000",
        framealpha=0.85,
        **kwargs,
    )


def _kde_line(ax: plt.Axes, scores: list, color: str, label: str,
              linestyle: str = "-", linewidth: float = 1.4,
              fill_alpha: float = 0.18, rug: bool = True) -> None:
    """Draw a single KDE curve"""
    arr = np.array(scores, dtype=np.float64)
    if len(arr) < 2:
        return
    xs = np.linspace(arr.min() - 1, arr.max() + 1, 500)
    try:
        kde = gaussian_kde(arr, bw_method="scott")
        ys  = kde(xs)
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        ax.fill_between(xs, ys, alpha=fill_alpha, color=color)
        if rug:
            ax.plot(arr, np.full_like(arr, -0.003), "|", color=color, alpha=0.35, markersize=3.5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------
def kde_plot(
    scores_by_label: Dict[str, list],
    suspect_cm: Optional[float] = None,
    title: str = "",
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """KDE plot of bonafide / synthetic CM score distributions.

    Parameters
    ----------
    scores_by_label : dict mapping label -> list of CM scores
    suspect_cm      : target CM score; drawn as a vertical line
    title           : subplot title
    threshold       : explicit decision threshold; if None the EER threshold
                      is computed automatically when both classes are present
    ax              : existing Axes to draw on; if None a new figure is created

    Returns
    -------
    Figure when ax is None (standalone mode), else None.
    """
    from detection import compute_eer

    standalone = ax is None
    if standalone:
        fig, ax = _new_fig((6, 3.0))

    all_sc = [v for lst in scores_by_label.values() for v in lst]
    if not all_sc:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="#aaa", transform=ax.transAxes, fontsize=_AX_LABEL_FS)
        if standalone:
            return fig
        return None

    eer_line = None
    nat = scores_by_label.get("bonafide", [])
    syn = scores_by_label.get("synthetic", [])
    if threshold is None and len(nat) >= 2 and len(syn) >= 2:
        try:
            sc       = np.array(nat + syn, dtype=np.float64)
            lb       = np.array([0] * len(nat) + [1] * len(syn), dtype=np.int32)
            eer_val, eer_tau, _, _ = compute_eer(sc, lb)
            tau_d    = float(np.clip(eer_tau, sc.min(), sc.max()))
            overflow = abs(eer_tau - tau_d) > 0.01
            eer_line = (tau_d, eer_val, overflow)
        except Exception:
            pass

    for label, scores in scores_by_label.items():
        _kde_line(ax, scores, color=LABEL_COLORS.get(label, "#888"),
                  label=f"{label} (n={len(scores)})")

    draw_tau = threshold if threshold is not None else (eer_line[0] if eer_line else None)
    if draw_tau is not None:
        if eer_line and threshold is None:
            tau_d, eer_v, overflow = eer_line
            lbl = (
                f"EER threshold > range (EER={eer_v * 100:.1f}%)" if overflow
                else f"EER threshold ({tau_d:.3f}, EER={eer_v * 100:.1f}%)"
            )
        else:
            lbl = f"threshold ({draw_tau:.3f})"
        ax.axvline(draw_tau, color=_THRESH_COLOR, linewidth=_THRESH_LW,
                   linestyle=_THRESH_LS, label=lbl, zorder=4)

    if suspect_cm is not None and not np.isnan(suspect_cm):
        ax.axvline(suspect_cm, color=_TARGET_COLOR, linewidth=_TARGET_LW,
                   linestyle=_TARGET_LS, label=f"target ({suspect_cm:.2f})", zorder=5)

    ax.set_xlabel("CM score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    _apply_style(ax)
    _ieee_legend(ax)

    if standalone:
        fig.tight_layout(pad=0.3)
        return fig


def bm_kde_sidebyside(
    bm_by: Dict[str, list],
    mbm_by: Dict[str, list],
    target_cm: Optional[float],
    pipe_hash: str = "",
) -> plt.Figure:
    """Side-by-side KDE plots for BM and M-BM."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), sharey=False)
    kde_plot(bm_by,  suspect_cm=target_cm, title="BM (original)", ax=axes[0])
    kde_plot(mbm_by, suspect_cm=target_cm,
             title=f"M-BM{f'  [{pipe_hash}]' if pipe_hash else ''}", ax=axes[1])
    fig.tight_layout(pad=0.3)
    return fig


def bm_kde_overlay(
    bm_by: Dict[str, list],
    mbm_by: Dict[str, list],
    target_cm: Optional[float],
) -> Optional[plt.Figure]:
    """Overlay KDE of BM vs M-BM. Returns None when fewer than 4 combined scores."""
    all_combined = (
        bm_by.get("bonafide", []) + bm_by.get("synthetic", []) +
        mbm_by.get("bonafide", []) + mbm_by.get("synthetic", [])
    )
    if len(all_combined) < 4:
        return None

    fig, ax = _new_fig((11, 3.0))
    for label, scores, ls, lw in [
        ("BM — bonafide",    bm_by.get("bonafide",  []), "-",  1.5),
        ("BM — synthetic",   bm_by.get("synthetic", []), "-",  1.5),
        ("M-BM — bonafide",  mbm_by.get("bonafide", []), "--", 1.2),
        ("M-BM — synthetic", mbm_by.get("synthetic",[]), "--", 1.2),
    ]:
        key = "bonafide" if "bonafide" in label else "synthetic"
        _kde_line(ax, scores, color=LABEL_COLORS[key], label=label,
                  linestyle=ls, linewidth=lw, fill_alpha=0.06, rug=False)

    if target_cm is not None and not np.isnan(target_cm):
        ax.axvline(target_cm, color=_TARGET_COLOR, linewidth=_TARGET_LW,
                   linestyle=_TARGET_LS, label=f"target ({target_cm:.2f})", zorder=6)

    ax.set_xlabel("CM score")
    ax.set_ylabel("Density")
    ax.set_title("BM vs M-BM overlay")
    _apply_style(ax)
    _ieee_legend(ax, ncol=2)
    fig.tight_layout(pad=0.3)
    return fig


def per_dataset_kde(
    selected_specs: list,
    per_ds_nat: Dict[str, list],
    per_ds_syn: Dict[str, list],
) -> Optional[plt.Figure]:
    """KDE overlay coloured by dataset. Returns None when fewer than 4 combined scores."""
    all_scores = [
        v for ds in selected_specs
        for v in per_ds_nat.get(ds.key, []) + per_ds_syn.get(ds.key, [])
    ]
    if len(all_scores) < 4:
        return None

    fig, ax = _new_fig((10, 3.0))
    for i, ds in enumerate(selected_specs):
        color = _ds_color(i)
        for scores, lbl, ls in [
            (per_ds_nat.get(ds.key, []), "bonafide",  "-"),
            (per_ds_syn.get(ds.key, []), "synthetic", "--"),
        ]:
            _kde_line(ax, scores, color=color,
                      label=f"{ds.display_name} — {lbl} (n={len(scores)})",
                      linestyle=ls, fill_alpha=0.07, rug=False)

    ax.set_xlabel("CM score")
    ax.set_ylabel("Density")
    ax.set_title("Per-dataset CM score distributions")
    _apply_style(ax)
    _ieee_legend(ax, ncol=2)
    fig.tight_layout(pad=0.3)
    return fig


def combo_shift_kde(
    uid_dict: dict,
    combo_key: str,
) -> Optional[plt.Figure]:
    """Before/after KDE for a single manipulation combination.

    Parameters
    ----------
    uid_dict  : {uid: {"label": str, "before": float, "after": float}}
    combo_key : display name shown as the figure title
    """
    if not uid_dict:
        return None

    fig, ax = _new_fig((6, 2.8))
    for lbl, color in LABEL_COLORS.items():
        before = [v["before"] for v in uid_dict.values() if v["label"] == lbl]
        after  = [v["after"]  for v in uid_dict.values() if v["label"] == lbl]
        if len(before) < 2:
            continue
        _kde_line(ax, before, color=color, label=f"{lbl} — before",
                  linestyle="-",  linewidth=1.4, fill_alpha=0.12, rug=False)
        _kde_line(ax, after,  color=color, label=f"{lbl} — after",
                  linestyle="--", linewidth=1.4, fill_alpha=0.06, rug=False)

    ax.set_xlabel("CM score")
    ax.set_ylabel("Density")
    ax.set_title(combo_key)
    _apply_style(ax)
    _ieee_legend(ax, ncol=2)
    fig.tight_layout(pad=0.3)
    return fig


def neighbourhood_composition(
    nb_bm: dict,
    nb_mbm: dict,
    sd_w: float,
) -> plt.Figure:
    """Stacked bar chart of bonafide/synthetic composition in a CM score window.

    Parameters
    ----------
    nb_bm     : neighbourhood() result dict for BM
    nb_mbm    : neighbourhood() result dict for M-BM
    sd_w      : window width in CM unit
    """
    fig, ax = _new_fig((3.0, 2))

    d_nat = nb_mbm["n_nat_in"] - nb_bm["n_nat_in"]
    d_syn = nb_mbm["n_syn_in"] - nb_bm["n_syn_in"]

    for i, nb in enumerate([nb_bm, nb_mbm]):
        total = nb["n_nat_in"] + nb["n_syn_in"]
        if total == 0:
            continue
        pct_nat = 100 * nb["n_nat_in"] / total
        pct_syn = 100 * nb["n_syn_in"] / total
        ax.bar(i, pct_nat, color=LABEL_COLORS["bonafide"],  width=0.5)
        ax.bar(i, pct_syn, bottom=pct_nat, color=LABEL_COLORS["synthetic"], width=0.5)
        ax.text(i, pct_nat / 2, f"{pct_nat:.1f}%",
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", fontfamily=_FONT_FAMILY)
    
        ax.text(i, pct_nat + pct_syn / 2, f"{pct_syn:.1f}%",
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", fontfamily=_FONT_FAMILY)

    _ieee_legend(ax, loc="upper left", handles=[
        Patch(color=LABEL_COLORS["bonafide"],  label=f"Bonafide  (Δ {_sign(d_nat)})"),
        Patch(color=LABEL_COLORS["synthetic"], label=f"Synthetic  (Δ {_sign(d_syn)})"),
    ])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BM", "M-BM"], fontsize=_TICK_FS + 1, fontfamily=_FONT_FAMILY)
    ax.set_ylabel("Composition in window (%)")
    ax.set_ylim(0, 100)
    _apply_style(ax)
    ax.set_title(f"Neighbourhood  ±SD ({sd_w:.2f} CM)", fontsize = 8, pad=6)
    fig.tight_layout(pad=0.3)
    return fig