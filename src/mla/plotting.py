"""Plotting helpers for the paper.

All functions return ``matplotlib.Figure`` objects so the caller can
``fig.savefig(...)`` to whatever path/format. Figures are styled
deliberately plainly — minimal chartjunk, single-colour where possible,
suitable for a 12-page LaTeX paper.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLMResults


# --------------------------------------------------------------------- #
# Figure 1: country-year exposure trajectories
# --------------------------------------------------------------------- #
def country_year_exposure_trajectories(
    cy_panel: pd.DataFrame,
    *,
    country_col: str = "cntry",
    round_col: str = "essround",
    value_col: str = "exposure_ct",
    static_col: str | None = "exposure_ct_static",
    figsize: tuple[float, float] = (8, 4.5),
) -> plt.Figure:
    """Per-country line trajectories of $\\bar E_{ct}$ across rounds.

    Highlights the R10 → R11 vintage break (2023 GBB → 2025 ILO–NASK)
    with a vertical reference line. Optionally overlays the static-
    vintage comparator in dashed lines so the gap (vintage shock vs
    compositional drift) is visible by eye.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for cntry, grp in cy_panel.sort_values(round_col).groupby(country_col, observed=True):
        ax.plot(
            grp[round_col], grp[value_col],
            color="black", alpha=0.25, linewidth=0.8, marker="o", markersize=2,
        )
    # Heavy mean line
    means = cy_panel.groupby(round_col)[value_col].mean()
    ax.plot(
        means.index, means.values,
        color="C3", linewidth=2.5, marker="s", markersize=5,
        label="cross-country mean",
    )
    if static_col and static_col in cy_panel.columns:
        static_means = cy_panel.groupby(round_col)[static_col].mean()
        ax.plot(
            static_means.index, static_means.values,
            color="C0", linewidth=2.0, linestyle="--", marker="^", markersize=4,
            label="static-vintage mean (2025 throughout)",
        )
    ax.axvline(10.5, color="grey", linestyle=":", alpha=0.6)
    ax.text(10.55, ax.get_ylim()[1] * 0.98, "vintage shift\n2023 → 2025",
            color="grey", fontsize=8, va="top")
    ax.set_xlabel("ESS round")
    ax.set_ylabel(r"$\bar E_{ct}$ (weighted country-year exposure)")
    ax.set_xticks(sorted(cy_panel[round_col].unique()))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Figure 2: caterpillar plot of country random intercepts
# --------------------------------------------------------------------- #
def caterpillar_country_intercepts(
    result: MixedLMResults,
    *,
    figsize: tuple[float, float] = (6, 8),
) -> plt.Figure:
    """Ranked dot plot of country random intercepts (empirical-Bayes BLUPs).

    Pulls ``result.random_effects`` (dict country → Series of BLUPs) and
    plots the posterior means ranked low-to-high. No per-country error
    bars are drawn: ``statsmodels.MixedLM`` does not expose cluster-specific
    conditional variances, and the single estimated ``sigma_v0`` would make
    any envelope identical across countries (hence uninformative). The
    spread of the points themselves is the quantity of interest.
    """
    res_dict = result.random_effects
    rows = [
        {"cntry": cntry, "intercept": float(np.asarray(blups)[0])}
        for cntry, blups in res_dict.items()
    ]
    df = pd.DataFrame(rows).sort_values("intercept").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(df))
    # stems to the grand mean to read sign/magnitude at a glance
    ax.hlines(y, 0, df["intercept"], color="grey", linewidth=0.5, zorder=1)
    ax.scatter(df["intercept"], y, color="black", s=14, zorder=2)
    ax.axvline(0, color="C3", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["cntry"], fontsize=8)
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_xlabel("country random intercept (BLUP) on standardised trust composite")
    ax.set_title("Between-country trust heterogeneity — ranked country BLUPs (M3a)",
                 fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Figure 3: conditional effects of within-exposure × education (M5)
# --------------------------------------------------------------------- #
def conditional_within_x_education(
    result: MixedLMResults,
    *,
    within_term: str = "exposure_ct_within",
    interaction_term: str = "exposure_ct_within:eisced_c",
    eisced_centre: int = 4,
    eisced_levels: Iterable[int] = range(0, 8),
    within_grid: tuple[float, float, int] = (-0.05, 0.05, 21),
    figsize: tuple[float, float] = (7, 4.5),
) -> plt.Figure:
    """Plot the conditional effect of within-country exposure on trust at
    each ISCED level (predicted shift in standardised trust).

    All other covariates are evaluated at zero (centred). The lines diverge
    iff the cross-level interaction coefficient is non-zero.
    """
    main = float(result.params.get(within_term, 0.0))
    inter = float(result.params.get(interaction_term, 0.0))
    grid = np.linspace(*within_grid)
    fig, ax = plt.subplots(figsize=figsize)
    for level in eisced_levels:
        delta = main * grid + inter * grid * (level - eisced_centre)
        ax.plot(grid, delta, label=f"ISCED {level}")
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.6)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.6)
    ax.set_xlabel(r"$E_{ct} - \bar E_c$  (within-country exposure deviation)")
    ax.set_ylabel("predicted Δ trust (standardised)")
    ax.set_title("Conditional within-exposure effect by individual education", fontsize=10)
    ax.legend(loc="best", fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Figure 4: variance components bar chart M0 → M6
# --------------------------------------------------------------------- #
def variance_components_bar(
    table: pd.DataFrame,
    *,
    model_col: str = "model",
    cols: tuple[str, str, str] = ("sigma_u0_sq", "sigma_v0_sq", "sigma_e_sq"),
    labels: tuple[str, str, str] = ("L3 country", "L2 country-year", "L1 individual"),
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Stacked bar of the three variance components per model row."""
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(table))
    colors = ("C3", "C0", "lightgrey")
    for c, lbl, color in zip(cols, labels, colors):
        ax.bar(table[model_col], table[c], bottom=bottom, label=lbl, color=color, edgecolor="white", linewidth=0.6)
        bottom = bottom + np.asarray(table[c].fillna(0))
    ax.set_ylabel("variance")
    ax.set_xlabel("model")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Variance decomposition across the M0 → M6 model progression", fontsize=10)
    fig.tight_layout()
    return fig
