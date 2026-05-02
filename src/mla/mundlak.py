"""Within-between (Mundlak) decomposition helpers.

Implements the country-year exposure aggregation and the Mundlak
specification from Bell & Jones (2015) and Schmidt-Catran & Fairbrother
(2016). All functions are pandas-only — no statsmodels dependency.

The two key transformations:

* :func:`country_year_aggregate` — weighted mean of an individual variable
  ``X`` within each (country, round) cell, optionally with a leave-one-out
  variant for the mechanical-correlation robustness check.
* :func:`within_between_decompose` — split a level-2 series ``X_ct`` into
  its between-country mean ``X̄_c`` and its within-country deviation
  ``X_ct − X̄_c``. The two together replace ``X_ct`` in the regression
  and let the between-coefficient ``γ_B`` and the within-coefficient
  ``γ_W`` differ.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
# 1. Country-year aggregation of an individual-level variable
# --------------------------------------------------------------------- #
def country_year_aggregate(
    panel: pd.DataFrame,
    value_col: str,
    *,
    weight_col: str | None = "pspwght",
    country_col: str = "cntry",
    round_col: str = "essround",
    out_col: str | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Compute the (weighted) country-round mean of ``value_col``.

    Returns a DataFrame keyed by (``country_col``, ``round_col``) with
    one column ``out_col`` (default = ``f"{value_col}_ct"``). Rows where
    ``value_col`` is NaN are dropped before averaging when ``drop_na=True``.

    Uses ``pspwght`` design weights by default (or any weight column), or
    an unweighted mean if ``weight_col=None``.
    """
    out_col = out_col or f"{value_col}_ct"
    df = panel.loc[:, [country_col, round_col, value_col] + (
        [weight_col] if weight_col else []
    )]
    if drop_na:
        df = df[df[value_col].notna()]
    if weight_col is None:
        agg = (
            df.groupby([country_col, round_col], observed=True)[value_col]
            .mean()
            .rename(out_col)
            .reset_index()
        )
    else:
        # Weighted mean: Σ(w·x) / Σ(w). Drop rows where weight is NaN/0
        # so a single missing weight doesn't poison the country-year cell.
        df = df[df[weight_col].notna() & (df[weight_col] > 0)]
        df = df.assign(_wx=df[weight_col] * df[value_col])
        grouped = df.groupby([country_col, round_col], observed=True)
        wx_sum = grouped["_wx"].sum()
        w_sum = grouped[weight_col].sum()
        agg = (wx_sum / w_sum).rename(out_col).reset_index()
    return agg


def country_year_aggregate_leave_one_out(
    panel: pd.DataFrame,
    value_col: str,
    *,
    weight_col: str | None = "pspwght",
    country_col: str = "cntry",
    round_col: str = "essround",
    out_col: str | None = None,
) -> pd.Series:
    """Per-individual leave-one-out country-round mean of ``value_col``.

    For each row i, returns the (weighted) mean of ``value_col`` over all
    *other* respondents in the same (country, round). Used as a Section-7
    robustness check on the mechanical correlation between an individual's
    own exposure and the country-year average.

    Returns a Series aligned to ``panel.index`` (NaN where ``value_col``
    or ``weight_col`` is NaN, or where the cell has only one respondent).
    """
    out_col = out_col or f"{value_col}_ct_loo"
    if weight_col is None:
        # Unweighted: x̄_(-i) = (Σ x − x_i) / (n − 1).
        grouped = panel.groupby([country_col, round_col], observed=True)[value_col]
        sum_x = grouped.transform("sum")
        n = grouped.transform("count")
        loo = (sum_x - panel[value_col]) / (n - 1)
        loo[n <= 1] = np.nan
        loo[panel[value_col].isna()] = np.nan
        return loo.rename(out_col)
    # Weighted: x̄_(-i) = (Σ w·x − w_i·x_i) / (Σ w − w_i).
    # CRITICAL: gate weights on BOTH value-validity and weight-validity.
    # Otherwise a respondent with NaN value but valid weight inflates the
    # denominator while contributing nothing to the numerator, biasing
    # LOO downward for every other respondent in the same cell.
    valid = panel[value_col].notna() & panel[weight_col].notna() & panel[weight_col].gt(0)
    w_eff = panel[weight_col].where(valid, other=0.0)
    wx_eff = (panel[value_col].fillna(0.0) * w_eff)
    grouped = panel.assign(_w=w_eff, _wx=wx_eff).groupby(
        [country_col, round_col], observed=True
    )
    sum_w = grouped["_w"].transform("sum")
    sum_wx = grouped["_wx"].transform("sum")
    n_pos = grouped["_w"].transform(lambda s: s.gt(0).sum())
    denom = sum_w - w_eff
    loo = (sum_wx - wx_eff) / denom
    loo[n_pos <= 1] = np.nan
    loo[~valid] = np.nan
    loo[denom.le(0)] = np.nan
    return loo.rename(out_col)


# --------------------------------------------------------------------- #
# 2. Within-between (Mundlak) decomposition of a level-2 variable
# --------------------------------------------------------------------- #
def within_between_decompose(
    cy: pd.DataFrame,
    value_col: str,
    *,
    country_col: str = "cntry",
    between_suffix: str = "_between",
    within_suffix: str = "_within",
) -> pd.DataFrame:
    """Decompose a country-year series into between-country mean + within-country deviation.

    Given a frame keyed on (``country_col``, ``round``) with one
    measurement column ``value_col`` per row, returns the same frame plus:

    * ``{value_col}_between`` — country-mean ``X̄_c`` (constant within country).
    * ``{value_col}_within``  — deviation from country-mean ``X_ct − X̄_c``.

    Identity: ``X_ct == between + within`` (within tolerance).

    This is the Mundlak (1978) / Bell & Jones (2015) / Schmidt-Catran &
    Fairbrother (2016) reformulation. Including both columns in the
    regression instead of ``X_ct`` lets the between-coefficient ``γ_B``
    differ from the within-coefficient ``γ_W``; the Wald test of equality
    is the substantive Mundlak test.
    """
    out = cy.copy()
    means = (
        out.groupby(country_col, observed=True)[value_col]
        .transform("mean")
    )
    out[value_col + between_suffix] = means
    out[value_col + within_suffix] = out[value_col] - means
    return out


def merge_country_year_to_panel(
    panel: pd.DataFrame,
    cy: pd.DataFrame,
    *,
    country_col: str = "cntry",
    round_col: str = "essround",
    cy_value_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Left-join a country-year frame onto an individual panel.

    Validates the join is many-to-one on (``country_col``, ``round_col``)
    and preserves panel size. If ``cy_value_cols`` is given, only those
    columns are merged.
    """
    if cy.duplicated([country_col, round_col]).any():
        raise ValueError(
            "country-year frame has duplicate (country, round) keys"
        )
    if cy_value_cols is not None:
        cy = cy[[country_col, round_col, *cy_value_cols]]
    n_pre = len(panel)
    out = panel.merge(cy, on=[country_col, round_col], how="left", validate="many_to_one")
    assert len(out) == n_pre, "merge changed panel length"
    return out
