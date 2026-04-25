"""statsmodels.MixedLM wrappers for the 3-level pooled cross-national design.

Implements the M0 → M6 progression from :doc:`/IMPLEMENTATION_PLAN` §6:

* M0 — empty 3-level intercept only.
* M1 — M0 + L1 fixed effects.
* M2 — M1 + L2 macro controls (incl. round dummies).
* M3a/M3b — M2 + Mundlak within-between (with / without round dummies).
* M4 — M3a + random slope on individual GenAI exposure.
* M5 — M4 + cross-level interaction `exposure_within × eisced`.
* M6 — M5 + L3 institutional moderator.

The 3-level structure (individuals → country-years → countries) is
encoded for statsmodels by passing ``groups="cntry"`` for the L3 random
intercept and ``vc_formula={"cy": "0 + C(country_year)"}`` for the L2
random intercept nested in L3. The variance components from the result
are:

* ``cov_re``  — L3 random-effect covariance (1×1 in intercept-only fits).
* ``vcomp``   — L2 variance components keyed by their formula name.
* ``scale``   — residual variance σ²_e.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults


# --------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------- #
def build_trust_composite(
    df: pd.DataFrame,
    items: Iterable[str] = ("trstprl", "trstlgl", "stfdem"),
    out_col: str = "trust",
    valid_range: tuple[float, float] = (0, 10),
) -> pd.DataFrame:
    """Compute the institutional-trust composite as the per-item z-standardised mean.

    Each item is z-standardised across the pooled panel (per item,
    globally) to put the three on a common scale, then averaged. Sentinel
    codes (>10) are coerced to NaN first.

    NB: the plan §5 wording ("z-standardise per round-country") would
    sweep out exactly the L2 and L3 variance the multilevel model is
    meant to estimate; we standardise per-item globally instead so the
    between-country and between-round variance is preserved. ICC at L3
    on trust composites in the plan's reference works (Schmidt-Catran &
    Fairbrother) is ~0.15–0.25, which is consistent with global
    standardisation.
    """
    out = df.copy()
    lo, hi = valid_range
    items = list(items)
    cleaned: list[pd.Series] = []
    for c in items:
        s = out[c].where(out[c].between(lo, hi))
        # Global per-item z-standardisation.
        z = (s - s.mean()) / s.std()
        cleaned.append(z.rename(c))
    z_df = pd.concat(cleaned, axis=1)
    out[out_col] = z_df.mean(axis=1)
    return out


def add_country_year_key(
    df: pd.DataFrame,
    country_col: str = "cntry",
    round_col: str = "essround",
    out_col: str = "country_year",
) -> pd.DataFrame:
    """Attach a ``country_year`` string key (e.g. ``BE_R6``)."""
    out = df.copy()
    out[out_col] = out[country_col].astype(str) + "_R" + out[round_col].astype(str)
    return out


# --------------------------------------------------------------------- #
# Variance components / ICC / VPC
# --------------------------------------------------------------------- #
@dataclass
class VarianceComponents:
    sigma_u0_sq: float  # L3 random intercept variance (countries)
    sigma_v0_sq: float  # L2 random intercept variance (country-years within country)
    sigma_e_sq: float   # L1 residual variance
    icc_l3: float       # σ²_u0 / total
    vpc_l2: float       # σ²_v0 / total
    vpc_l1: float       # σ²_e / total
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "sigma_u0_sq": self.sigma_u0_sq,
            "sigma_v0_sq": self.sigma_v0_sq,
            "sigma_e_sq": self.sigma_e_sq,
            "icc_l3": self.icc_l3,
            "vpc_l2": self.vpc_l2,
            "vpc_l1": self.vpc_l1,
            "total": self.total,
        }


def variance_components(result: MixedLMResults) -> VarianceComponents:
    """Decompose a 3-level MixedLM result into σ²_u0 (L3), σ²_v0 (L2), σ²_e (L1).

    Assumes the model was fit with ``groups=cntry``, ``re_formula="~1"``
    (intercept-only L3), and a single L2 variance component named in
    ``vc_formula``.
    """
    # L3: covariance matrix of the L3 random effects. For intercept-only
    # we get a 1×1 matrix.
    cov_re = np.asarray(result.cov_re)
    sigma_u0_sq = float(cov_re[0, 0]) if cov_re.size else 0.0

    # L2: the vcomp dict maps VC name → variance estimate. Sum across
    # entries so this works even if multiple VCs are in play later.
    vcomp = result.vcomp
    sigma_v0_sq = float(np.sum(np.asarray(vcomp))) if vcomp is not None else 0.0

    # L1: residual.
    sigma_e_sq = float(result.scale)

    total = sigma_u0_sq + sigma_v0_sq + sigma_e_sq
    return VarianceComponents(
        sigma_u0_sq=sigma_u0_sq,
        sigma_v0_sq=sigma_v0_sq,
        sigma_e_sq=sigma_e_sq,
        icc_l3=sigma_u0_sq / total if total > 0 else float("nan"),
        vpc_l2=sigma_v0_sq / total if total > 0 else float("nan"),
        vpc_l1=sigma_e_sq / total if total > 0 else float("nan"),
        total=total,
    )


# --------------------------------------------------------------------- #
# Three-level fitter
# --------------------------------------------------------------------- #
def fit_3level(
    formula: str,
    data: pd.DataFrame,
    *,
    country_col: str = "cntry",
    country_year_col: str = "country_year",
    re_formula: str = "~1",
    method: list[str] | str | None = None,
    reml: bool = True,
    maxiter: int = 200,
) -> MixedLMResults:
    """Fit a 3-level MixedLM: individuals in country-years in countries.

    Drops rows where any term referenced in the formula is NaN before
    fitting (statsmodels requires complete cases).
    """
    # Use lbfgs by default — more robust than the default Newton-Raphson
    # for variance components on the boundary.
    method = method or ["lbfgs"]
    md = smf.mixedlm(
        formula,
        data=data,
        groups=country_col,
        re_formula=re_formula,
        vc_formula={"cy": f"0 + C({country_year_col})"},
        missing="drop",
    )
    return md.fit(reml=reml, method=method, maxiter=maxiter)


# --------------------------------------------------------------------- #
# Master table builder for M0 → M6
# --------------------------------------------------------------------- #
def master_row(
    name: str,
    result: MixedLMResults,
    *,
    coefs: Iterable[str] = (),
) -> dict[str, float | str]:
    """One row of the master regression table.

    Includes (for ``coefs``) the point estimate and SE from the fixed
    part, plus the variance components and a few global fit statistics.
    """
    vc = variance_components(result)
    row: dict[str, float | str] = {
        "model": name,
        "n_obs": int(result.nobs),
        "n_groups_l3": int(result.model.n_groups),
        "loglik": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "sigma_u0_sq": vc.sigma_u0_sq,
        "sigma_v0_sq": vc.sigma_v0_sq,
        "sigma_e_sq": vc.sigma_e_sq,
        "icc_l3": vc.icc_l3,
        "vpc_l2": vc.vpc_l2,
    }
    for c in coefs:
        if c in result.params.index:
            row[f"{c}__beta"] = float(result.params[c])
            row[f"{c}__se"] = float(result.bse[c])
        else:
            row[f"{c}__beta"] = float("nan")
            row[f"{c}__se"] = float("nan")
    return row


def master_table(
    results: dict[str, MixedLMResults],
    coefs: Iterable[str] = (),
) -> pd.DataFrame:
    """Build a master table comparing several MixedLM fits in one row each."""
    rows = [master_row(name, res, coefs=coefs) for name, res in results.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Snijders & Bosker proportional variance reduction
# --------------------------------------------------------------------- #
def proportional_variance_reduction(
    baseline: MixedLMResults, candidate: MixedLMResults
) -> dict[str, float]:
    """% reduction in each variance component going from ``baseline`` to ``candidate``.

    Convention: positive value = candidate explains that share of the
    baseline variance at that level. Useful for the Snijders & Bosker
    style "M1 explains X% of L3 variance" reporting.
    """
    base = variance_components(baseline)
    cand = variance_components(candidate)
    return {
        "delta_sigma_u0_sq_pct": 100 * (base.sigma_u0_sq - cand.sigma_u0_sq) / base.sigma_u0_sq if base.sigma_u0_sq > 0 else float("nan"),
        "delta_sigma_v0_sq_pct": 100 * (base.sigma_v0_sq - cand.sigma_v0_sq) / base.sigma_v0_sq if base.sigma_v0_sq > 0 else float("nan"),
        "delta_sigma_e_sq_pct": 100 * (base.sigma_e_sq - cand.sigma_e_sq) / base.sigma_e_sq if base.sigma_e_sq > 0 else float("nan"),
    }


# --------------------------------------------------------------------- #
# Mundlak Wald test of γ_W = γ_B
# --------------------------------------------------------------------- #
def mundlak_wald(
    result: MixedLMResults,
    within_term: str,
    between_term: str,
) -> dict[str, float]:
    """Wald test of equality of within- and between-coefficients.

    Returns ``{chi2, df, pvalue, gamma_w, gamma_b, diff, se_diff}``.
    """
    from scipy.stats import chi2 as _chi2

    if within_term not in result.params.index or between_term not in result.params.index:
        raise KeyError(
            f"Need both {within_term} and {between_term} in the model's fixed effects."
        )
    g_w = result.params[within_term]
    g_b = result.params[between_term]
    diff = g_w - g_b

    # Variance of the difference: Var(g_w) + Var(g_b) − 2·Cov(g_w, g_b).
    cov = result.cov_params()
    v_w = cov.loc[within_term, within_term]
    v_b = cov.loc[between_term, between_term]
    cov_wb = cov.loc[within_term, between_term]
    var_diff = v_w + v_b - 2 * cov_wb
    se_diff = float(np.sqrt(max(var_diff, 0.0)))

    chi2_stat = float(diff**2 / var_diff) if var_diff > 0 else float("nan")
    p = float(1 - _chi2.cdf(chi2_stat, df=1)) if not np.isnan(chi2_stat) else float("nan")
    return {
        "gamma_w": float(g_w),
        "gamma_b": float(g_b),
        "diff": float(diff),
        "se_diff": se_diff,
        "chi2": chi2_stat,
        "df": 1.0,
        "pvalue": p,
    }
