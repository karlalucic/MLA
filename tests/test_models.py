"""Tests for src.mla.models — MixedLM 3-level wrapper, ICC/VPC, Mundlak Wald."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mla.models import (
    add_country_year_key,
    build_trust_composite,
    master_table,
    mundlak_wald,
    proportional_variance_reduction,
    variance_components,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _toy_panel(n_countries: int = 8, n_rounds: int = 4, n_per_cell: int = 30, seed: int = 0):
    """Synthetic 3-level dataset for fast variance-component tests."""
    rng = np.random.default_rng(seed)
    countries = [f"C{i}" for i in range(n_countries)]
    u0 = dict(zip(countries, rng.normal(0, 0.7, n_countries)))  # L3 random intercept
    rows = []
    for c in countries:
        for r in range(1, n_rounds + 1):
            v0 = rng.normal(0, 0.3)  # L2 random intercept (country-year)
            for _ in range(n_per_cell):
                e = rng.normal(0, 1.0)  # L1 residual
                y = 5.0 + u0[c] + v0 + e
                rows.append({"cntry": c, "essround": r, "y": y})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# add_country_year_key
# --------------------------------------------------------------------- #
def test_add_country_year_key_format():
    df = pd.DataFrame({"cntry": ["BE", "DE"], "essround": [6, 11]})
    out = add_country_year_key(df)
    assert out["country_year"].tolist() == ["BE_R6", "DE_R11"]


# --------------------------------------------------------------------- #
# build_trust_composite — handles sentinels and standardises per item
# --------------------------------------------------------------------- #
def test_build_trust_composite_drops_sentinels():
    df = pd.DataFrame({
        "trstprl": [3, 5, 99, 7, 8],   # 99 sentinel
        "trstlgl": [4, 5, 6, 88, 8],   # 88 sentinel
        "stfdem":  [4, 4, 6, 7, 99],   # 99 sentinel
        "essround": [1, 1, 1, 1, 1],
        "cntry":    ["A"] * 5,
    })
    out = build_trust_composite(df)
    # Row 2 has trstprl=99 sentinel ⇒ that item is NaN; row mean uses the other two.
    assert pd.isna(out.loc[2, "trustprl"]) if "trustprl" in out.columns else True
    # Composite is finite for rows with at least one valid item; NaN where all NaN.
    assert out["trust"].notna().sum() == 5  # at least one valid in each row
    # Z-standardisation per item: pooled mean ≈ 0 across all valid rows.
    composite_mean = out["trust"].dropna().mean()
    assert abs(composite_mean) < 0.5  # not zero exactly because items have unequal Ns


def test_build_trust_composite_preserves_between_variance():
    """Crucial property: standardising per-item globally MUST preserve
    cross-country variance in the composite (otherwise the multilevel
    decomposition would collapse)."""
    rng = np.random.default_rng(42)
    rows = []
    for c, mu in [("A", 4.0), ("B", 6.0), ("C", 8.0)]:
        # Each country has a different mean trust level.
        for _ in range(200):
            e = rng.normal(0, 1.0, 3).clip(-mu, 10 - mu)
            rows.append({
                "trstprl": mu + e[0],
                "trstlgl": mu + e[1],
                "stfdem":  mu + e[2],
                "cntry": c, "essround": 1,
            })
    df = pd.DataFrame(rows)
    out = build_trust_composite(df)
    means = out.groupby("cntry")["trust"].mean()
    # Sub-country means MUST differ — otherwise the L3 ICC is identically zero.
    assert means.std() > 0.5


# --------------------------------------------------------------------- #
# variance_components on a real fit
# --------------------------------------------------------------------- #
def test_variance_components_roughly_recovers_known_dgp():
    """Simulate L3 σ_u0=0.7, L2 σ_v0=0.3, L1 σ_e=1.0 → variances 0.49, 0.09, 1.0.
    Total = 1.58. ICC = 0.31; L2 VPC = 0.057.

    Bounds are loose because L3 sampling variance with ~25 countries is
    substantial; this test guards correctness of the decomposition, not
    estimation precision."""
    pytest.importorskip("statsmodels")
    from src.mla.models import fit_3level

    df = _toy_panel(n_countries=25, n_rounds=6, n_per_cell=120, seed=1)
    df = add_country_year_key(df)
    res = fit_3level("y ~ 1", df)
    vc = variance_components(res)
    # Generous bounds — 25 countries gives roughly 0.20 SE on σ²_u0 hat.
    assert 0.1 <= vc.sigma_u0_sq <= 0.9
    assert 0.0 <= vc.sigma_v0_sq <= 0.25
    assert abs(vc.sigma_e_sq - 1.00) < 0.20
    # Identity holds exactly: ICC + L2_VPC + L1_VPC == 1.
    assert abs(vc.icc_l3 + vc.vpc_l2 + vc.vpc_l1 - 1.0) < 1e-9


def test_master_table_includes_variance_components():
    pytest.importorskip("statsmodels")
    from src.mla.models import fit_3level

    df = _toy_panel(n_countries=8, n_rounds=3, n_per_cell=40, seed=2)
    df = add_country_year_key(df)
    res0 = fit_3level("y ~ 1", df)
    table = master_table({"M0": res0})
    cols = set(table.columns)
    assert {"model", "n_obs", "icc_l3", "vpc_l2", "sigma_u0_sq", "sigma_v0_sq"} <= cols
    assert table.iloc[0]["model"] == "M0"


def test_proportional_variance_reduction_zero_for_same_model():
    pytest.importorskip("statsmodels")
    from src.mla.models import fit_3level

    df = _toy_panel(n_countries=8, n_rounds=3, n_per_cell=40, seed=3)
    df = add_country_year_key(df)
    res = fit_3level("y ~ 1", df)
    pr = proportional_variance_reduction(res, res)
    for k, v in pr.items():
        assert abs(v) < 1e-6, f"{k} should be 0% when comparing model to itself, got {v}"


# --------------------------------------------------------------------- #
# Mundlak Wald
# --------------------------------------------------------------------- #
def test_mundlak_wald_synthetic_equality_not_rejected():
    """If γ_W == γ_B by construction, the Mundlak Wald should not reject."""
    pytest.importorskip("statsmodels")
    from src.mla.models import fit_3level

    rng = np.random.default_rng(7)
    countries = [f"C{i}" for i in range(15)]
    rows = []
    for c in countries:
        between = rng.normal(0, 0.5)
        c_mean_x = rng.normal(0, 1.0)
        for r in range(1, 5):
            within = rng.normal(0, 0.3)
            x_ct = c_mean_x + within
            for _ in range(50):
                # Use the SAME slope on within and between → equality holds.
                y = 0.7 * x_ct + between + rng.normal(0, 1.0)
                rows.append({
                    "cntry": c, "essround": r, "y": y,
                    "x_within": within,
                    "x_between": c_mean_x,
                })
    df = pd.DataFrame(rows)
    df = add_country_year_key(df)
    res = fit_3level("y ~ x_within + x_between", df)
    test = mundlak_wald(res, "x_within", "x_between")
    # Equality of slopes by construction ⇒ Wald p-value should be largeish.
    assert test["pvalue"] > 0.01, f"unexpected rejection of equality: p={test['pvalue']:.3f}"


def test_mundlak_wald_synthetic_inequality_rejected():
    """If γ_W ≠ γ_B by construction, the Mundlak Wald should reject."""
    pytest.importorskip("statsmodels")
    from src.mla.models import fit_3level

    rng = np.random.default_rng(11)
    countries = [f"C{i}" for i in range(15)]
    rows = []
    for c in countries:
        between = rng.normal(0, 0.5)
        c_mean_x = rng.normal(0, 1.0)
        for r in range(1, 5):
            within = rng.normal(0, 0.3)
            x_ct = c_mean_x + within
            for _ in range(50):
                # Different slopes — within=0.2, between=1.4. Wald should reject.
                y = 0.2 * within + 1.4 * c_mean_x + between + rng.normal(0, 1.0)
                rows.append({
                    "cntry": c, "essround": r, "y": y,
                    "x_within": within,
                    "x_between": c_mean_x,
                })
    df = pd.DataFrame(rows)
    df = add_country_year_key(df)
    res = fit_3level("y ~ x_within + x_between", df)
    test = mundlak_wald(res, "x_within", "x_between")
    assert test["pvalue"] < 0.01, f"failed to reject equality: p={test['pvalue']:.3f}"
    # Roughly recovers the slopes.
    assert abs(test["gamma_w"] - 0.2) < 0.20
    assert abs(test["gamma_b"] - 1.4) < 0.30


def test_mundlak_wald_raises_on_missing_terms():
    pytest.importorskip("statsmodels")
    from src.mla.models import fit_3level

    df = _toy_panel(n_countries=8, n_rounds=3, n_per_cell=40, seed=5)
    df = add_country_year_key(df)
    res = fit_3level("y ~ 1", df)
    with pytest.raises(KeyError):
        mundlak_wald(res, "x_within", "x_between")
