"""Tests for src.mla.mundlak: country-year aggregation + within-between decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mla.mundlak import (
    country_year_aggregate,
    country_year_aggregate_leave_one_out,
    merge_country_year_to_panel,
    within_between_decompose,
)


# --------------------------------------------------------------------- #
# country_year_aggregate
# --------------------------------------------------------------------- #
def _toy_panel() -> pd.DataFrame:
    """Two countries (A, B) × two rounds (1, 2) × ~3 individuals.

    Designed so the unweighted and weighted means differ when weights
    are uneven, allowing the weighting branch to be exercised distinctly.
    """
    return pd.DataFrame(
        [
            {"cntry": "A", "essround": 1, "x": 1.0, "w": 1.0},
            {"cntry": "A", "essround": 1, "x": 3.0, "w": 1.0},
            {"cntry": "A", "essround": 1, "x": 5.0, "w": 3.0},  # heavy
            {"cntry": "A", "essround": 2, "x": 2.0, "w": 1.0},
            {"cntry": "A", "essround": 2, "x": 4.0, "w": 1.0},
            {"cntry": "B", "essround": 1, "x": 10.0, "w": 1.0},
            {"cntry": "B", "essround": 1, "x": 12.0, "w": 1.0},
            {"cntry": "B", "essround": 2, "x": 11.0, "w": 1.0},
            {"cntry": "B", "essround": 2, "x": 13.0, "w": 1.0},
        ]
    )


def test_country_year_aggregate_unweighted():
    panel = _toy_panel()
    cy = country_year_aggregate(panel, "x", weight_col=None)
    cy = cy.sort_values(["cntry", "essround"]).reset_index(drop=True)
    # Expected unweighted means: A1=(1+3+5)/3=3.0, A2=(2+4)/2=3.0,
    # B1=11.0, B2=12.0.
    expected = pd.Series([3.0, 3.0, 11.0, 12.0])
    np.testing.assert_array_almost_equal(cy["x_ct"].to_numpy(), expected.to_numpy())


def test_country_year_aggregate_weighted_differs():
    panel = _toy_panel()
    unw = country_year_aggregate(panel, "x", weight_col=None).set_index(["cntry", "essround"])
    w = country_year_aggregate(panel, "x", weight_col="w").set_index(["cntry", "essround"])
    # A1 weighted = (1·1 + 3·1 + 5·3) / (1+1+3) = (1+3+15)/5 = 3.8.
    # Differs from unweighted 3.0.
    assert pytest.approx(w.loc[("A", 1), "x_ct"]) == 3.8
    assert pytest.approx(unw.loc[("A", 1), "x_ct"]) == 3.0
    # B1 weights uniform → weighted == unweighted.
    assert pytest.approx(w.loc[("B", 1), "x_ct"]) == unw.loc[("B", 1), "x_ct"]


def test_country_year_aggregate_drops_na():
    panel = _toy_panel()
    panel.loc[0, "x"] = np.nan  # one missing value in A1
    cy = country_year_aggregate(panel, "x", weight_col=None)
    a1 = cy.loc[(cy["cntry"] == "A") & (cy["essround"] == 1), "x_ct"].iloc[0]
    # Now A1 = (3 + 5) / 2 = 4.0
    assert pytest.approx(a1) == 4.0


def test_country_year_aggregate_drops_zero_or_nan_weights():
    panel = _toy_panel()
    panel.loc[2, "w"] = 0.0  # the heavy A1 row
    cy = country_year_aggregate(panel, "x", weight_col="w")
    a1 = cy.loc[(cy["cntry"] == "A") & (cy["essround"] == 1), "x_ct"].iloc[0]
    # Weight zero ⇒ row excluded ⇒ A1 = (1 + 3) / 2 = 2.0.
    assert pytest.approx(a1) == 2.0


# --------------------------------------------------------------------- #
# Leave-one-out variant
# --------------------------------------------------------------------- #
def test_loo_unweighted_reduces_each_observation():
    panel = _toy_panel()
    panel = panel.reset_index(drop=True)
    loo = country_year_aggregate_leave_one_out(panel, "x", weight_col=None)
    # A1 has values [1, 3, 5]. Leave-one-out for x=1 ⇒ (3+5)/2 = 4.0.
    a1_idx = panel[(panel["cntry"] == "A") & (panel["essround"] == 1)].index
    expected = {1.0: (3 + 5) / 2, 3.0: (1 + 5) / 2, 5.0: (1 + 3) / 2}
    for idx in a1_idx:
        x = panel.loc[idx, "x"]
        np.testing.assert_almost_equal(loo.loc[idx], expected[x])


def test_loo_returns_nan_for_singleton_cells():
    panel = pd.DataFrame(
        [
            {"cntry": "C", "essround": 9, "x": 7.0, "w": 1.0},  # only one in cell
            {"cntry": "C", "essround": 8, "x": 5.0, "w": 1.0},
            {"cntry": "C", "essround": 8, "x": 6.0, "w": 1.0},
        ]
    )
    loo = country_year_aggregate_leave_one_out(panel, "x", weight_col=None)
    assert pd.isna(loo.iloc[0])
    # The C/8 cell has 2 observations → LOO defined.
    assert pytest.approx(loo.iloc[1]) == 6.0
    assert pytest.approx(loo.iloc[2]) == 5.0


def test_loo_propagates_nan_value():
    panel = _toy_panel()
    panel.loc[0, "x"] = np.nan
    loo = country_year_aggregate_leave_one_out(panel, "x", weight_col=None)
    assert pd.isna(loo.loc[0])


# --------------------------------------------------------------------- #
# within_between_decompose
# --------------------------------------------------------------------- #
def test_within_between_decomposition_identity():
    """The Mundlak identity X_ct == X̄_c + (X_ct − X̄_c) holds exactly."""
    panel = _toy_panel()
    cy = country_year_aggregate(panel, "x", weight_col=None)
    decomp = within_between_decompose(cy, "x_ct")
    sum_ = decomp["x_ct_between"] + decomp["x_ct_within"]
    np.testing.assert_array_almost_equal(decomp["x_ct"].to_numpy(), sum_.to_numpy())


def test_within_between_within_means_zero_per_country():
    """The within component sums to zero within each country."""
    panel = _toy_panel()
    cy = country_year_aggregate(panel, "x", weight_col=None)
    decomp = within_between_decompose(cy, "x_ct")
    # For each country: sum of within-component across rounds == 0.
    sums = decomp.groupby("cntry")["x_ct_within"].sum()
    np.testing.assert_array_almost_equal(sums.to_numpy(), np.zeros(len(sums)))


def test_within_between_between_constant_per_country():
    """The between component is constant within country."""
    panel = _toy_panel()
    cy = country_year_aggregate(panel, "x", weight_col=None)
    decomp = within_between_decompose(cy, "x_ct")
    n_distinct = decomp.groupby("cntry")["x_ct_between"].nunique()
    assert (n_distinct == 1).all()


# --------------------------------------------------------------------- #
# merge_country_year_to_panel
# --------------------------------------------------------------------- #
def test_merge_country_year_preserves_panel_size():
    panel = _toy_panel()
    cy = country_year_aggregate(panel, "x", weight_col=None)
    decomp = within_between_decompose(cy, "x_ct")
    out = merge_country_year_to_panel(panel, decomp)
    assert len(out) == len(panel)
    for col in ("x_ct", "x_ct_between", "x_ct_within"):
        assert col in out.columns


def test_merge_country_year_rejects_duplicate_keys():
    panel = _toy_panel()
    cy = pd.DataFrame(
        {
            "cntry": ["A", "A"],
            "essround": [1, 1],
            "x_ct": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        merge_country_year_to_panel(panel, cy)
