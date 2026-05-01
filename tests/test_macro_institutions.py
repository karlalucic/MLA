"""Tests for src.mla.macro (Eurostat fetcher) and src.mla.institutions
(EPL + welfare regime tables).

The Eurostat tests are kept offline-friendly — we test the JSON-stat
parser on a hand-built payload and verify the code-mapping helpers, but
do not hit the live API in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.mla.institutions import (
    EPL_V1_2007,
    WELFARE_REGIME,
    build_l3_frame,
    epl_v1_table,
    welfare_regime_table,
)
from src.mla.macro import (
    ESS_TO_EUROSTAT,
    EUROSTAT_TO_ESS,
    _ess_to_eurostat_geos,
    _jsonstat_to_long,
    _restore_ess_geos,
)


# --------------------------------------------------------------------- #
# Eurostat code mapping
# --------------------------------------------------------------------- #
def test_ess_to_eurostat_geos_translates_only_special_cases():
    out = _ess_to_eurostat_geos(["BE", "DE", "GR", "GB", "IL"])
    # GR → EL, GB → UK; others passed through.
    assert out == ["BE", "DE", "EL", "UK", "IL"]


def test_eurostat_to_ess_inverse_of_mapping():
    for ess, euro in ESS_TO_EUROSTAT.items():
        assert EUROSTAT_TO_ESS[euro] == ess


def test_restore_ess_geos_reverses_translation():
    df = pd.DataFrame({"cntry": ["BE", "EL", "UK", "IL"], "value": [1, 2, 3, 4]})
    out = _restore_ess_geos(df)
    assert out["cntry"].tolist() == ["BE", "GR", "GB", "IL"]


# --------------------------------------------------------------------- #
# JSON-stat 2.0 parser
# --------------------------------------------------------------------- #
def _toy_jsonstat_payload() -> dict:
    """Build a synthetic JSON-stat payload covering 2 dimensions × 2×3 cells."""
    return {
        "id": ["geo", "time"],
        "size": [2, 3],
        "dimension": {
            "geo": {"category": {"index": {"BE": 0, "DE": 1}}},
            "time": {"category": {"index": {"2018": 0, "2019": 1, "2020": 2}}},
        },
        # Flat indexing: idx = geo*3 + time → (geo, time) pairs.
        # 0 → (BE, 2018), 1 → (BE, 2019), 2 → (BE, 2020),
        # 3 → (DE, 2018), 4 → (DE, 2019), 5 → (DE, 2020).
        "value": {"0": 1.0, "1": 1.5, "2": 2.0, "3": 3.0, "4": 3.5, "5": 4.0},
    }


def test_jsonstat_to_long_decodes_coords_correctly():
    payload = _toy_jsonstat_payload()
    df = _jsonstat_to_long(payload, value_name="x").sort_values(["geo", "time"]).reset_index(drop=True)
    assert list(df.columns) == ["geo", "time", "x"]
    assert len(df) == 6
    # Spot-check a few cells
    cell = df[(df["geo"] == "BE") & (df["time"] == "2018")]["x"].iloc[0]
    assert cell == 1.0
    cell = df[(df["geo"] == "DE") & (df["time"] == "2020")]["x"].iloc[0]
    assert cell == 4.0


def test_jsonstat_to_long_handles_list_value():
    """Variant: some Eurostat responses encode `value` as a list rather than dict."""
    payload = _toy_jsonstat_payload()
    payload["value"] = [1.0, 1.5, 2.0, 3.0, 3.5, 4.0]
    df = _jsonstat_to_long(payload, value_name="x").sort_values(["geo", "time"]).reset_index(drop=True)
    assert df.shape == (6, 3)
    assert df["x"].sum() == 15.0


def test_jsonstat_to_long_skips_none_in_list():
    payload = _toy_jsonstat_payload()
    payload["value"] = [1.0, None, 2.0, 3.0, None, 4.0]
    df = _jsonstat_to_long(payload, value_name="x")
    assert len(df) == 4  # two None cells dropped


def test_jsonstat_to_long_skips_none_in_dict():
    """Regression: dict-form `value` payloads must drop None entries the
    same way list-form does. Pre-fix the dict branch lacked the filter
    and emitted a NaN row instead of silently dropping the cell."""
    payload = _toy_jsonstat_payload()
    payload["value"] = {"0": 1.0, "1": None, "2": 2.0, "3": 3.0, "4": None, "5": 4.0}
    df = _jsonstat_to_long(payload, value_name="x")
    assert len(df) == 4
    assert df["x"].isna().sum() == 0


# --------------------------------------------------------------------- #
# Institutions: EPL + welfare regime
# --------------------------------------------------------------------- #
def test_epl_v1_table_shape_and_values():
    df = epl_v1_table()
    assert set(df.columns) == {"cntry", "epl_c"}
    # All 25 OECD-Europe codes present; values in [0, 6] except NaN.
    assert len(df) == len(EPL_V1_2007)
    valid = df["epl_c"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 6).all()
    # Spot-check: PT is the highest in our set (4.17).
    assert df.loc[df["cntry"] == "PT", "epl_c"].iloc[0] == pytest.approx(4.17)
    # GB liberal regime → low EPL (~1.20).
    assert df.loc[df["cntry"] == "GB", "epl_c"].iloc[0] == pytest.approx(1.20)


def test_welfare_regime_categories_are_the_five_classes():
    df = welfare_regime_table()
    assert set(df.columns) == {"cntry", "welfare_regime"}
    expected = {
        "social-democratic", "conservative-corporatist", "liberal",
        "mediterranean", "eastern-european",
    }
    assert set(df["welfare_regime"].unique()) == expected


def test_welfare_regime_known_assignments():
    wr = WELFARE_REGIME
    assert wr["DK"] == "social-democratic"
    assert wr["DE"] == "conservative-corporatist"
    assert wr["GB"] == "liberal"
    assert wr["IT"] == "mediterranean"
    assert wr["PL"] == "eastern-european"


def test_build_l3_frame_combines_both_tables():
    df = build_l3_frame()
    assert set(df.columns) == {"cntry", "epl_c", "welfare_regime"}
    # Every welfare-regime country appears in the L3 frame (outer join).
    assert set(WELFARE_REGIME.keys()) <= set(df["cntry"].tolist())
    # Eastern-European countries that aren't OECD members → epl_c NaN.
    al = df[df["cntry"] == "AL"]
    assert al["welfare_regime"].iloc[0] == "eastern-european"
    assert pd.isna(al["epl_c"].iloc[0])
