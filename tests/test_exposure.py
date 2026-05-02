"""Tests for src.mla.exposure: ILO–NASK loader + vintage-rule merge."""

from __future__ import annotations


import pandas as pd
import pytest

from src.mla.exposure import (
    ILO_NASK_XLSX,
    VINTAGE_FOR_ROUND,
    assign_vintage_score,
    load_ilo_nask_scores,
    load_robustness_scores,
    report_score_coverage,
)


# --------------------------------------------------------------------- #
# load_ilo_nask_scores: shape, schema, sentinel values
# --------------------------------------------------------------------- #
@pytest.mark.skipif(
    not ILO_NASK_XLSX.exists(),
    reason="ILO–NASK xlsx not yet downloaded; run notebooks/02_occupation_scores.ipynb",
)
def test_ilo_nask_shape_and_columns():
    s = load_ilo_nask_scores()
    assert isinstance(s, pd.DataFrame)
    # 427 occupations observed; allow a band so future editions don't break the test.
    assert 400 <= len(s) <= 450
    expected = {"isco08", "title", "score_2023", "score_2025", "sd_2023", "sd_2025"}
    assert set(s.columns) == expected
    # ISCO-08 4-digit codes are in [1000, 9999].
    assert s["isco08"].between(1000, 9999).all()
    # Scores in [0, 1] per the published methodology.
    for c in ("score_2023", "score_2025"):
        assert s[c].between(0.0, 1.0).all()
    # No duplicates after dedup.
    assert s["isco08"].is_unique


@pytest.mark.skipif(
    not ILO_NASK_XLSX.exists(),
    reason="ILO–NASK xlsx not yet downloaded",
)
def test_ilo_nask_published_summary_stats():
    """Sanity check against the plan §2 published statistics:
    2023 mean 0.30 / sd ~0.16; 2025 mean 0.29 / sd ~0.14."""
    s = load_ilo_nask_scores()
    assert abs(s["score_2023"].mean() - 0.30) < 0.05
    assert abs(s["score_2025"].mean() - 0.29) < 0.05
    assert abs(s["score_2025"].std() - 0.145) < 0.03


# --------------------------------------------------------------------- #
# Vintage application logic
# --------------------------------------------------------------------- #
def _synthetic_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isco08": [1111, 2512, 7411],
            "title": ["a", "b", "c"],
            "score_2023": [0.10, 0.50, 0.30],
            "score_2025": [0.20, 0.65, 0.30],
            "sd_2023": [0.05, 0.10, 0.08],
            "sd_2025": [0.04, 0.09, 0.08],
        }
    )


def _synthetic_panel() -> pd.DataFrame:
    """One individual per (round, isco08) cell; a few rounds and codes."""
    rows = []
    for rnd in (6, 7, 9, 10, 11):
        for code in (1111, 2512, 7411, 9999):  # 9999 = unmatched on purpose
            rows.append({"essround": rnd, "isco08": code, "idno": rnd * 10000 + code})
    return pd.DataFrame(rows)


def test_assign_vintage_score_rule_applied_correctly():
    panel = _synthetic_panel()
    scores = _synthetic_scores()
    out = assign_vintage_score(panel, scores)
    # R6–R10 use score_2023 → ISCO 2512 should be 0.50.
    pre = out[(out["essround"] != 11) & (out["isco08"] == 2512)]
    assert (pre["genai_i"] == 0.50).all()
    # R11 uses score_2025 → ISCO 2512 should be 0.65.
    post = out[(out["essround"] == 11) & (out["isco08"] == 2512)]
    assert (post["genai_i"] == 0.65).all()
    # The static comparator is 2025 throughout.
    static_2512 = out[out["isco08"] == 2512]
    assert (static_2512["genai_i_static"] == 0.65).all()
    # Unmatched ISCO 9999 → NaN in both columns.
    unmatched = out[out["isco08"] == 9999]
    assert unmatched["genai_i"].isna().all()
    assert unmatched["genai_i_static"].isna().all()


def test_assign_vintage_score_preserves_panel_size():
    panel = _synthetic_panel()
    scores = _synthetic_scores()
    out = assign_vintage_score(panel, scores)
    assert len(out) == len(panel)
    assert set(out["idno"]) == set(panel["idno"])


def test_assign_vintage_score_custom_static_vintage():
    """When `static_vintage=2023`, genai_i_static should match score_2023."""
    panel = _synthetic_panel()
    scores = _synthetic_scores()
    out = assign_vintage_score(panel, scores, static_vintage=2023)
    matched = out[out["isco08"] == 2512]
    assert (matched["genai_i_static"] == 0.50).all()


def test_vintage_for_round_table_matches_plan():
    """Plan §2: 2023 GBB for R6–R10, 2025 ILO–NASK for R11."""
    for r in range(6, 11):
        assert VINTAGE_FOR_ROUND[r] == 2023, f"R{r} should use 2023 vintage"
    assert VINTAGE_FOR_ROUND[11] == 2025


def test_report_score_coverage_shape():
    panel = _synthetic_panel()
    scores = _synthetic_scores()
    out = assign_vintage_score(panel, scores)
    cov = report_score_coverage(out, "genai_i")
    assert set(cov.columns) == {"n_individuals", "n_with_score", "share_with_score"}
    # 4 individuals per round; 1 unmatched (9999) → share 0.75.
    assert (cov["share_with_score"] == 0.75).all()


# --------------------------------------------------------------------- #
# Robustness path is stubbed for Day 2 Track B
# --------------------------------------------------------------------- #
def test_load_robustness_scores_not_implemented():
    with pytest.raises(NotImplementedError):
        load_robustness_scores()
