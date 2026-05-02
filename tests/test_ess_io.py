"""Tests for src.mla.ess_io loader paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat
import pytest

from src.mla.ess_io import (
    TEACHING_COLUMNS,
    COURSE_STATA_PATH,
    find_round_file,
    harmonise_columns,
    load_round_from_file,
    load_teaching_panel,
    load_full_panel,
    select_analysis_columns,
)


# --------------------------------------------------------------------- #
# Teaching panel: shape and columns we know
# --------------------------------------------------------------------- #
# The teaching DTA lives under KUL_MultilevelAnalysis_26/, which is gitignored.
# When run in CI or a fresh clone the file is absent → skip.
_COURSE_DTA_MISSING = not COURSE_STATA_PATH.exists()


@pytest.mark.skipif(
    _COURSE_DTA_MISSING,
    reason="course DTA gitignored from repo (KUL_MultilevelAnalysis_26/R/Data/Stata/ESSrounds1to9.dta)",
)
def test_teaching_panel_shape_and_columns():
    df = load_teaching_panel(rounds=(6, 7, 8, 9))
    assert isinstance(df, pd.DataFrame)
    # The lecturer's R6–R9 teaching subset has ~38k rows.
    assert 30_000 <= len(df) <= 50_000
    for col in TEACHING_COLUMNS:
        assert col in df.columns, f"teaching panel missing {col}"
    assert set(df["essround"].unique()).issubset({6, 7, 8, 9})


@pytest.mark.skipif(
    _COURSE_DTA_MISSING,
    reason="course DTA gitignored from repo",
)
def test_teaching_panel_round_filter_disjoint():
    df68 = load_teaching_panel(rounds=(6, 8))
    df79 = load_teaching_panel(rounds=(7, 9))
    assert set(df68["essround"].unique()) == {6, 8}
    assert set(df79["essround"].unique()) == {7, 9}


def test_teaching_panel_missing_file(tmp_path: Path):
    bogus = tmp_path / "does_not_exist.dta"
    with pytest.raises(FileNotFoundError):
        load_teaching_panel(path=bogus)


@pytest.mark.skipif(
    _COURSE_DTA_MISSING,
    reason="course DTA gitignored from repo",
)
def test_harmonise_columns_renames_country():
    df = load_teaching_panel(rounds=(6,))
    out = harmonise_columns(df)
    assert "cntry" in out.columns
    assert "Country" not in out.columns


# --------------------------------------------------------------------- #
# Per-round file discovery
# --------------------------------------------------------------------- #
def test_find_round_file_returns_none_when_dir_missing(tmp_path: Path):
    assert find_round_file(11, raw_dir=tmp_path / "nope") is None


def test_find_round_file_picks_highest_edition(tmp_path: Path):
    """When multiple edition files exist, find_round_file should return the
    alphabetically-last (i.e. typically highest edition)."""
    raw = tmp_path / "ess"
    raw.mkdir()
    (raw / "ESS11e01_0.sav").write_bytes(b"x")
    (raw / "ESS11e04_0.sav").write_bytes(b"x")
    chosen = find_round_file(11, raw_dir=raw)
    assert chosen is not None
    assert chosen.name == "ESS11e04_0.sav"


def test_load_round_from_file_synthetic_sav(tmp_path: Path):
    """Round-trip: write a tiny synthetic SAV with the analysis schema and
    confirm the loader picks it up and tags essround correctly."""
    raw = tmp_path / "ess"
    raw.mkdir()
    syn = pd.DataFrame(
        {
            "cntry": ["BE", "DE", "FR"],
            "idno": [1, 2, 3],
            "trstprl": [5.0, 6.0, 7.0],
            "isco08": [2512, 2511, 7411],
            "agea": [40, 50, 30],
            "pspwght": [1.0, 1.1, 0.9],
        }
    )
    sav_path = raw / "ESS11.sav"
    pyreadstat.write_sav(syn, str(sav_path))

    df = load_round_from_file(
        11,
        raw_dir=raw,
        columns=("cntry", "idno", "trstprl", "isco08", "agea", "pspwght"),
    )
    assert df is not None
    assert len(df) == 3
    assert set(df.columns) == {
        "cntry", "idno", "trstprl", "isco08", "agea", "pspwght", "essround",
    }
    assert df["essround"].dtype == "Int8"
    assert (df["essround"] == 11).all()


def test_load_round_from_file_synthetic_parquet(tmp_path: Path):
    """Parquet path: write a synthetic parquet matching the API's default
    output format and confirm the loader picks it up and selects only
    requested columns even when others exist in the schema."""
    raw = tmp_path / "ess"
    raw.mkdir()
    syn = pd.DataFrame(
        {
            "cntry": ["BE", "DE"],
            "idno": [1, 2],
            "trstprl": [5.0, 6.0],
            "stfdem": [4.0, 7.0],
            "agea": [40, 50],
            "name": ["alice", "bob"],  # extra column not in CORE_COLUMNS
        }
    )
    syn.to_parquet(raw / "ess11e04_0.parquet", index=False)
    df = load_round_from_file(
        11,
        raw_dir=raw,
        columns=("cntry", "idno", "trstprl", "stfdem", "agea", "isco08"),
    )
    assert df is not None
    # `isco08` was requested but absent from the parquet schema → should be
    # silently dropped (not raise).
    assert "isco08" not in df.columns
    assert "name" not in df.columns
    assert set(df.columns) >= {"cntry", "idno", "trstprl", "stfdem", "agea", "essround"}
    assert (df["essround"] == 11).all()


def test_default_doi_suffix_covers_R6_to_R11():
    from src.mla.ess_io import DEFAULT_DOI_SUFFIX

    for r in (6, 7, 8, 9, 10, 11):
        assert r in DEFAULT_DOI_SUFFIX
        assert DEFAULT_DOI_SUFFIX[r].startswith(f"ess{r}")


# --------------------------------------------------------------------- #
# Full-panel orchestrator
# --------------------------------------------------------------------- #
def test_load_full_panel_raises_when_nothing_available(tmp_path: Path):
    """No SAVs anywhere → RuntimeError with helpful pointer."""
    with pytest.raises(RuntimeError, match="No ESS rounds could be loaded"):
        load_full_panel(rounds=(6, 7), raw_dir=tmp_path / "empty")


def test_load_full_panel_skips_missing_when_not_strict(tmp_path: Path):
    """Mix one present round with several missing → returns the present
    round and prints a missing-rounds warning. Captured stdout asserted."""
    raw = tmp_path / "ess"
    raw.mkdir()
    syn = pd.DataFrame({"cntry": ["BE"], "idno": [1], "trstprl": [5.0]})
    pyreadstat.write_sav(syn, str(raw / "ESS11.sav"))
    df = load_full_panel(
        rounds=(10, 11),
        raw_dir=raw,
        columns=("cntry", "idno", "trstprl"),
        strict=False,
    )
    assert set(df["essround"].unique()) == {11}
    assert len(df) == 1


def test_load_full_panel_strict_raises_on_missing(tmp_path: Path):
    raw = tmp_path / "ess"
    raw.mkdir()
    syn = pd.DataFrame({"cntry": ["BE"], "idno": [1], "trstprl": [5.0]})
    pyreadstat.write_sav(syn, str(raw / "ESS11.sav"))
    with pytest.raises(FileNotFoundError):
        load_full_panel(
            rounds=(10, 11),
            raw_dir=raw,
            columns=("cntry", "idno", "trstprl"),
            strict=True,
        )


# --------------------------------------------------------------------- #
# select_analysis_columns is tolerant of missing cols
# --------------------------------------------------------------------- #
def test_select_analysis_columns_subsets_what_is_present():
    df = pd.DataFrame({"cntry": ["BE"], "idno": [1], "agea": [40]})
    out = select_analysis_columns(df, columns=("cntry", "idno", "trstprl", "agea"))
    assert list(out.columns) == ["cntry", "idno", "agea"]
