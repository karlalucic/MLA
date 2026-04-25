"""Eurostat macro-panel fetcher (GDP growth, unemployment, HICP inflation).

Uses the public REST API at
``https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/<code>``
(no authentication required). Returns long-format frames keyed on
``(cntry, year)`` ready to merge onto the ESS panel.

The three indicators selected match :doc:`/IMPLEMENTATION_PLAN` §5:

* :func:`fetch_gdp_growth`     — real GDP growth rate (% change on prev. year)
* :func:`fetch_unemployment`   — annual unemployment rate (% of active pop.)
* :func:`fetch_hicp_inflation` — annual avg HICP rate of change (%)

Outputs are cached under ``data/raw/eurostat/`` to avoid re-hitting the
API on rerun. Use ``force_refresh=True`` to bypass the cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
EUROSTAT_DIR = REPO_ROOT / "data" / "raw" / "eurostat"
EUROSTAT_BASE = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
)

# ESS uses ISO 3166-1 alpha-2; Eurostat uses some quirky codes for Greece
# and the UK. Translate both ways. Countries not in Eurostat (IL, RU) are
# returned as missing rows; downstream code handles this via listwise
# deletion in the macro-controlled models.
ESS_TO_EUROSTAT: dict[str, str] = {"GB": "UK", "GR": "EL"}
EUROSTAT_TO_ESS: dict[str, str] = {v: k for k, v in ESS_TO_EUROSTAT.items()}


def _ess_to_eurostat_geos(geos: Iterable[str]) -> list[str]:
    return [ESS_TO_EUROSTAT.get(g, g) for g in geos]


def _restore_ess_geos(df: pd.DataFrame, col: str = "cntry") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].replace(EUROSTAT_TO_ESS)
    return df


# --------------------------------------------------------------------- #
# Generic helper: hit the API and parse JSON-stat 2.0 → long DataFrame
# --------------------------------------------------------------------- #
def _fetch_eurostat_dataset(
    code: str,
    params: dict[str, Any],
    *,
    cache_dir: Path = EUROSTAT_DIR,
    force_refresh: bool = False,
    timeout: float = 60.0,
) -> dict:
    """GET an Eurostat dataset as JSON-stat 2.0; cache the raw JSON to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Build a short, deterministic cache filename from a hash of params.
    canonical = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"{code}__{digest}.json"
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())
    r = requests.get(f"{EUROSTAT_BASE}/{code}", params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    cache_path.write_text(json.dumps(payload))
    return payload


def _jsonstat_to_long(payload: dict, value_name: str = "value") -> pd.DataFrame:
    """Convert JSON-stat 2.0 to a long DataFrame.

    Each cell index in `value` decodes to a coordinate via the
    `dimension`/`size` fields. Returns a DataFrame with one column per
    dimension + one column ``value_name``.
    """
    dims: list[str] = payload["id"]
    sizes: list[int] = payload["size"]
    # Per-dimension category index → category code.
    cat_codes: dict[str, list[str]] = {}
    for dim in dims:
        cat = payload["dimension"][dim]["category"]
        # `index` maps category code → position; we want position → code.
        idx_map = cat["index"]
        if isinstance(idx_map, dict):
            ordered = sorted(idx_map.items(), key=lambda kv: kv[1])
            cat_codes[dim] = [k for k, _ in ordered]
        else:  # list-style index
            cat_codes[dim] = list(idx_map)
    values = payload["value"]  # dict {flat_index_str: number} OR list

    rows: list[dict[str, Any]] = []
    if isinstance(values, dict):
        items = ((int(k), v) for k, v in values.items())
    else:
        items = ((i, v) for i, v in enumerate(values) if v is not None)
    for flat_idx, val in items:
        coords: dict[str, Any] = {}
        rem = flat_idx
        for dim, n in zip(reversed(dims), reversed(sizes)):
            coords[dim] = cat_codes[dim][rem % n]
            rem //= n
        coords[value_name] = val
        rows.append(coords)
    out = pd.DataFrame(rows)
    return out[dims + [value_name]]


# --------------------------------------------------------------------- #
# GDP growth: nama_10_gdp, B1GQ at chain-linked volumes, % change on prev. year
# --------------------------------------------------------------------- #
def fetch_gdp_growth(
    geos: Iterable[str] | None = None,
    year_min: int = 2010,
    year_max: int = 2024,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Annual real GDP growth rate (%) per country-year.

    Returns a long DataFrame ``[cntry, year, gdp_growth]``.
    """
    params: dict[str, Any] = {
        "format": "JSON",
        "lang": "en",
        "na_item": "B1GQ",
        "unit": "CLV_PCH_PRE",
        "time": [str(y) for y in range(year_min, year_max + 1)],
    }
    if geos:
        params["geo"] = _ess_to_eurostat_geos(geos)
    payload = _fetch_eurostat_dataset(
        "nama_10_gdp", params, force_refresh=force_refresh
    )
    df = _jsonstat_to_long(payload, value_name="gdp_growth")
    return _restore_ess_geos(
        df.rename(columns={"geo": "cntry", "time": "year"})
        .loc[:, ["cntry", "year", "gdp_growth"]]
        .assign(year=lambda d: d["year"].astype(int))
    )


# --------------------------------------------------------------------- #
# Unemployment: une_rt_a, total Y15-74, both sexes, % of active population
# --------------------------------------------------------------------- #
def fetch_unemployment(
    geos: Iterable[str] | None = None,
    year_min: int = 2010,
    year_max: int = 2024,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Annual unemployment rate (%) per country-year, ages 15–74, both sexes."""
    params: dict[str, Any] = {
        "format": "JSON",
        "lang": "en",
        "age": "Y15-74",
        "sex": "T",
        "unit": "PC_ACT",
        "time": [str(y) for y in range(year_min, year_max + 1)],
    }
    if geos:
        params["geo"] = _ess_to_eurostat_geos(geos)
    payload = _fetch_eurostat_dataset(
        "une_rt_a", params, force_refresh=force_refresh
    )
    df = _jsonstat_to_long(payload, value_name="unemp_rate")
    return _restore_ess_geos(
        df.rename(columns={"geo": "cntry", "time": "year"})
        .loc[:, ["cntry", "year", "unemp_rate"]]
        .assign(year=lambda d: d["year"].astype(int))
    )


# --------------------------------------------------------------------- #
# HICP inflation: prc_hicp_aind, all-items, annual avg rate of change
# --------------------------------------------------------------------- #
def fetch_hicp_inflation(
    geos: Iterable[str] | None = None,
    year_min: int = 2010,
    year_max: int = 2024,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Annual average HICP rate of change (%) per country-year — i.e. inflation."""
    params: dict[str, Any] = {
        "format": "JSON",
        "lang": "en",
        "coicop": "CP00",
        "unit": "RCH_A_AVG",
        "time": [str(y) for y in range(year_min, year_max + 1)],
    }
    if geos:
        params["geo"] = _ess_to_eurostat_geos(geos)
    payload = _fetch_eurostat_dataset(
        "prc_hicp_aind", params, force_refresh=force_refresh
    )
    df = _jsonstat_to_long(payload, value_name="hicp_inflation")
    return _restore_ess_geos(
        df.rename(columns={"geo": "cntry", "time": "year"})
        .loc[:, ["cntry", "year", "hicp_inflation"]]
        .assign(year=lambda d: d["year"].astype(int))
    )


# --------------------------------------------------------------------- #
# Composite: build the L2 macro panel
# --------------------------------------------------------------------- #
def build_macro_panel(
    geos: Iterable[str] | None = None,
    year_min: int = 2010,
    year_max: int = 2024,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """One-shot helper: returns a wide DataFrame indexed on (cntry, year) with
    columns ``gdp_growth``, ``unemp_rate``, ``hicp_inflation``."""
    g = fetch_gdp_growth(geos=geos, year_min=year_min, year_max=year_max, force_refresh=force_refresh)
    u = fetch_unemployment(geos=geos, year_min=year_min, year_max=year_max, force_refresh=force_refresh)
    h = fetch_hicp_inflation(geos=geos, year_min=year_min, year_max=year_max, force_refresh=force_refresh)
    return (
        g.merge(u, on=["cntry", "year"], how="outer")
        .merge(h, on=["cntry", "year"], how="outer")
        .sort_values(["cntry", "year"])
        .reset_index(drop=True)
    )
