"""L3 country-level institutional variables: EPL_v1 and welfare regime.

Both variables are time-invariant in the analysis. EPL is fixed at its
2007 pre-period value so it predates the ESS exposure window. The tables
are embedded in code because:

* EPL_v1 — published OECD figures, ~30 OECD members; small, widely cited.
* Welfare-regime classification — Esping-Andersen tradition, hand-coded.

Both are exposed as plain pandas DataFrames keyed on the ESS ``cntry``
ISO-2 code. Non-OECD ESS countries (most Western Balkans, post-Soviet)
have NaN EPL — the M6 institutional-moderation model excludes them via
listwise deletion, with a footnote in the paper noting the restriction.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------- #
# OECD EPL_v1 (regular contracts), individual dismissals, 2007 baseline
# --------------------------------------------------------------------- #
# Pre-2008 baseline so the index predates the AI exposure period.
# Values are from the OECD Employment Protection Database, Version 1,
# 2007 vintage, scale 0-6 (higher = more strict).
EPL_V1_2007: dict[str, float] = {
    # Continental / Conservative-corporatist
    "AT": 2.37, "BE": 1.73, "CH": 1.60, "DE": 2.85, "FR": 2.39, "NL": 2.72,
    # Nordic / Social-democratic
    "DK": 1.63, "FI": 2.17, "IS": 1.73, "NO": 2.31, "SE": 2.61,
    # Mediterranean
    "ES": 2.36, "GR": 2.80, "IT": 1.77, "PT": 4.17, "CY": pd.NA,  # CY: not in OECD
    # Liberal
    "GB": 1.20, "IE": 1.39,
    # Eastern European OECD members
    "CZ": 3.05, "EE": 1.95, "HU": 1.92, "PL": 2.06, "SI": 2.51, "SK": 1.93,
    # Non-OECD ESS countries — left as NaN; documented limitation.
}


def epl_v1_table() -> pd.DataFrame:
    """Return EPL_v1 (2007) per ESS country code as a 2-column DataFrame.

    Columns: ``cntry``, ``epl_c``. Non-OECD ESS countries are NaN.
    """
    rows = [
        {"cntry": k, "epl_c": float(v) if v is not pd.NA else float("nan")}
        for k, v in EPL_V1_2007.items()
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Welfare-regime classification (Esping-Andersen + extensions)
# --------------------------------------------------------------------- #
# Five-class welfare-state typology used as a country-level control:
# social-democratic, conservative-corporatist, liberal, mediterranean,
# and eastern-european. Western Balkans are assigned to eastern-european
# following the Cerami/Vanhuysse extension; Cyprus is mediterranean.
WELFARE_REGIME: dict[str, str] = {
    # Social-democratic
    "DK": "social-democratic", "FI": "social-democratic",
    "IS": "social-democratic", "NO": "social-democratic",
    "SE": "social-democratic",
    # Conservative-corporatist
    "AT": "conservative-corporatist", "BE": "conservative-corporatist",
    "CH": "conservative-corporatist", "DE": "conservative-corporatist",
    "FR": "conservative-corporatist", "NL": "conservative-corporatist",
    # Liberal
    "GB": "liberal", "IE": "liberal",
    # Mediterranean
    "CY": "mediterranean", "ES": "mediterranean", "GR": "mediterranean",
    "IT": "mediterranean", "PT": "mediterranean",
    # Eastern European (incl. Western Balkans)
    "AL": "eastern-european", "BG": "eastern-european",
    "CZ": "eastern-european", "EE": "eastern-european",
    "HR": "eastern-european", "HU": "eastern-european",
    "LT": "eastern-european", "LV": "eastern-european",
    "ME": "eastern-european", "MK": "eastern-european",
    "PL": "eastern-european", "RS": "eastern-european",
    "SI": "eastern-european", "SK": "eastern-european",
    "UA": "eastern-european", "XK": "eastern-european",
    # Non-European in ESS — left as NaN; not in the European welfare-regime tradition.
    # IL, RU map to NaN below.
}


def welfare_regime_table() -> pd.DataFrame:
    """Return welfare-regime category per ESS country code.

    Columns: ``cntry``, ``welfare_regime``. Israel/Russia → NaN.
    """
    rows = [{"cntry": k, "welfare_regime": v} for k, v in WELFARE_REGIME.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Composite L3 frame
# --------------------------------------------------------------------- #
def build_l3_frame() -> pd.DataFrame:
    """Combine EPL_v1 and welfare regime into one L3 country-level frame."""
    return (
        epl_v1_table()
        .merge(welfare_regime_table(), on="cntry", how="outer")
        .sort_values("cntry")
        .reset_index(drop=True)
    )
