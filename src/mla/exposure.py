"""AI / GenAI occupation-exposure score loaders and ESS merge helpers.

The primary exposure measure is the ILO-NASK occupation score from
Gmyrek et al. (2025), keyed directly on ISCO-08 4-digit occupations and
available in two score vintages:

* ``score_2023`` — applied to ESS R6–R10 (pre-GPT-4 era).
* ``score_2025`` — applied to ESS R11 (post-GPT-4 era).

Alternative exposure measures such as Webb (2020), Felten et al.
(2021/2023), and Eloundou et al. (2024) are SOC-keyed and require an
ISCO-08 to SOC-2010 crosswalk. They are not implemented in this
repository's reproducible pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "data" / "raw" / "scores"

ILO_NASK_XLSX = SCORES_DIR / "Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx"

# ESS round -> ILO-NASK score vintage. R6-R10 use the 2023 score to
# represent the pre-GPT-4 period; R11 uses the 2025 score.
VINTAGE_FOR_ROUND: dict[int, int] = {
    6: 2023,
    7: 2023,
    8: 2023,
    9: 2023,
    10: 2023,
    11: 2025,
}


# --------------------------------------------------------------------- #
# Primary: ILO–NASK 2025 (Gmyrek et al.) — both 2023 and 2025 vintages
# --------------------------------------------------------------------- #
def load_ilo_nask_scores(
    xlsx_path: Path = ILO_NASK_XLSX,
) -> pd.DataFrame:
    """Load occupation-level ILO–NASK GenAI exposure scores per ISCO-08 4-digit.

    Returns one row per ISCO-08 4-digit occupation with columns:

    * ``isco08``   — int, 4-digit code (e.g. 1111, 2512, 7411).
    * ``title``    — occupation title.
    * ``score_2023`` — 2023 GBB vintage mean automation score (0–1).
    * ``score_2025`` — 2025 ILO–NASK vintage mean automation score (0–1).
    * ``sd_2023``, ``sd_2025`` — within-occupation SD across tasks.

    The Excel file is task-level (~3000 rows); we deduplicate to the
    occupation-mean by taking the first observation per ``ISCO_08`` since
    ``mean_score_2023`` / ``mean_score_2025`` are constant per occupation.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"ILO–NASK score file not found at {xlsx_path}. Run "
            f"`src/mla/exposure.py` notebook 02_occupation_scores.ipynb to "
            f"download from "
            f"github.com/pgmyrek/2025_GenAI_scores_ISCO08."
        )
    raw = pd.read_excel(xlsx_path)
    # Pick the columns we need *first*, then rename — the workbook also has
    # a task-level `score_2023` that would collide with the renamed
    # occupation-mean column otherwise.
    src_cols = ["ISCO_08", "Title", "mean_score_2023", "mean_score_2025", "SD_2023", "SD_2025"]
    rename = {
        "ISCO_08": "isco08",
        "Title": "title",
        "mean_score_2023": "score_2023",
        "mean_score_2025": "score_2025",
        "SD_2023": "sd_2023",
        "SD_2025": "sd_2025",
    }
    out = (
        raw.loc[:, src_cols]
        .rename(columns=rename)
        .drop_duplicates(subset=["isco08"])
        .reset_index(drop=True)
    )
    out["isco08"] = out["isco08"].astype("Int32")
    for c in ("score_2023", "score_2025", "sd_2023", "sd_2025"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


# --------------------------------------------------------------------- #
# Vintage application: build per-individual genai_i and genai_i_static
# --------------------------------------------------------------------- #
def assign_vintage_score(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    vintage_for_round: dict[int, int] | None = None,
    isco_col: str = "isco08",
    round_col: str = "essround",
    out_col: str = "genai_i",
    static_vintage: int | None = None,
    static_out_col: str = "genai_i_static",
) -> pd.DataFrame:
    """Merge ILO–NASK scores into the ESS panel under the vintage rule.

    Adds two columns:

    * ``genai_i``         — vintage-assigned per ``vintage_for_round``
      (default :data:`VINTAGE_FOR_ROUND`: 2023 for R6–R10, 2025 for R11).
    * ``genai_i_static``  — score from a single fixed vintage applied
      uniformly across all rounds (default 2025); for the Section-7
      robustness "vintage held constant" comparator.

    Rows whose ISCO-08 code is not in the score table get NaN. The merge
    is left-join to preserve panel size; report nulls downstream.
    """
    vintage_for_round = vintage_for_round or VINTAGE_FOR_ROUND
    static_vintage = static_vintage or 2025

    score_2023 = scores.set_index("isco08")["score_2023"]
    score_2025 = scores.set_index("isco08")["score_2025"]

    # Map (round → score column) once.
    round_to_year = pd.Series(vintage_for_round, name="vintage")

    out = panel.copy()
    out[isco_col] = out[isco_col].astype("Int32")

    # Per-row vintage selection
    vintages = out[round_col].map(round_to_year)
    s2023 = out[isco_col].map(score_2023)
    s2025 = out[isco_col].map(score_2025)
    out[out_col] = np.where(vintages == 2023, s2023, np.where(vintages == 2025, s2025, np.nan))

    # Static comparator
    static_map = score_2025 if static_vintage == 2025 else score_2023
    out[static_out_col] = out[isco_col].map(static_map)

    return out


def report_score_coverage(panel: pd.DataFrame, score_col: str = "genai_i") -> pd.DataFrame:
    """Diagnostic: per-round share of individuals with a non-null exposure score.

    Useful to surface ISCO-08 coding gaps (e.g. ``isco08 == 9999`` or
    out-of-range codes that don't appear in the ILO–NASK table).
    """
    g = panel.groupby("essround")[score_col]
    out = pd.DataFrame(
        {
            "n_individuals": g.size(),
            "n_with_score": g.count(),
        }
    )
    out["share_with_score"] = out["n_with_score"] / out["n_individuals"]
    return out


# --------------------------------------------------------------------- #
# Alternative exposure measures requiring an ISCO-SOC crosswalk.
# --------------------------------------------------------------------- #
def load_robustness_scores(
    measures: Iterable[str] = ("webb", "aioe", "felten_genai", "eloundou"),
) -> pd.DataFrame:
    """Load robustness exposure measures keyed on ISCO-08 4-digit.

    These measures are not part of the submitted replication pipeline.
    Implementing them requires downloading SOC-keyed source files,
    applying an ISCO-08 <-> SOC-2010 crosswalk, and aggregating many-to-
    many matches to ISCO-08 4-digit occupations.
    """
    raise NotImplementedError(
        "Alternative SOC-keyed exposure measures are not implemented in "
        "this repository. The implemented primary measure is ILO-NASK."
    )
