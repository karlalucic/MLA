"""ESS (European Social Survey) loaders for the KUL MLA paper.

**Important discovery on disk inspection (2026-04-25):** the files shipped
with the course materials —
``KUL_MultilevelAnalysis_26/R/Data/Stata/ESSrounds1to9.dta``,
``ESSround8.dta``, and the corresponding ``.Rdata`` files — are *teaching
subsets* with only ~11 columns (Country, essround, ais, res_mig, res_educ,
lrscale, gndr, agea, gdppc/unemp/infl interpolated). They lack ``idno``,
survey weights, trust outcomes (``trstprl``, ``trstlgl``, ``stfdem``),
occupation codes (``isco08``), education (``eisced``), income
(``hinctnta``), main activity (``mnactic``), and most other ESS variables
needed for the paper analysis.

Therefore the analysis must rely on the **integrated SAVs** from ESS Sikt
(R6 through R11). Two acquisition paths:

1. **Manual fallback (always works)** — register at
   https://ess.sikt.no/, download the integrated SAV file for each round
   you need, and place it in ``data/raw/ess/`` (e.g. ``ESS6e02_6.sav``,
   ``ESS11e04_0.sav``). The loader auto-detects.
2. **ESS API at https://api.ess.sikt.no/** — credential ``ESS_USER_ID``
   loaded from ``.env`` via ``python-dotenv``. The API is in beta as of
   April 2026; if its SAV download endpoint is not yet stable, the
   loader raises with a clear pointer to path 1.

The course teaching subset is still useful for replicating lecture
exercises — see :func:`load_teaching_panel`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyreadstat
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_STATA_PATH = (
    REPO_ROOT
    / "KUL_MultilevelAnalysis_26"
    / "R"
    / "Data"
    / "Stata"
    / "ESSrounds1to9.dta"
)
RAW_ESS_DIR = REPO_ROOT / "data" / "raw" / "ess"

# Core analysis columns expected in the integrated ESS files.
# Outcomes + weights + keys + L1 controls.
CORE_COLUMNS: tuple[str, ...] = (
    # keys
    "cntry", "essround", "idno",
    # weights (pre-R9 needs pspwght*pweight; R9+ gives anweight)
    "pspwght", "pweight",
    # primary outcomes
    "trstprl", "trstlgl", "stfdem",
    # additional trust items (sensitivity / robustness)
    "trstplt", "trstplc", "trstep", "trstun",
    # L1 controls
    "isco08", "eisced", "hinctnta",
    "agea", "gndr", "mnactic", "domicil",
    # extras occasionally useful in the model or robustness
    "lrscale", "polintr",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("anweight",)
TEACHING_COLUMNS: tuple[str, ...] = (
    "Country", "essround", "ais", "res_mig", "res_educ",
    "lrscale", "gndr", "agea",
    "gdppc_inter", "unemp_rate_inter", "infl_for_perc_inter",
)


# --------------------------------------------------------------------- #
# Course teaching subset (narrow; for exercise replication only)
# --------------------------------------------------------------------- #
def load_teaching_panel(
    path: Path = COURSE_STATA_PATH,
    rounds: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load the narrow teaching subset of ESS R1–R9 from the course materials.

    Returns the lecturer's prepared dataset (anti-immigrant sentiment
    framing). Useful for replicating exercises 4, 6, 12, 13 from the
    course; **not sufficient for the paper analysis** (lacks idno, weights,
    trust outcomes, isco08, eisced, hinctnta, mnactic, domicil).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected course teaching file at {path}. Verify the "
            "course materials (KUL_MultilevelAnalysis_26/) are present."
        )
    df, _meta = pyreadstat.read_dta(str(path), disable_datetime_conversion=True)
    if rounds is not None:
        df = df[df["essround"].isin(list(rounds))].copy()
    df["essround"] = df["essround"].astype("Int8")
    return df


# --------------------------------------------------------------------- #
# Per-round integrated SAV/DTA in data/raw/ess/
# --------------------------------------------------------------------- #
_ROUND_FILE_PATTERNS: dict[int, list[str]] = {
    n: [
        f"ESS{n}.sav", f"ESS{n}.dta",
        f"ESS{n}e*.sav", f"ESS{n}e*.dta",
        f"ess{n}.sav", f"ess{n}.dta",
    ]
    for n in range(1, 12)
}


def find_round_file(round_num: int, raw_dir: Path = RAW_ESS_DIR) -> Path | None:
    """Glob ``data/raw/ess/`` for a SAV or DTA file matching round ``round_num``.

    Returns the alphabetically-last match (typically the highest edition)
    or ``None`` if nothing matches.
    """
    if not raw_dir.exists():
        return None
    candidates: list[Path] = []
    for pat in _ROUND_FILE_PATTERNS.get(round_num, []):
        candidates.extend(raw_dir.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _read_ess_file(path: Path, columns: Iterable[str] | None) -> pd.DataFrame:
    keep = list(columns) if columns is not None else None
    if path.suffix.lower() == ".sav":
        df, _meta = pyreadstat.read_sav(
            str(path), usecols=keep, disable_datetime_conversion=True
        )
    elif path.suffix.lower() == ".dta":
        df, _meta = pyreadstat.read_dta(
            str(path), usecols=keep, disable_datetime_conversion=True
        )
    else:
        raise ValueError(f"Unsupported ESS file extension: {path}")
    return df


def load_round_from_file(
    round_num: int,
    raw_dir: Path = RAW_ESS_DIR,
    columns: Iterable[str] | None = CORE_COLUMNS,
) -> pd.DataFrame | None:
    """Load a single ESS round from ``data/raw/ess/<pattern>``. ``None`` if absent.

    The integrated ESS SAV files have lowercase ``cntry`` (not the course
    file's capitalised ``Country``). We do not auto-rename, to avoid
    silently masking schema mismatches. Use :func:`harmonise_columns` if
    needed downstream.
    """
    path = find_round_file(round_num, raw_dir)
    if path is None:
        return None
    df = _read_ess_file(path, columns)
    if "essround" not in df.columns:
        df["essround"] = round_num
    df["essround"] = df["essround"].astype("Int8")
    return df


# --------------------------------------------------------------------- #
# ESS API at https://api.ess.sikt.no/  (beta as of April 2026)
# --------------------------------------------------------------------- #
ESS_API_BASE = "https://api.ess.sikt.no"


def fetch_round_via_api(
    round_num: int,
    user_id: str | None = None,
    dest_dir: Path = RAW_ESS_DIR,
    timeout: float = 60.0,
) -> Path:
    """Try to download an ESS round SAV file via the ESS API.

    The API is in beta; the SAV download endpoint may not be stable.
    We probe a small set of plausible URL shapes; if none deliver an SAV,
    we raise with a pointer to the manual fallback path.
    """
    import requests  # local import keeps top-level optional

    load_dotenv(REPO_ROOT / ".env")
    user_id = user_id or os.environ.get("ESS_USER_ID")
    if not user_id:
        raise RuntimeError(
            "ESS_USER_ID not set. Either:\n"
            "  (a) register at https://ess.sikt.no/, copy your user id into .env\n"
            "      as ESS_USER_ID=<id>, then retry; or\n"
            "  (b) drop ESS{round}.sav into data/raw/ess/ manually."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"ESS{round_num}.sav"

    # Refine these once https://api.ess.sikt.no/docs solidifies the schema.
    candidate_urls = [
        f"{ESS_API_BASE}/v1/datasets/ess{round_num}/file/spss",
        f"{ESS_API_BASE}/datasets/ess-{round_num}/spss",
        f"{ESS_API_BASE}/integrated-files/ess{round_num}/sav",
    ]
    headers = {"X-ESS-User-Id": user_id, "Accept": "application/octet-stream"}

    last_err: Exception | None = None
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            ct = r.headers.get("content-type", "")
            if r.status_code == 200 and ct.startswith(
                ("application/octet-stream", "application/x-spss")
            ):
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                return dest
        except requests.RequestException as e:
            last_err = e

    raise RuntimeError(
        f"ESS API did not return a SAV for round {round_num} on any probed "
        f"endpoint. The beta API at {ESS_API_BASE}/docs may not yet expose "
        f"SAV download.\n"
        f"Manual fallback: download ESS{round_num} integrated SAV from "
        f"https://ess.sikt.no/ and place at {dest}.\n"
        f"Last network error (if any): {last_err!r}"
    )


# --------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------- #
def load_full_panel(
    rounds: Iterable[int] = (6, 7, 8, 9, 10, 11),
    columns: Iterable[str] | None = CORE_COLUMNS,
    raw_dir: Path = RAW_ESS_DIR,
    try_api: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Load and concatenate the integrated ESS rounds into one long DataFrame.

    Resolution per round:

    1. If a SAV/DTA file matching the round exists in ``data/raw/ess/``,
       load it.
    2. Else if ``try_api=True``, attempt fetch via :func:`fetch_round_via_api`.
    3. Else record the round as missing.

    With ``strict=False`` (default), missing rounds are skipped with a
    warning. Allows the pipeline to develop end-to-end on whatever rounds
    are available while the others are being acquired.
    """
    rounds = list(rounds)
    frames: list[pd.DataFrame] = []
    missing: list[int] = []

    for r in rounds:
        df = load_round_from_file(r, raw_dir=raw_dir, columns=columns)
        if df is None and try_api:
            try:
                fetch_round_via_api(r, dest_dir=raw_dir)
                df = load_round_from_file(r, raw_dir=raw_dir, columns=columns)
            except Exception as e:  # noqa: BLE001
                if strict:
                    raise
                print(f"[ess_io] API fetch failed for R{r}: {e}")
        if df is None:
            missing.append(r)
            continue
        frames.append(df)

    if missing:
        msg = (
            f"[ess_io] Missing rounds: {missing}. "
            f"Drop ESS{{N}}.sav into {raw_dir} or set try_api=True with "
            f"ESS_USER_ID in .env."
        )
        if strict:
            raise FileNotFoundError(msg)
        print(msg)

    if not frames:
        raise RuntimeError(
            "No ESS rounds could be loaded. Place integrated SAV files in "
            f"{raw_dir} (e.g. ESS6e02_6.sav, …, ESS11e04_0.sav) or supply "
            "ESS_USER_ID in .env and pass try_api=True."
        )

    out = pd.concat(frames, axis=0, ignore_index=True)
    out["essround"] = out["essround"].astype("Int8")
    return out


def select_analysis_columns(
    df: pd.DataFrame, columns: Iterable[str] = CORE_COLUMNS
) -> pd.DataFrame:
    """Restrict a wide ESS frame to the modeling columns; tolerate missing optional cols."""
    have = [c for c in columns if c in df.columns]
    return df[have].copy()


def harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Light-touch harmonisation between the course teaching subset and the
    integrated ESS files (e.g. ``Country`` → ``cntry``)."""
    out = df.rename(columns={"Country": "cntry"})
    return out
