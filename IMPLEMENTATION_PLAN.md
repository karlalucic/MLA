# Multilevel Analysis Final Paper — Implementation Plan

## Context

**Course**: KU Leuven *Multilevel Analysis* (G0W07a), Spring 2026. Lecturer: Prof. Dr. Alexander Schmidt-Catran. Submission: ~12-page paper applying multilevel models to a dataset, with theoretical motivation, hypotheses, and a focus on specification, estimation, and interpretation. Deadlines: 2026-06-20 (2nd exam period) or 2026-08-28 (3rd exam period).

**Why this design**: Schmidt-Catran's signature methodological contribution is the within-between specification for pooled cross-national data (Schmidt-Catran & Fairbrother 2016, *European Sociological Review* 32(1); Bell & Jones 2015, *Political Science Research and Methods*). Mirroring that design exactly maximises rubric fit. The substantive twist — applying it to **GenAI exposure and political trust** rather than the more-trodden anti-immigrant attitudes — adds genuine novelty by extending the robot/automation-exposure literature (Anelli, Colantone & Stanig 2021; Frey, Berger & Chen 2018) to white-collar/services GenAI exposure with the ILO–NASK 2025 occupation-level scores (Gmyrek et al. 2025), which are ISCO-08 native and admit a 2023→2025 vintage shift that injects genuine within-country time variation in $E_o$.

**Approach**: Hybrid framing. Primary outcome is a political-trust composite (α-framing) with full ESS panel coverage. Secondary M5 cross-level interaction adjudicates the γ-style substitution-vs-curator hypothesis (within-country exposure × individual education).

**Stack**: Pure Python (Jupyter for analysis + LaTeX for paper).

---

## 1. Theoretical framing and hypotheses

Three literatures converge:

1. **Performance theories of trust** (Hetherington 1998; van der Meer 2017): institutional trust responds to perceived performance and structural conditions.
2. **Structural-grievance / loser theory** (Gidron & Hall 2017; Kurer 2020): technological change generates losers whose political attitudes shift via grievance.
3. **Robot/automation exposure and political behaviour** (Anelli, Colantone & Stanig 2021; Frey, Berger & Chen 2018; Acemoglu & Restrepo 2020): empirical evidence in single-country and Western-European settings.

Contribution: extend from manufacturing-biased robot exposure to white-collar AI exposure, in a pooled cross-national design that disentangles between-country level effects from within-country shock effects. Add an underspecified-direction education-moderation test (substitution vs. curator) as the secondary contribution.

### Hypotheses (final list to write into the paper)

- **H1 (between)**: Across countries, higher mean AI exposure $\bar E_c$ is associated with lower mean institutional trust. $\gamma_B < 0$.
- **H2 (within)**: Within countries, increases in AI exposure $(E_{ct}-\bar E_c)$ erode institutional trust. $\gamma_W < 0$.
- **H2a (decomposition)**: $\gamma_W \neq \gamma_B$. Mundlak/Wald test.
- **H3 (compositional)**: Individuals in AI-exposed occupations show lower trust net of country context. $\beta_\text{genai} < 0$.
- **H4 (cross-level moderation, γ-flavoured)**: Within-country exposure shock interacts with individual education. **Direction is theoretically underspecified** — the paper pre-specifies three competing outcomes:
  - **H4a substitution**: higher-educated react more strongly (perceive direct competition); negative interaction.
  - **H4b curator**: higher-educated react less strongly (identify with expertise); positive interaction.
  - **H4₀ null / composition-only**: no education-moderation; the within-country exposure effect is uniform across education groups, and any apparent education gradient is compositional. A precise null is itself substantive — it falsifies both substitution and curator framings.
- **H5 (institutional moderation)**: Within-effect is attenuated where employment-protection (EPL) is stronger. Coefficient on $\text{EPL}_c \times (E_{ct}-\bar E_c)$ is positive.

---

## 2. Data architecture

| Layer | Source | Notes |
|---|---|---|
| L1 individual | ESS rounds 6–11 (2012–2024) | R1–R9 already on disk at `KUL_MultilevelAnalysis_26/R/Data/RData/ess1to9.Rdata`. R10 and R11 fetched via the **ESS API** at <https://api.ess.sikt.no/docs> using `ESS_USER_ID` loaded from `.env` via `python-dotenv` (never hardcoded). The API is currently in beta — **fallback**: if it does not yet support SAV downloads, place R10 and R11 SAV files manually in `data/raw/ess/` and the loader skips the API path. R11 final integrated file (Edition 4.0) released November 2025, covers 30 countries with fielding October 2023 through 2024. ISCO-08 occupational coding present from R6 onwards. Survey weights `pspwght` (design) and `pweight` (population) per round. |
| L2 country-year | Country-year aggregated AI exposure (ESS-internal) + Eurostat macro panel | $\bar E_{ct} = \sum_i (\text{pspwght}_{ict}\cdot E_{o(i),v(t)}) / \sum_i \text{pspwght}_{ict}$, computed on ESS individuals i in country c, round t, with $E_{o(i),v(t)}$ = ILO–NASK GenAI exposure score for individual i's ISCO-08 4-digit occupation $o(i)$ at vintage $v(t)$ (vintage assignment rule below). Macro: real GDP growth, unemployment rate, HICP inflation. |
| L3 country | OECD EPL index (pre-period, 2008 baseline), welfare-regime typology (pre-period), optional OECD AI Policy Observatory stringency | Time-invariant moderators; pre-period to limit reverse-causation concerns. |
| Occupation scores — primary | **ILO–NASK 2025 (Gmyrek et al. 2025, ILO Working Paper 140)** | Public dataset at `github.com/pgmyrek/2025_GenAI_scores_ISCO08` (`Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx`). ISCO-08 4-digit native; no SOC crosswalk needed. Task-level scores also available (`4digits_with_tasks.xlsx`). Score range 0–1, methodology combines task-level data, expert review, and AI prediction; 30,000 task scores aggregated to 436 ISCO-08 4-digit occupations. CC BY 4.0 license per ILO publication norms. |
| Occupation scores — 2023 vintage | **GBB 2023 (Gmyrek, Berg & Bescond 2023, ILO Working Paper 96)** | The 2025 paper re-scores the original 2023 ISCO-08 occupations using the refined methodology, so both vintages are derivable from the 2025 release. The 2023 mean automation score is 0.30 (SD 0.30); the 2025 score is 0.29 (SD 0.14), reflecting GenAI capability advances especially in voice, image, and video generation. Used to construct vintage-stratified time variation (vintage rule below). |
| Occupation scores — robustness | Webb (2020) patent-based exposure; Felten et al. 2021 (AIOE); Felten et al. 2023 (GenAI); Eloundou et al. 2024 (GPT-task exposure) | Run as Section-7 robustness checks. Convergent results across measures = strength; divergence is itself reportable as construct-validity evidence. |
| Crosswalks | None needed for primary measure | ILO–NASK is ISCO-08 native. SOC↔ISCO-08 crosswalk only required for Webb / Felten / Eloundou robustness checks; ILO publishes the weighted crosswalk. |
| LFS shares (robustness only) | Eurostat `lfsa_egais` (1-digit ISCO) and `lfsa_egan2` (2-digit) | Eurostat aggregation used only as a robustness check on the country-year exposure construction. ESS-internal aggregation at 4-digit is primary. |
| Missing data | ESS items, especially `hinctnta` (income, ~15–25% missing) | Listwise deletion as primary; multiple imputation by chained equations (`statsmodels.imputation.mice`) as robustness if effective N drops materially. |

**Vintage assignment rule.** Each ESS round is paired with the ILO–NASK score vintage that best reflects publicly available GenAI capabilities at the time the round was fielded:

- ESS R6 (2012) through R10 (fielded 2020–2022): paired with **2023 GBB scores** (representing pre-ChatGPT through GPT-3.5 era capabilities).
- ESS R11 (fielded October 2023 – 2024): paired with **2025 ILO–NASK scores** (representing GPT-4, GPT-4o, Gemini Flash 1.5 era capabilities).

This creates a structural break in $E_o$ at R11 that genuinely captures the GenAI capability discontinuity, addressing the previously-flagged concern that earlier exposure measures were time-invariant. Within-country variation in $\bar E_{ct}$ now arises from two sources: compositional drift in $w_{oct}$ (slow, steady) and the vintage shift at R11 (sharp, capability-driven). Robustness check: re-estimate with 2025 scores applied uniformly across all rounds (no vintage variation) to isolate the pure compositional effect.

Expected panel: 6 rounds × ~25 countries → ~150 country-years (L2), ~25 countries (L3), ~200,000+ individual respondents (L1). Comfortably above Bryan & Jenkins (2016) thresholds. Round 11 country coverage at submission: 30 countries available in Edition 4.0 of the integrated file (November 2025 release).

---

## 3. Project layout

```
MLA/
├── IMPLEMENTATION_PLAN.md   # this plan, lives at project root, committed in initial commit
├── README.md                # replication instructions: uv sync → add ESS_USER_ID to .env → nbmake notebooks/
├── .gitignore               # excludes .env, data/raw/, data/interim/, data/analysis/, .venv/, *.parquet
├── .env                     # ESS_USER_ID and any other secrets; GITIGNORED, never committed
├── pyproject.toml           # uv-managed env
├── paper/                   # LaTeX
│   ├── main.tex
│   ├── sections/{intro, theory, data, methods, results, discussion}.tex
│   ├── tables/              # auto-generated from notebooks
│   ├── figures/             # auto-generated from notebooks
│   └── refs.bib             # mostly imports from KUL_MultilevelAnalysis_26/Presentation/MyBib.bib
├── notebooks/
│   ├── 00_setup.ipynb               # env check, package versions
│   ├── 01_ess_load.ipynb            # load R1-R9 from disk + R10/R11 via ESS API (or manual fallback)
│   ├── 02_occupation_scores.ipynb   # ISCO-08 native ILO–NASK merge (primary); SOC crosswalk for robustness only
│   ├── 03_country_year_exposure.ipynb # ESS-internal weighted aggregation of ILO–NASK genai_i to E_ct (primary); LFS-based aggregation as robustness
│   ├── 04_macro_merge.ipynb         # Eurostat GDP/unemp/HICP, OECD EPL
│   ├── 05_descriptives.ipynb        # univariate, ICC priors, balance
│   ├── 05b_power.ipynb              # Monte Carlo power calibration for γ_W
│   ├── 06_models_M0_M2.ipynb        # empty + L1 + L2 controls
│   ├── 07_models_M3_M6.ipynb        # within-between core + interactions, M3a/M3b round-dummy split
│   ├── 08_robustness.ipynb          # alternative exposure measures (Webb/Felten/Eloundou), vintage-static, country drop, FE compare, MICE
│   └── 09_figures_tables.ipynb      # all paper outputs
├── data/
│   ├── raw/                 # ESS sav files (gitignored), Felten csv, ILO crosswalk, Eurostat csv
│   ├── interim/             # cleaned per-source frames (gitignored)
│   └── analysis/            # final 3-level analysis frame as parquet (gitignored)
├── src/mla/                 # importable helpers
│   ├── ess_io.py            # ESS API client (reads ESS_USER_ID from .env via python-dotenv) + pyreadstat-based SAV loaders + manual-fallback shim
│   ├── crosswalk.py         # ISCO-08 ↔ SOC-2010
│   ├── exposure.py          # ILO–NASK score merge (primary) + Webb/Felten/Eloundou robustness merges + country-year aggregation
│   ├── mundlak.py           # within-between decomposition helper
│   ├── models.py            # statsmodels MixedLM wrappers + ICC/VPC
│   └── plotting.py          # caterpillar, conditional-effects, variance plots
└── tests/                   # pytest sanity checks on crosswalk + decomposition + ESS API loader
```

**Git workflow.** The project is a Git repository from Day 1. Each day ends with a single `git commit -m "day N: <summary>"` on `main`. Day 7 additionally creates an annotated tag `submission-v1` and pushes tags to origin. Sensitive material (`.env`) and large binary derivatives (`data/raw/`, `data/interim/`, `data/analysis/`, `*.parquet`, `.venv/`) are excluded by `.gitignore`; the repo therefore contains only code, notebooks, the paper LaTeX source, and lightweight metadata.

**README.** Includes setup-from-scratch replication steps:

```text
# replication
1. clone the repo: git clone <URL> && cd MLA
2. install python env: uv sync
3. create .env at project root with: ESS_USER_ID=<your-id>
   (register an ESS account at ess.sikt.no to obtain a user id)
4. run all notebooks: uv run nbmake notebooks/
5. build paper: cd paper && latexmk -pdf main.tex
```

---

## 4. Software stack (pure Python)

```python
# core
import numpy as np, pandas as pd, pyreadstat
import statsmodels.api as sm, statsmodels.formula.api as smf

# panel sanity-check
from linearmodels.panel import PanelOLS, RandomEffects, compare

# plotting
import matplotlib.pyplot as plt, seaborn as sns

# tables for paper
from stargazer.stargazer import Stargazer  # statsmodels-compatible LaTeX tables

# secrets / API
from dotenv import load_dotenv; load_dotenv()
import os, requests
ESS_USER_ID = os.environ["ESS_USER_ID"]  # never hardcoded
```

Key calls:

```python
# Empty 3-level model: country-year nested within country
md0 = smf.mixedlm(
    "trust ~ 1",
    data=df,
    groups="country_iso",
    re_formula="~1",
    vc_formula={"cy": "0 + C(country_year)"},
)
res0 = md0.fit(reml=True, method=["lbfgs"])

# Within-between core (M3a, with round dummies)
md3a = smf.mixedlm(
    "trust ~ genai_i + C(eisced) + agea + agea_sq + female + C(domicil) "
    "+ C(mnactic) + hinctnta "
    "+ gdp_growth_ct + unemp_ct + hicp_ct + C(essround) "
    "+ exposure_within + exposure_between",
    data=df, groups="country_iso",
    re_formula="~1",
    vc_formula={"cy": "0 + C(country_year)"},
)
res3a = md3a.fit(reml=True)

# M3b: same as M3a but drop C(essround) to recover global-trend within-effect
# Cross-validate via pymer4 on 5-country subset; use lmerTest Kenward-Roger
# for L3 coefficients in M6.
```

**Validation**: spot-check final M3/M5 fits against `pymer4` (R-bridge) on a single country subset to confirm `MixedLM` 3-level behaviour. If `MixedLM` convergence is unstable on the full sample, fall back to **2-level + country fixed effects** (still recovers $\gamma_W$ and $\gamma_B$ via Mundlak; loses cross-country variance decomposition but rubric-defensible). Document either choice explicitly.

---

## 5. Variable construction

**Outcome**: institutional-trust composite. Z-standardise `trstprl`, `trstlgl`, `stfdem` per round-country, then average. Cronbach's α expected ≥ 0.75. Run sensitivity on each item separately.

**L1**:

- `genai_i`: ILO–NASK GenAI exposure score for individual's `isco08` 4-digit occupation, vintage-assigned per the rule in §2 (2023 GBB for R6–R10, 2025 ILO–NASK for R11). Z-standardised within sample. No crosswalk required.
- `genai_i_static`: same individual exposure constructed using 2025 scores uniformly across all rounds, used in robustness as a "vintage held constant" comparator.
- `webb_i`, `aioe_i`, `eloundou_i`: alternative individual exposure measures, mapped via ILO ISCO-08↔SOC-2010 crosswalk; reported in Section 7 robustness only.
- `eisced`: ISCED-based education, treated categorical (ref = ISCED 0–2).
- `hinctnta`: household income decile (numeric, ~15–25% missing — see missing-data plan in §2).
- `agea`, `agea_sq`: centred at 45.
- `mnactic`: main activity (dummy: employed / unemployed / retired / student / other).
- `gndr`, `domicil`: gender, urban/rural.
- Survey weights: `pspwght` carried throughout; multiplied by a country-rescaling factor so each country contributes equally in pooled analyses (standard ESS practice).

**L2 (country-year)** — computed by ESS-internal weighted aggregation at 4-digit ISCO:

- `exposure_ct`: weighted mean of `genai_i` over individuals in (country, round) using `pspwght`.
- `exposure_within = exposure_ct - mean(exposure_ct by country)`.
- `exposure_between = mean(exposure_ct by country)`.
- **Leave-one-out sensitivity**: recompute `exposure_ct` excluding individual i (i.e., $\bar E_{ct}^{(-i)}$) to break mechanical correlation with own occupation; expect coefficients within Monte Carlo error.
- **Vintage sensitivity**: also compute `exposure_ct_static` using `genai_i_static` (2025 scores throughout); the gap between the two diagnoses how much of the within-effect is vintage-driven (capability shock) vs. compositional drift (slow occupation-mix change).
- `gdp_growth_ct`, `unemp_ct`, `hicp_ct`: Eurostat (`nama_10_gdp`, `une_rt_a`, `prc_hicp_aind`).
- Round dummies: included in M2+ as primary (M3a); **also estimate M3b without round dummies** to recover the global-trend within-effect (Section 6 discusses the interpretive trade-off).

**L3 (country)** — pre-period for exogeneity:

- `epl_c`: OECD EPL index measured at 2008 baseline.
- `welfare_regime_c`: categorical (Social-democratic / Conservative-corporatist / Liberal / Mediterranean / Eastern European), assigned from pre-2008 typologies.
- `ai_policy_c` (optional): OECD AI Policy Observatory stringency, earliest available wave.

**Survey weighting strategy**: primary specifications are unweighted (Schmidt-Catran's own convention in pooled cross-national work, since `MixedLM` does not natively support survey weights for variance components). Weighted re-estimation via `pymer4`/`lme4::lmer(weights = pspwght)` reported in robustness.

---

## 6. Model-building sequence

| Model | Specification | Purpose |
|---|---|---|
| **M0** | Empty 3-level intercept-only | Compute ICC at L3 and VPC at L2; justify multilevel. |
| **M1** | M0 + L1 fixed effects | Composition: how much L2/L3 variance is compositional? Report % variance reduction Snijders/Bosker style. |
| **M2** | M1 + L2 macro controls (round dummies, GDP growth, unemp, HICP) | Establish baseline before AI variable. |
| **M3a** | M2 + AI exposure with Mundlak decomposition (`exposure_within` + `exposure_between`), **with** round dummies | **Core within-between test**; H1, H2, H2a. Wald test of $\gamma_W = \gamma_B$. With round dummies, $\gamma_W$ identifies country-specific deviations from the global AI-exposure trend. |
| **M3b** | M3a **without** round dummies | $\gamma_W$ now absorbs the global trend; substantively wider but confounded with anything else co-trending. Reported alongside M3a in main table; the gap between M3a and M3b is itself informative. |
| **M4** | M3a + random slope on individual `genai_i` (group-mean-centred) | Allow heterogeneity across countries; visualise via caterpillar plot. |
| **M5** | M4 + cross-level interaction `exposure_within × eisced` | **γ-flavoured H4a / H4b / H4₀ adjudication**. |
| **M6** | M5 + L3 institutional moderator (`epl_c × exposure_within`) | H5: varieties-of-capitalism buffering. |

Estimation: REML for final reported estimates; ML for nested LR comparisons of fixed effects. For random-effect tests, halve the p-value (boundary-problem convention from Lesson 9).

**Small-cluster L3 inference (≈25–30 countries)**: `MixedLM`'s default Wald-normal SEs are anti-conservative at this cluster count (McNeish & Stapleton 2016). For all L3-relevant coefficients (`exposure_between`, `epl_c`, welfare-regime contrasts, and the `epl_c × exposure_within` cross-level interaction in M6), re-estimate via `pymer4`'s `Lmer` with `lmerTest`'s **Kenward-Roger** corrected SEs and degrees of freedom. Report the Kenward-Roger column as primary for these coefficients; document the discrepancy with `MixedLM` Wald-normal in an appendix table. L1- and L2-level coefficients are sufficiently powered for `MixedLM`'s default inference.

**Power note**: with only ~25 L3 units and slow-moving compositional drift in $\bar E_{ct}$, the within-between Wald test is the most power-constrained test in the paper. A half-day Monte Carlo (notebook `05b_power.ipynb`) calibrates the minimum detectable $\gamma_W$ at 80% power across plausible $\sigma^2_{v_0}$ values, used both to interpret null findings honestly and to justify the design ex ante.

Reporting: produce one master table (M0 → M6) with coefficients, variance components ($\sigma_{u_0}^2$, $\sigma_{v_0}^2$, $\sigma_e^2$, ICC, L2 VPC), and proportional variance reduction at each level.

---

## 7. Robustness checks (Section 6 of paper)

1. **Alternative exposure measures**: Webb (2020), Felten et al. 2021 (AIOE), Felten et al. 2023 (GenAI), and Eloundou et al. 2024 (GPT-task exposure) as substitutes for ILO–NASK; results should qualitatively replicate. Disagreement is itself reportable as construct-validity evidence. Includes a **vintage-static** variant using 2025 ILO–NASK scores applied uniformly across all rounds, which isolates the pure compositional effect from the R10→R11 vintage shock.
2. **Round-dummy specification**: M3a (with) vs M3b (without) reported side-by-side in main text; further variants (linear time trend, country-specific linear trend) in appendix.
3. **Country-year aggregation source**: ESS-internal (primary, 4-digit ISCO) vs Eurostat LFS (1- and 2-digit). Expect attenuation under Eurostat aggregation; magnitude of attenuation diagnoses how much exposure precision matters.
4. **Leave-one-out exposure**: $\bar E_{ct}^{(-i)}$ excludes individual i from the country-year mean; addresses mechanical correlation between $\text{webb}_i$ and $\bar E_{ct}$.
5. **Item-by-item outcome**: `trstprl`, `trstlgl`, `stfdem` separately.
6. **Country leverage**: drop one country at a time; report stability of $\gamma_W$ and $\gamma_B$.
7. **Within-only FE specification**: country-fixed-effects regression via `linearmodels.PanelOLS`; expect $\hat\beta_{FE}\approx\hat\gamma_W$ from M3a.
8. **Mundlak vs Hausman-Mundlak**: report both forms, with Wald and LR variants of the equality test.
9. **Cross-validation against `pymer4`/`lme4`** on a country subset; agreement to 3 decimals expected.
10. **Survey-weighted re-estimation**: `pymer4` with `pspwght` rescaled to country-equal contribution.
11. **Multiple imputation for `hinctnta` missingness**: `statsmodels.imputation.mice` with 20 imputations; pool via Rubin's rules.
12. **Round subsetting**: drop R10 (COVID-disrupted, mode-mixed fieldwork) and report whether $\gamma_W$ survives.

---

## 8. Paper structure (~12 pages, LaTeX)

1. **Introduction** (~1.5 pp): puzzle, contribution, roadmap.
2. **Theory and hypotheses** (~2 pp): three literatures, derivation of H1–H5.
3. **Data and measurement** (~2 pp): ESS R6–R11, ILO–NASK 2025 GenAI exposure (primary, ISCO-08 native, with 2023/2025 vintage rule), Webb (2020)/Felten 2021/Felten 2023/Eloundou 2024 (robustness), Eurostat macro; Table 1 descriptive stats; Figure 1 country-year exposure trajectories highlighting the R10→R11 vintage shift.
4. **Methods** (~2 pp): 3-level model, within-between specification, estimation. Equation block for M0/M3/M5.
5. **Results** (~3 pp): Table 2 model progression M0–M6; Figure 2 caterpillar plot of country random intercepts; Figure 3 conditional effects of within-exposure × education; variance-component reduction table.
6. **Robustness** (~1 pp): summary table referencing appendix.
7. **Discussion** (~1 pp): substantive interpretation of $\gamma_W$ vs $\gamma_B$ pattern (which scenario A–E from Part II of options.md materialises); limitations; future work. Must include an explicit **endogeneity paragraph**: the within-between specification handles time-invariant country confounders but not time-varying ones (e.g., simultaneous shifts in welfare policy, political polarisation, or media environment). Frame the contribution as identifying *associations within a multilevel structure*, not causal effects. Reverse causation — declining trust accelerating AI adoption via labour-market deregulation or reduced public investment in retraining — is a serious alternative that the design cannot rule out.
8. **References** (~0.5 pp): import from `KUL_MultilevelAnalysis_26/Presentation/MyBib.bib` + AI/automation-exposure additions (Gmyrek et al. 2025, ILO Working Paper 140; Gmyrek, Berg & Bescond 2023, ILO Working Paper 96; Webb 2020; Felten et al. 2021, 2023; Eloundou et al. 2024; Anelli, Colantone & Stanig 2021; Frey, Berger & Chen 2018; Acemoglu & Restrepo 2020; Gidron & Hall 2017; Kurer 2020).

Appendix (separate, not counted): full robustness tables, occupation-mapping detail, convergence diagnostics, code repo URL.

---

## 9. Critical files

- **Read for theory & method alignment**: `KUL_MultilevelAnalysis_26/Presentation/KUL_MultilevelAnalysis_26_Sessions7to9.Rmd` (within-between, three-level, testing); `MyBib.bib` (citation pool).
- **Read for exercise patterns to mirror**: `KUL_MultilevelAnalysis_26/R/exercise12_key.R` (within-between + cross-level interaction), `exercise13_key.R` (3-level + ESS pooled).
- **Existing data**: `KUL_MultilevelAnalysis_26/R/Data/RData/ess1to9.Rdata` — load via `pyreadr` or re-import the original `Data/Stata/ESSrounds1to9.dta` with `pyreadstat`.
- **New data to fetch**: ESS R10 + R11 (Edition 4.0, Nov 2025 release) via the **ESS API** at <https://api.ess.sikt.no/docs> with `ESS_USER_ID` from `.env` (manual SAV drop into `data/raw/ess/` is the documented fallback while the API beta stabilises); **ILO–NASK 2025 + GBB 2023 scores** from `github.com/pgmyrek/2025_GenAI_scores_ISCO08` (`Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx`); Webb 2020, Felten 2021 + 2023, Eloundou 2024 CSVs (robustness only); ILO ISCO-08↔SOC-2010 crosswalk (robustness only); Eurostat `nama_10_gdp`, `une_rt_a`, `prc_hicp_aind` (and `lfsa_egais` / `lfsa_egan2` for robustness); OECD EPL.

---

## 10. Verification

**End-to-end verification** that the paper is complete and correct:

1. **Crosswalk sanity** (`tests/test_crosswalk.py`): assert every ESS `isco08` 4-digit code maps to ≥1 SOC code; spot-check 5 high-stakes occupations (e.g., software developers ISCO 2512 ≈ SOC 15-1252).
2. **Decomposition identity**: assert `exposure_ct == exposure_within + exposure_between` per row, up to floating tolerance.
3. **ICC reproduction**: M0 ICC at country level should reproduce the lecturer's ESS ICC (~0.15–0.20 for trust items) within a few percentage points; if not, investigate clustering definition.
4. **L2 VPC check**: M0 country-year VPC ≥ 0.02; if lower, retain the design but explicitly justify it via threat-of-bias rather than variance share (Schmidt-Catran's framing).
5. **Mundlak test**: report Wald χ² for $\gamma_W = \gamma_B$ in M3a and M3b separately; expect divergence between the two specifications.
6. **Round-dummy sensitivity**: report side-by-side $\gamma_W$ from M3a and M3b; gap quantifies how much of the within-effect is global trend vs. country-specific deviation.
7. **ILO–NASK / Webb / Felten / Eloundou agreement**: at the individual level, |corr(`genai_i`, `webb_i`)|, |corr(`genai_i`, `aioe_i`)|, |corr(`genai_i`, `eloundou_i`)| each ≥ 0.5 expected; if substantially lower for any pair, investigate the SOC↔ISCO-08 crosswalk before reporting that measure in robustness.
8. **Leave-one-out check**: M3a `exposure_within` coefficient using $\bar E_{ct}^{(-i)}$ within Monte Carlo error of standard $\bar E_{ct}$ specification.
9. **R cross-check**: refit M3a in `pymer4` on a 5-country subset; coefficients should match `MixedLM` to 3 decimals.
10. **Kenward-Roger comparison**: for L3 coefficients, compare `MixedLM` Wald-normal SEs against `pymer4`+`lmerTest` Kenward-Roger; if discrepancy materially changes inference (e.g., flips significance of `epl_c × exposure_within`), Kenward-Roger is reported as primary in the main table.
11. **Power calibration**: simulation in `05b_power.ipynb` confirms the design has ≥80% power to detect the literature's reported within-effect magnitude under our $\sigma^2_{v_0}$; null findings are reported with the corresponding minimum-detectable-effect.
12. **Full pipeline rerun**: `make all` (or `nbmake notebooks/`) reruns notebooks 01→09 from raw data to final figures/tables in <30 min on laptop.
13. **LaTeX build**: `latexmk -pdf paper/main.tex` produces a ≤12-page PDF with all figures/tables embedded; bibliography resolves; no missing references.
14. **Page-count check**: main body ≤12 pages excluding references and appendix.

---

## 11. Compressed 7-day timeline with parallel subagent delegation

Replaces the original 8-week plan. Targets a 7-day intensive sprint via parallel-agent delegation. The principle: each day spawns 2–3 Explore/general-purpose subagents working on independent slices of the day's work; the main thread integrates their outputs at end-of-day. The original 8-week plan remains the fallback if a day's integration reveals a structural blocker (e.g., persistent 3-level convergence failure).

Each day ends with a single `git commit` on `main`. Day 7 additionally creates the `submission-v1` annotated tag and pushes tags to origin.

The compressed schedule still hits the 2026-06-20 deadline with ~7 weeks of polish/buffer.

| Day | Parallel tracks | End-of-day integration |
| --- | --- | --- |
| **Day 1 — Setup, Git init & ESS load** | A: `uv` Python env, project skeleton, package pins; `git init`; `git remote add origin <URL>`; commit `IMPLEMENTATION_PLAN.md`, `README.md`, `.gitignore`, `pyproject.toml` as the initial commit. B: pull ESS R10 + R11 via the ESS API (`src/mla/ess_io.py`, credentials from `.env`); fall back to manual SAV drop in `data/raw/ess/` if API beta cannot deliver SAV; verify variable names against R1–R9; load all rounds via `pyreadstat` into a single long-format frame. | Single tidy `ess_full.parquet` (gitignored) keyed on (`cntry`, `essround`, `idno`); `pspwght`/`pweight` carried; smoke test on `trstprl` distribution per round. End-of-day `git commit -m "day 1: project skeleton + ESS load"`. |
| **Day 2 — Occupation pipeline (lighter than original)** | A: ISCO-08 → ILO–NASK score merge — both 2023 GBB and 2025 vintages — direct, no crosswalk. Apply vintage rule: 2023 to R6–R10, 2025 to R11. B: SOC↔ISCO-08 crosswalk + Webb (2020) + Felten 2021 + Felten 2023 + Eloundou 2024 for robustness only. C: write `tests/test_scores.py` (every ESS `isco08` 4-digit gets a primary score; spot-check 5 high-stakes occupations across primary and robustness measures). | One `occupation_scores.parquet` with primary `genai` (2023 + 2025 vintages) and robustness `webb`, `aioe`, `felten_genai`, `eloundou` per ISCO-08 4-digit; merge into ESS frame to produce `genai_i`, `genai_i_static`, plus robustness measures per individual. ~Half-day saved vs. crosswalk-primary path; bank into Day 5 robustness. End-of-day `git commit -m "day 2: occupation scores pipeline"`. |
| **Day 3 — Country-year exposure & macro merge** | A: ESS-internal weighted aggregation $\bar E_{ct}$ at 4-digit ISCO; build `exposure_within`/`exposure_between` via Mundlak helper in `src/mla/mundlak.py`. B: Eurostat macro panel (`nama_10_gdp`, `une_rt_a`, `prc_hicp_aind`). C: OECD EPL pre-2008 + welfare-regime typology + Eurostat LFS shares for robustness aggregation. | Final `analysis.parquet` (3-level keyed, gitignored). Identity check: `exposure_ct == exposure_within + exposure_between`. End-of-day `git commit -m "day 3: country-year exposure + macro merge"`. |
| **Day 4 — Descriptives, M0–M2, power simulation** | A: descriptive table 1, country-year exposure trajectory plot (figure 1). B: M0 empty 3-level + M1 (L1 fixed) + M2 (+L2 macro); compute ICC, L2 VPC, variance reduction. C: half-day Monte Carlo power simulation across plausible $\sigma^2_{v_0}$ values. | Decision point: confirm L2 VPC ≥ 0.02 (or pre-register narrower contribution claim if not). Flag any 3-level convergence pathology now, not later. End-of-day `git commit -m "day 4: descriptives + M0–M2 + power"`. |
| **Day 5 — M3–M6 + robustness battery** | A: M3a (with round dummies), M3b (without), M4 (random slope), M5 (cross-level interaction), M6 (EPL moderation) in `MixedLM`. B: `pymer4` cross-validation on 5-country subset; Kenward-Roger SEs for L3 coefficients via `pymer4`+`lmerTest`. C: full robustness battery (12 checks from §7). | Master regression table M0→M6; robustness summary table; Mundlak Wald χ². End-of-day `git commit -m "day 5: M3–M6 + robustness"`. |
| **Day 6 — Figures, tables, LaTeX draft** | A: caterpillar plot, conditional-effects plot for M5, variance-component bar plot (`src/mla/plotting.py`). B: convert all results to LaTeX via `Stargazer` + `pandas.DataFrame.to_latex`; auto-write to `paper/tables/`. C: LaTeX skeleton + draft of intro, theory, data, methods sections. | First compileable PDF of paper; figures embedded; placeholders only in results/discussion. End-of-day `git commit -m "day 6: figures + LaTeX draft"`. |
| **Day 7 — Results, discussion, polish, submit** | A: results section narrative tracking master table. B: discussion (scenario-A–E interpretation, endogeneity paragraph, limitations). C: bibliography import from `MyBib.bib` + AI/automation additions; final `latexmk -pdf` build, page-count trim to 12, proofread, submit. | Final PDF emailed to <alex@alexanderwschmidt.de>. End-of-day `git commit -m "day 7: submission"`; `git tag -a submission-v1 -m "Submission to Schmidt-Catran, 2nd exam period"`; `git push origin main --tags`. |

Delegation rules: each subagent gets a self-contained brief specifying inputs, outputs, file paths, and the single deliverable to return. Main thread reviews returned artifacts at end-of-day before launching next day's tracks. Worktree isolation (`isolation: "worktree"`) recommended for concurrent file-writing tracks to avoid merge collisions.

Hard checkpoints (do not advance without):

- Day 1: ESS R10+R11 loaded with variable parity vs. R1–R9; initial commit pushed.
- Day 3: identity `exposure_ct == exposure_within + exposure_between` holds element-wise.
- Day 4: M0 ICC reproduces lecturer's ESS trust ICCs to within ~3 pp.
- Day 5: `MixedLM` and `pymer4` agree on M3a coefficients to 3 decimals on the 5-country subset.
- Day 7: PDF page count ≤ 12 (excluding refs/appendix); no LaTeX warnings about missing refs; `submission-v1` tag pushed.

---

## 12. Risks and contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| `MixedLM` 3-level convergence failure | Medium | Fall back to 2-level + country FE (Mundlak still works); cross-validate via `pymer4`. |
| ESS API beta does not yet support SAV download | Medium | Documented manual fallback: drop R10 + R11 SAVs into `data/raw/ess/`; loader detects local files and skips API path. |
| ESS R10/R11 unavailable through any path | Low | Trim to R6–R9 (already on disk); flag in limitations. |
| ISCO-08 ↔ SOC crosswalk many-to-many ambiguity | Medium | Use ILO weighted crosswalk; sensitivity-test with Eurostat 2-digit aggregation. |
| L2 VPC < 0.02 (within-between weakly motivated) | Low | Pre-register the design choice; argue from threat-of-bias rather than from variance share (Schmidt-Catran's own framing). |
| Within-coefficient $\gamma_W$ underpowered because of slow compositional drift | Medium-Low | The R10→R11 vintage shift in ILO–NASK scores adds genuine within-variation beyond pure compositional drift; vintage-static robustness isolates the two channels. Document both sources of within-variation explicitly; report power floor from `05b_power.ipynb`. |
| Round dummies absorb global trend, narrowing $\gamma_W$ | Medium | Report M3a (with) and M3b (without) side-by-side; discuss interpretive trade-off in methods. |
| Null result on H2 within-effect | Low–Medium | Scenario D in options.md is publishable; reframe as informative null with cross-sectional/within distinction emphasised; pair with power calibration. |
| L3 inference too liberal (~25 countries) | Medium | Kenward-Roger SEs via `pymer4`+`lmerTest` for L3 coefficients (`exposure_between`, `epl_c`, welfare-regime, M6 cross-level); report KR as primary in main table. |
| ILO–NASK / Webb / Felten / Eloundou disagreement | Low | Treat as construct-validity finding; report all columns; discuss substantive implications in robustness section. |
| Page-count overrun | High | Push detailed model tables and robustness to appendix; keep main body lean. |
| `pymer4` / R bridge install fails on macOS | Medium | Document; accept anti-conservative L3 SEs; note as limitation; the rest of the analysis is unaffected. |
| `.env` accidentally committed | Low | `.gitignore` excludes `.env` from Day 1; `git status` reviewed before each commit; credential rotation procedure documented in README if leak occurs. |

---

## 13. Prerequisites checklist (complete before kicking off Day 1)

- [ ] **(a) ESS account created and user ID in `.env`**: register at <https://ess.sikt.no/> and obtain an `ESS_USER_ID`. Create `.env` at the project root containing `ESS_USER_ID=<your-id>` (one line, no quotes). The API loader in `src/mla/ess_io.py` reads it via `python-dotenv`. `.env` is gitignored from Day 1 and must never be committed.
- [ ] **(b) GitHub repo created and URL noted**: create an empty private GitHub repo for the project, copy the clone URL, and record it in this document or in `README.md`. Day 1 Track A executes `git init` + `git remote add origin <URL>` + the initial commit.
- [ ] **(c) IMPLEMENTATION_PLAN.md saved to project root**: this file lives at `/path/to/MLA/IMPLEMENTATION_PLAN.md` and is part of the initial commit. Future plan revisions are committed as ordinary changes on `main`.
