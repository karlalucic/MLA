# Detailed development: Options α and γ

## Part I: Shared methodological architecture

Both papers use the same data architecture, model-building sequence, and software stack. Only the substantive variables and theoretical framing differ.

### Data sources

| Layer | Source | Variable type | Notes |
|-------|--------|---------------|-------|
| L1 individual | ESS rounds 6 to 11 | Survey responses | Already on disk |
| L2 country-year | Constructed from ESS occupation shares + Felten exposure scores; Eurostat macro controls | Continuous | LFS occupation shares are free; macro indicators standard |
| L3 country | OECD (EPL, AI policy), Eurostat (R&D), welfare-regime typology | Time-invariant or slow-moving | Standard sources |
| Occupation crosswalk | Felten et al. 2021 (AIOE), Felten et al. 2023 (GenAI) | Public CSVs | Need ISCO-08 to SOC mapping (ILO publishes the crosswalk) |

ESS rounds 6 to 11 cover roughly 2012 to 2024. With 22 to 30 countries participating per round, the country-year panel has roughly 130 to 180 observations at L2, 22 to 30 at L3, and 250,000+ individuals at L1. This is well above the minimum for stable 3-level estimation (Bryan and Jenkins 2016 suggest at least 25 L3 units; with country-year as L2 you have many more L2 units than L3 units).

### Software stack

```python
# Primary: statsmodels for linear mixed models
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 3-level via vc_formula: works for our nested case
# (country-year nested within country)
md = smf.mixedlm(
    "outcome ~ <fixed effects>",
    data=df,
    groups="country",
    re_formula="~1 + <random slope variable>",
    vc_formula={"cyear": "0 + C(country_year)"}
)
result = md.fit(reml=True)
```

The `statsmodels.MixedLM` route is the verify-needed step from your original analysis. It works for nested 3-level structures (country-year within country), which is what you have. Validate against `pymer4` (R bridge to `lme4`) on a subset for the final models. `bambi` is a Bayesian alternative if you want posterior intervals on variance components.

### Within-between (Mundlak) decomposition

For any L2 variable $X_{ct}$, decompose:

$$X_{ct} = \bar{X}_c + (X_{ct} - \bar{X}_c)$$

where $\bar{X}_c$ is the country-specific mean across years (between effect) and $(X_{ct} - \bar{X}_c)$ is the deviation from country mean (within effect). Including both terms separately in the model allows the within-coefficient $\gamma_W$ and between-coefficient $\gamma_B$ to differ. The Hausman-style test of $H_0: \gamma_W = \gamma_B$ is the substantive Mundlak test and is the rubric centrepiece (Bell and Jones 2015; Schmidt-Catran and Fairbrother 2016).

### Stepwise model-building sequence

Both papers run an identical sequence:

| Model | Specification | Purpose |
|-------|---------------|---------|
| M0 | Empty 3-level intercept-only | Compute ICC at L3 and VPC at L2; justify multilevel |
| M1 | M0 + L1 fixed effects | Composition: how much L2/L3 variance is compositional? |
| M2 | M1 + L2 macro controls (round dummies, GDP growth, unemployment) | Establish baseline before AI variable |
| M3 | M2 + AI exposure with Mundlak decomposition | Core within-between test |
| M4 | M3 + random slope on individual AI exposure | Allow heterogeneity across countries |
| M5 | M4 + cross-level interaction (within $\times$ individual exposure) | Test H4 |
| M6 | M5 + L3 institutional moderator | Test H5 |

Reported with: REML estimates for final models, ML for nested model comparison via likelihood ratio tests (since REML log-likelihoods are not comparable across different fixed-effects specifications).

### Expected variance components (priors, not measurements)

Drawing on Fairbrother (2014) and Schmidt-Catran and Fairbrother (2016) patterns for ESS pooled trust outcomes:

- **L3 country ICC:** 0.15 to 0.25 for political trust outcomes; 0.10 to 0.20 for satisfaction with democracy and education.
- **L2 country-year VPC:** 0.03 to 0.08. This is the key quantity: it is the share of variance attributable to country-years within countries, and within-between decomposition is methodologically motivated when this is non-trivially positive.
- **L1 residual:** the remainder, typically 0.70 to 0.85.

If L2 VPC is below 0.02, the within-between decomposition adds little, and Schmidt-Catran would expect you to argue for the design anyway because cross-sectional pooling creates the threat that motivates the method. State this explicitly in the paper.

---

## Part II: Option α: AI exposure and political trust

### Theoretical framing

Three literatures converge here:

1. **Performance theories of trust** (van der Meer 2017, Hetherington 1998): institutional trust responds to perceived institutional performance, including macroeconomic and structural conditions.
2. **Structural-economic-grievance theory** (Gidron and Hall 2017, Kurer 2020): technological change generates winners and losers, and losers' political behaviour shifts via grievance mechanisms.
3. **Robot exposure and political behaviour** (Anelli, Colantone, and Stanig 2021; Frey, Berger, and Chen 2018): empirical evidence that automation exposure predicts populist voting and reduced trust in single-country US and Western European studies.

The contribution: extend from robot exposure (manufacturing-biased) to AI exposure (services-biased, white-collar), use ESS pooled cross-national data, and apply Schmidt-Catran-style within-between decomposition to separate country-level differences from country-level changes. The Bell and Jones (2015) point is that previous cross-sectional and within-country studies have been answering different questions without acknowledging it.

### Hypotheses

**H1 (between, level effect):** Across countries, higher mean AI exposure $\bar{E}_c$ associates with lower mean institutional trust. Direction: $\gamma_B < 0$.

**H2 (within, shock effect):** Within countries over time, increases in AI exposure $(E_{ct} - \bar{E}_c)$ erode institutional trust. Direction: $\gamma_W < 0$.

**H2a (decomposition):** $\gamma_W \neq \gamma_B$. Either coefficient could be larger in absolute value; theory does not pin this down. The substantive interpretation differs sharply: a strong $\gamma_B$ with weak $\gamma_W$ would suggest selection or institutional adaptation; a strong $\gamma_W$ with weak $\gamma_B$ would suggest a genuine shock effect that countries do not yet equilibrate to.

**H3 (compositional):** At the individual level, workers in AI-exposed occupations show lower institutional trust net of country-context, education, and income. Direction: $\beta_{\text{AIOE}_i} < 0$.

**H4 (cross-level interaction):** The within-country effect of AI exposure is amplified for individuals in AI-exposed occupations. Direction: the interaction coefficient on $(E_{ct} - \bar{E}_c) \times \text{AIOE}_{ict}$ is negative.

**H5 (institutional moderation):** The within-effect is attenuated in countries with stronger employment protection. Grounded in varieties-of-capitalism literature (Hall and Soskice 2001) and welfare-state buffering arguments. Direction: positive coefficient on $\text{EPL}_c \times (E_{ct} - \bar{E}_c)$.

### Variable operationalisation

**Outcomes (continuous, 0 to 10):**
- `trstprl`: trust in [country] parliament
- `trstlgl`: trust in legal system
- `stfdem`: satisfaction with democracy
- Composite: standardise each, average, refit. Sensitivity analysis with each item separately.

**L1 fixed effects:**
- `aioe_i`: individual AIOE score, mapped from `isco08` via Felten ISCO-to-SOC crosswalk; standardised within sample.
- `eisced`: education (ISCED), treated as categorical with reference category lower-secondary.
- `hinctnta`: household income decile, treated as continuous after checking linearity.
- `agea`: age, centred at 45, with quadratic term.
- `mnactic`: main activity (employed, unemployed, retired, student, other), categorical.
- `gndr`: gender.
- `domicil`: urban/rural, categorical.

**L2 country-year:**
- $\bar{E}_{ct}$: weighted-average AIOE, $\bar{E}_{ct} = \sum_o w_{oct} \cdot E_o$, where $w_{oct}$ is the employment share of occupation $o$ in country $c$ at round $t$ (Eurostat LFS), and $E_o$ is the Felten AIOE score for occupation $o$. Decomposed via Mundlak into $\bar{E}_c$ (between) and $E_{ct} - \bar{E}_c$ (within).
- Round dummies: capture period effects (COVID, etc.).
- `gdp_growth_ct`: real GDP growth (Eurostat).
- `unemp_ct`: unemployment rate (Eurostat).

**L3 country:**
- `epl_c`: OECD employment protection index, averaged over study period (or measured pre-period for exogeneity).
- `welfare_regime_c`: categorical (Social-democratic, Conservative-corporatist, Liberal, Mediterranean, Eastern European).
- `ai_policy_c` (optional): OECD AI Policy Observatory stringency measure.

### Model specifications

**M0 (empty 3-level):**

$$y_{ict} = \gamma_{000} + u_{0c} + v_{0ct} + e_{ict}$$

Report: $\sigma^2_{u_0}, \sigma^2_{v_0}, \sigma^2_{e}$, ICC at country level $= \sigma^2_{u_0} / (\sigma^2_{u_0} + \sigma^2_{v_0} + \sigma^2_{e})$, VPC at country-year level $= \sigma^2_{v_0} / (\sigma^2_{u_0} + \sigma^2_{v_0} + \sigma^2_{e})$.

**M3 (within-between core model):**

$$y_{ict} = \gamma_{000} + \sum_{p} \beta_p X_{p,ict} + \gamma_W (E_{ct} - \bar{E}_c) + \gamma_B \bar{E}_c + \boldsymbol{\delta}^\top \mathbf{Z}_{ct} + u_{0c} + v_{0ct} + e_{ict}$$

where $\mathbf{Z}_{ct}$ contains L2 controls. Test $H_0: \gamma_W = \gamma_B$ via Wald.

**M5 (cross-level interaction, the substantive payoff):**

$$y_{ict} = \gamma_{000} + (\beta_1 + u_{1c}) \cdot \text{AIOE}_{ict} + \gamma_W (E_{ct} - \bar{E}_c) + \gamma_B \bar{E}_c + \gamma_{\text{int}} (E_{ct} - \bar{E}_c) \cdot \text{AIOE}_{ict} + \dots + u_{0c} + v_{0ct} + e_{ict}$$

### Expected results scenarios

| Scenario | $\gamma_B$ | $\gamma_W$ | Interpretation |
|----------|------------|------------|-----------------|
| A | Negative, large | Negative, large | Both level and shock effects: AI exposure structurally erodes trust |
| B | Negative, moderate | Null | Selection or institutional adaptation; cross-sectional finding is misleading |
| C | Null | Negative, moderate | Genuine shock effect; cross-sectional studies miss this |
| D | Null | Null | No structural relationship at population level (still publishable; valuable null) |
| E | Positive | Negative | Equilibrium adjustment paradox; rich theoretical discussion |

The paper is interesting in all five cases. This matters for risk management.

### Robustness checks

1. **Alternative exposure measure:** Webb (2020) patent-based AI exposure as a robustness check; results should qualitatively replicate.
2. **Alternative outcomes:** each trust item separately to detect heterogeneity.
3. **Country exclusion:** drop one country at a time to check leverage.
4. **FE specification:** compare to country-fixed-effects regression (which collapses to within-only); should recover similar within-coefficient.
5. **Mundlak vs Hausman-Mundlak:** report both.

---

## Part III: Option γ: GenAI exposure and trust in expertise

### Theoretical framing

Two literatures converge:

1. **Trust in expertise and the digital divide** (Norris 2001, Gauchat 2012): trust in scientific expertise varies systematically with education, ideology, and political context. A digital-stratification reading suggests new technologies amplify or shift these gradients.
2. **AI and epistemic authority** (Acemoglu 2024, Bender et al. 2021): GenAI shifts the locus of knowledge production and credentialing, potentially destabilising the legitimacy claims of formal expertise.

The contribution: test whether GenAI-driven occupational change at the population level erodes trust in formal expertise. The mechanism is theorised as a substitution-and-mediation effect: when AI mediates knowledge work at scale, the public's relationship to credentialed experts shifts. The cross-level interaction with individual education is theoretically interesting because the direction is genuinely underspecified by theory: more-educated respondents may either trust expertise more (curator effect) or less (substitution effect).

### Honest data caveat

Trust in scientists (`trstsci` or equivalent) is a rotating-module item in ESS. To my best understanding, it appears in Round 7 (Climate Change module) and selected later rounds, but coverage is uneven. This means the within-country panel for the strict trust-in-scientists outcome is thin.

**Mitigation:** primary outcome is `stfedu` (satisfaction with state of education), which is in every round and serves as a proxy for legitimacy of formal-credential institutions. Secondary outcomes are trust-in-scientists items where available. Frame the paper as testing trust in formal-credential institutions broadly, with scientists as a sensitivity analysis.

This is a real constraint, not a fatal one, but it does slightly weaken γ relative to α.

### Hypotheses

**H1 (between):** Across countries, higher mean GenAI exposure associates with lower satisfaction with the education system. Direction: $\gamma_B < 0$.

**H2 (within):** Within countries, increases in GenAI exposure erode satisfaction with education. Direction: $\gamma_W < 0$.

**H2a (decomposition):** $\gamma_W \neq \gamma_B$.

**H3 (compositional):** Individual GenAI exposure associates with lower trust in formal-credential institutions, net of education. Direction: $\beta_{\text{GenAI}_i} < 0$.

**H4 (cross-level moderation, the theoretically interesting test):** The within-country effect of GenAI exposure is moderated by individual education, but theory underspecifies direction. The paper tests two competing hypotheses:

- **H4a (substitution effect):** Higher-educated respondents react more strongly because they perceive direct competition from GenAI.
- **H4b (curator effect):** Higher-educated respondents react less strongly because they identify with expertise institutions.

Letting the data adjudicate is the substantive contribution. This kind of underspecified-theory test is rhetorically strong because both outcomes are interpretable.

### Variable operationalisation

Identical to α, with these substitutions:

- AIOE replaced by Felten et al. (2023) GenAI exposure score
- Outcomes: `stfedu` primary; `trstsci` and related where available, secondary
- L3 institutional moderator: R&D intensity (Eurostat) and public R&D funding share, in place of EPL

### Model specifications

Identical structure to α. The key cross-level interaction in M5:

$$y_{ict} = \dots + \gamma_{\text{int}} (E_{ct} - \bar{E}_c) \cdot \text{ISCED}_{ict} + \dots$$

where $\text{ISCED}_{ict}$ is the individual education level. Sign of $\gamma_{\text{int}}$ adjudicates H4a vs H4b.

### Expected variance components

Slightly higher L3 ICC than α expected for `stfedu` (likely 0.18 to 0.25), because satisfaction with education has stronger national-institutional roots than political trust. L2 VPC similar to α at 0.03 to 0.06.

### Robustness checks

Same five as α, plus:
6. **Outcome substitution:** run on `trstsci` for the rounds where available; report whether results qualitatively replicate.
7. **Round restriction:** restrict to rounds 7+ where the GenAI exposure measure is more conceptually applicable (the GenAI score is from 2023 but applies retrospectively given AI capability ramps).

---

## Part IV: Comparison and recommendation

| Dimension | α: Political trust | γ: Trust in expertise |
|-----------|-------------------|----------------------|
| Outcome data coverage | Every ESS round, balanced panel | `stfedu` every round; `trstsci` rotating, thin |
| Time series length | 6 rounds, 12 years | Same for primary outcome; thinner for secondary |
| Theoretical novelty | Moderate-high (extends robot literature to AI) | High (GenAI exposure barely tested on attitude outcomes) |
| Rubric fit | Maximal | Strong |
| Career-talkability | Strong (political consequences of AI) | Strong (knowledge institutions and AI) |
| Risk of weak findings | Low (effect documented for robots; plausible to extend) | Medium-high (mechanism more speculative) |
| Personal coherence with your trajectory | Strong | Stronger |

### Recommendation

Develop **α** as primary, with the option to add a γ-style cross-level interaction (individual education $\times$ within-country exposure) within α as a secondary analysis. This gives you the rubric-safe paper while preserving the knowledge-and-expertise angle as a sub-finding.

If you want maximum distinctiveness and are willing to accept slightly higher data risk, run γ. The paper would be more memorable.

The honest assessment: α is the better paper to write. γ is the better paper to talk about at dinner. If those goals conflict, α wins because the paper is what gets graded and what you'd cite later.

A hybrid is genuinely viable: title it with the political-trust framing (α), but include `stfedu` as a secondary outcome and run the education-moderation interaction. This gives you both the rubric-safe core and the more distinctive findings without committing to a thin data setup.