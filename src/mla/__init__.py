"""Helpers for the KUL Multilevel Analysis 2026 final paper.

Submodules:
    ess_io      ESS loaders (API + manual SAV fallback)
    crosswalk   ISCO-08 ↔ SOC-2010 (robustness path only)
    exposure    Occupation-score merges + country-year aggregation
    mundlak     Within-between decomposition
    models      MixedLM wrappers + ICC/VPC
    plotting    Caterpillar, conditional-effects, variance plots
"""

__version__ = "0.1.0"
