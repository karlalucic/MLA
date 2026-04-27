# Multilevel Analysis Final Paper — KU Leuven 2026

Within-between multilevel analysis of GenAI exposure and political trust across European countries (ESS R6–R11, ILO–NASK 2025 occupation scores). See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full design.

## Replication

```sh
# 1. clone
git clone <repo-url> && cd MLA

# 2. install python env (managed by uv)
uv sync

# 3. add ESS API credentials
cp .env.example .env
# edit .env and set ESS_USER_ID=<your-id>
# register an ESS account at https://ess.sikt.no/ to obtain a user id

# 4. run all notebooks end-to-end (raw data → figures + tables)
uv run nbmake notebooks/

# 5. build the paper
cd paper && latexmk -pdf main.tex
```

If the ESS API beta does not yet support SAV downloads, place R10 and R11 SAV files manually in `data/raw/ess/`; the loader auto-detects and skips the API path.

## Project layout

See [`IMPLEMENTATION_PLAN.md` §3](IMPLEMENTATION_PLAN.md). Briefly:

- `paper/` — LaTeX sources for the 12-page paper.
- `notebooks/` — analysis notebooks `00_…` through `09_…`.
- `src/mla/` — importable Python helpers (ESS loader, exposure merger, Mundlak decomposition, model wrappers, plotting).
- `data/raw|interim|analysis/` — gitignored data tree.
- `tests/` — pytest sanity checks.
- `KUL_MultilevelAnalysis_26/` — course materials (lectures, exercises, R example data).

## Course context

- **Course**: KU Leuven *Multilevel Analysis* (G0W07a), Spring 2026.
- **Lecturer**: Prof. Dr. Alexander Schmidt-Catran.
