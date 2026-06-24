# Results notebooks

Each notebook presents **one experiment / claim** of the paper, numbered in reading order.
Notebooks are *dashboards*: they load results already computed by the pipeline scripts and
render the figures, tables, and statistics. They run in seconds (the heavy work — data
download, model training, SHAP — lives in the top-level `*.py` scripts and is saved to
`../*.csv`). Outputs are committed so the notebooks render on GitHub without execution.

| notebook | claim / contribution | status |
|----------|----------------------|--------|
| `00_synthetic_ground_truth.ipynb` | Controlled synthetic check: the explanation-shift metric fires on benign covariate movement and stays silent under the concept shift that actually hurts accuracy (appropriate-vs-spurious). Produces Figure 4. Source: `../../week1_demo.py` Part A. | ✅ done |
| `01_early_warning_test.ipynb` | **Decisive pre-registered test:** does label-free explanation shift predict accuracy drop beyond cheap drift baselines? Result: **NO-GO** (clean null). Source: `../../week2_powered.py`, `../week2_rows_full.csv`. | ✅ done |
| `02_trade_off_audit.ipynb` | Four-property shift audit (predictive performance, calibration, group fairness, explanation stability) — do they degrade together or trade off? Result: they degrade **independently** (accuracy is a weak proxy). Source: `../../week3_audit.py`, `../week3_audit_full.csv`. | ✅ done |
| `03_paper_figures.ipynb` | Assembles paper exhibits from the saved tables: Table 1 (design), Table 2 (four properties, in-dist vs shifted, with bootstrap CIs), Figure 6 (summary heatmap). Exports to `../tables/` and `../figures/`. Fig 1/4 await `00`. | ✅ done |

## Conventions
- **Engine vs dashboard:** `*.py` scripts = reproducible compute; `*.ipynb` = presentation.
- **Data** lands in `../*.csv` (raw per-row tables) and `../tables/` (summary tables).
- **Exported figures** for the manuscript go in `../figures/`.
- Re-running a notebook needs the analysis env: `pip install -r ../../requirements.txt` plus
  `ipykernel`; then select the `.venv` kernel.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python week2_powered.py --scale full   # regenerates ../week2_rows_full.csv (~30–45 min)
# then run the notebooks top-to-bottom
```
