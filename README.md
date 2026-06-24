# Trustworthiness Auditing of Tabular Classifiers Under Distribution Shift

A research project (target: Scopus Q1/Q2 applied-ML venue) that audits four
properties of tabular classifiers **together** under temporal and geographic
distribution shift: predictive performance, probability calibration, group
fairness, and explanation stability. The decisive question is whether a
**label-free explanation-shift signal** can warn of accuracy loss before labels
arrive, and whether it beats cheap drift baselines.

Full design: see the
[design blueprint (Google Doc)](https://docs.google.com/document/d/1gsNWNSKXBdAZeCbQ_YBc0HKspa7NxgrmP0xVHcp8Uvw/edit).

## Status (June 2026)

Week-1 go/no-go executed (`week1_demo.py`):

- **Synthetic ground-truth check: passed.** Observable explanation shift tracks
  *covariate* movement (a false alarm; accuracy did not drop), and stays nearly
  silent under *concept* shift, which is exactly when accuracy degrades.
- **Real-data go/no-go (Folktables): inconclusive / lean no-go.** On N=8 states
  with very small accuracy drops, no signal cleanly predicts the drop and
  explanation shift does not beat the baselines. The test bed was underpowered;
  it needs temporal x geographic domains and harder tasks before a final call.

## Data (two tiers)

- **Synthetic** — a small, fully controlled "credit scoring" world with named
  variables (age, income, education, debt ratio, employment length) and a known
  rule, so ground truth for "should explanations change?" is available.
- **Real** — Folktables (US Census ACS), restricted to five features
  (AGEP, SCHL, WKHP, COW, MAR). TableShift planned for external validity.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python week1_demo.py
```

The first run downloads ~1 GB of public census CSVs into `data/` (gitignored).

## Roadmap

1. Setup
2. Week-1 decisive experiment (go/no-go) — done, inconclusive; strengthen next
3. Validate metrics on synthetic ground truth
4. Full four-property audit on real data (+ TableShift)
5. Early-warning analysis (mixed-effects, vs drift baselines)
6. Figures and tables
7. Write the paper
8. Internal review and revision
9. Citation and reproducibility checks, then submit

## Guardrails

- Strictly temporal/geographic splits, no leakage; tuning on validation only.
- Fixed SHAP background across domains; bootstrap CIs on every metric.
- Pre-register primary comparisons; report null results honestly.
