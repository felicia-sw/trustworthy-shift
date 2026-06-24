# Week-3 findings — four-property shift audit (the trade-off map)

Run: `week3_audit.py` (2026-06-24). Table: `week3_audit_full.csv` (531 rows = 3 tasks × 3
models × 59 target domains). Same grid / frozen models / sampling as the early-warning run, so
it joins to `week2_rows_full.csv` on (task, model, state, year). Presentation:
`notebooks/02_trade_off_audit.ipynb`.

## What was measured
Per domain (in-distribution source **and** every shifted target), for 3 models on 3 tasks:
- **Performance:** accuracy, ROC-AUC, PR-AUC.
- **Calibration:** adaptive ECE (equal-count bins), Brier; plus ECE after a source-fit isotonic
  recalibration (does in-distribution calibration transfer under shift?).
- **Group fairness:** demographic-parity gap and equalized-odds gap, for SEX (M/F) and
  RACE (White / non-White).
- **Explanation stability:** the SHAP explanation-shift metric from the early-warning table.

## Headline — under shift, every property degrades, and not in lock-step
Mean change under shift (positive = degraded), pooled over all domains:

| property | mean change |
|----------|------------:|
| accuracy drop | +0.018 |
| **calibration error (ECE) increase** | **+0.022** |
| demographic-parity gap, sex | +0.017 |
| demographic-parity gap, race | +0.017 |

Calibration degrades **more** than accuracy — accuracy alone understates trustworthiness loss.

## The trade-off structure (Spearman corr. of per-domain *changes*)
|  | acc_drop | auc_drop | ECE+ | DPgap_sex+ | DPgap_race+ |
|--|--:|--:|--:|--:|--:|
| acc_drop | 1.00 | 0.49 | **0.02** | **−0.20** | **−0.21** |
| auc_drop | 0.49 | 1.00 | 0.06 | 0.27 | 0.22 |
| ECE+ | 0.02 | 0.06 | 1.00 | 0.02 | −0.02 |
| DPgap_sex+ | −0.20 | 0.27 | 0.02 | 1.00 | 0.57 |
| DPgap_race+ | −0.21 | 0.22 | −0.02 | 0.57 | 1.00 |

Reading: performance metrics cluster; the two fairness metrics cluster (+0.57); **calibration is
statistically independent of accuracy (+0.02)**; and **fairness gaps trade off against accuracy
(−0.20/−0.21)**. The four properties are not a single axis — a model can hold accuracy while
calibration and fairness move independently.

## The decisive "accuracy is a weak proxy" result
Restricting to the **86 domains where accuracy barely changed** (|Δacc| < 1pp): calibration still
worsened (mean ECE +0.0098) and fairness gaps still widened (sex +0.026, race +0.021). **Watching
accuracy alone would miss real, silent calibration and fairness degradation.** This is the core
motivation for the multi-property audit and the paper's main descriptive contribution.

## Heterogeneity (where the action is, for the trade-off map figure)
Per-task mean change:
| task | Δacc | ΔECE | ΔDPgap_sex | ΔDPgap_race |
|------|----:|----:|----:|----:|
| income | +0.018 | +0.023 | +0.042 | +0.046 |
| mobility | +0.044 | +0.014 | +0.002 | −0.002 |
| pubcov | −0.009 | +0.028 | +0.006 | +0.006 |

- **income** degrades on everything; fairness gaps widen most.
- **mobility** loses the most accuracy but fairness barely moves (its predictions hardly depend
  on sex/race).
- **pubcov** accuracy slightly *improves* yet calibration degrades most — a pure "accuracy fine,
  calibration broken" case.

Per-model: the simple **logreg** degrades least on calibration (+0.016) and fairness; **rf/xgb**
degrade more (ECE +0.027/+0.022; larger fairness gaps). Model complexity costs
trustworthiness-robustness under shift.

Recalibration: source-fit isotonic only nudges target ECE (0.055 → 0.051) — in-distribution
calibration does **not** transfer well under shift.

## Caveats (carry into the paper)
1. **The accuracy↔fairness trade-off may be partly mechanical:** a model degrading toward random
   predictions naturally exhibits smaller group disparities, so a shrinking gap is not
   necessarily a benefit. Reported as a correlation, flagged explicitly, not sold as a free lunch.
2. Effect sizes are modest (correlations |0.2–0.5|); the strong claim is the **decoupling**
   (accuracy carries little information about calibration/fairness), which the |Δacc|<1pp slice
   shows most cleanly.
3. US-census tasks, one data family; `mobility` is near-chance (AUC ≈ 0.57). Group fairness uses
   binarized SEX and RACE (White / non-White); age is a feature / population filter, not a
   fairness axis here.

## Where this sits in the paper
Combined with the pre-registered early-warning **NO-GO**, this audit is the empirical core of the
Q2 paper: a **shift-aware trade-off map** showing the four properties degrade independently
(accuracy is a weak proxy), plus a falsifiable null on the label-free explanation-shift early
warning. Lead with the null; the trade-off map is the supporting contribution.
