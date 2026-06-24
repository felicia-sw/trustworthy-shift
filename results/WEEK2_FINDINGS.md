# Week-2 findings — strengthened, pre-registered early-warning test

Run: `week2_powered.py --scale full` (2026-06-24). Pre-registration: `../PREREGISTRATION.md`.
Raw per-shift-pair table: `week2_rows_full.csv`. Full console log: `../results_full.log`.

## Design as run
- 3 tasks (income, pubcov, mobility), restricted feature sets, correct population filters.
- Train on (CA, 2014); test on a (state, year) grid: 12 states × 2014–2018.
- **N = 531 shift-pair rows** = 3 tasks × 3 models × 59 target domains.
- Independent geographic clusters (states) = 12. Domains (state, year) = 59.
- Fixed SHAP background (interventional TreeExplainer / LinearExplainer).
- Primary outcome: accuracy-drop. Robustness: ROC-AUC-drop, PR-AUC-drop.
- Inference: mixed model (state random intercept; task, model fixed) + state cluster
  bootstrap (300/300 successful).

## Headline result — pre-registered NO-GO
The label-free **explanation-shift signal does not carry incremental early-warning
value** beyond the two cheap baselines (covariate shift, prediction shift).

Primary test (acc_drop, global z-scored signals, as pre-registered):
| signal | β (z-scored) | 2-sided p | one-sided p (β>0) |
|--------|-------------:|----------:|------------------:|
| covariate_shift | −0.0034 | 0.123 | — |
| prediction_shift | +0.0030 | 0.085 | — |
| **explanation_shift (ours)** | **−0.0036** | 0.019 | **0.990** |

- Incremental-value LRT (add expl beyond baselines): χ²(1)=5.54, p=0.019 — expl moves the
  fit, but in the **wrong direction** (β<0).
- State cluster-bootstrap 95% CI for β_expl: **[−0.0079, +0.0005]** (straddles/negative).
- Pre-registered rule (GO iff β_expl>0 AND LRT p<0.05 AND CI_lo>0) → **NO-GO**
  (β_expl>0 is False).

## The important nuance — the "wrong direction" is mostly a pooling artifact
Raw pooled Spearman(expl, acc_drop) = −0.267, but this is **not** uniform:

| task | Spearman(expl, acc_drop) | Spearman(expl, auc_drop) |
|------|------------------------:|-------------------------:|
| income | +0.139 | +0.077 |
| mobility | +0.218 | −0.003 |
| pubcov | −0.261 | −0.352 |

By shift type: geographic ρ=−0.482, combined −0.278, temporal +0.063 — i.e. explanation
shift looks most "misleading" exactly under **geographic** covariate movement, the benign
shift that need not hurt accuracy (consistent with the synthetic Part-A mechanism).

When the cross-task confound is removed, the effect collapses to a **clean null**:
- Within-task standardized mixed model: β_expl = −0.0016, LRT p=0.247, CI [−0.0048, +0.0016].
- Per-task incremental tests (controlling for both baselines): β_expl is slightly
  **positive** in all three tasks but non-significant everywhere
  (income LRT p=0.54, mobility 0.19, pubcov 0.28; all bootstrap CIs straddle 0).

**Honest conclusion:** explanation shift is **not robustly anti-predictive** (that was a
Simpson/aggregation illusion); it is simply **uninformative** — it adds no reliable value
over cheaper signals once task and the baselines are accounted for. This is the
pre-committed NULL branch.

## Which label-free signal is best? (practitioner takeaway)
The **cheapest** baseline is the most reliable. `prediction_shift` is consistently
right-signed (positive) and is strongly significant for AUC-drop (β=+0.0039, p<0.001);
`covariate_shift` is mixed; the expensive `explanation_shift` adds nothing. So a SHAP-based
explanation-shift monitor is not worth its compute as an accuracy early-warning here.

## Caveats (report these honestly in the paper)
1. **Small drops / low ceiling.** Accuracy/AUC drops are modest (income especially
   geographically stable); even the baselines are weak predictors. The null is "no
   *incremental* value," in a regime where total predictability is low.
2. **Mobility is near-chance** (in-dist AUC ≈ 0.57); its accuracy is base-rate-dominated,
   so its acc_drop partly reflects label-prevalence shift, not model degradation.
3. **Operationalization-specific.** This tests the *global* explanation-shift metric
   (TV distance of 5-feature mean-|SHAP| vectors, fixed background). The *local* per-instance
   variant (per-feature Wasserstein) was pre-registered as secondary "only if the global
   signal is promising" — it is not, so we do not chase it. The null is specific to the
   global signal as defined.
4. 12 geographic clusters: powered relative to Week-1 (N=8) but still bounded; more states
   would tighten CIs further.

## Implication for the paper
Pivot to the pre-committed Q2 story: a **shift-aware trade-off map** of the four properties,
with a **falsifiable, honestly-null early-warning result** — explanation shift does not earn
its keep as a label-free accuracy warning, and the cheapest drift signal (prediction shift)
is the better monitor. Lead with the (null) early-warning test; the four-property audit is
the supporting infrastructure.
