# Pre-registration — Early-warning test (strengthened, powered run)

Committed **before** the powered modeling run, per the blueprint ("the primary
comparisons are pre-registered before any modeling") and ``. Frozen on
2026-06-24. Anything not specified here, or done after seeing results, is
**exploratory** and will be labeled as such in the paper.

This pre-registers only the **decisive early-warning test** ( §4 / roadmap
step 5). The full four-property audit (calibration, fairness) is a separate,
later pre-registration.

---

## 1. Primary research question

Does a **label-free explanation-shift signal** predict the realized accuracy drop
under distribution shift **beyond** what two cheaper label-free drift baselines
(input-distribution shift, prediction-distribution shift) already explain?

## 2. Frozen design (decided before seeing any powered result)

- **Tasks (3):** ACSIncome, ACSPublicCoverage, ACSMobility — restricted to a small
  fixed feature set each (see `week2_powered.py: TASKS`).
- **Models (3):** logistic regression, random forest, XGBoost. Hyperparameters are
  **fixed a priori** (Week-1-style sensible defaults; see `week2_powered.py:
  get_frozen_models`) and frozen for every shifted evaluation. No per-shift tuning.
- **Shift structure:** train on a single source domain `(state=CA, year=SOURCE_YEAR)`
  per task/model; evaluate on a grid of target domains `(state, year)` spanning
  temporal-only, geographic-only, and combined shift. The unit of analysis is one
  **shift pair** (source → target) for a given (task, model).
- **SHAP:** fixed background (source training subsample), identical across all
  domains, to avoid the reference-distribution confound. A shifted-reference
  variant is **secondary/exploratory**.
- **Signals (all label-free):**
  - `cov_shift` — domain-classifier AUC between source and target inputs (baseline 1).
  - `pred_shift` — KS distance between predicted-probability distributions (baseline 2).
  - `expl_shift` — total-variation distance between normalized SHAP importance
    vectors, source vs target, fixed background (**ours**).
- **Outcome:** `acc_drop = acc(source-val) − acc(target)` (positive = accuracy fell).

## 3. Primary statistical test

A linear **mixed-effects model** on all shift-pair rows (pooled across tasks and
models), signals standardized (z-scored) so coefficients are comparable:

```
acc_drop ~ cov_z + pred_z + expl_z + C(task) + C(model)
           + (random intercept | state)
```

`state` is the random intercept — it is the unit of geographic independence and
the source of repeated measurement (same state across years/tasks/models).
`task` and `model` enter as **fixed** effects: each has only 3 levels, where a
random effect is both statistically inappropriate (variance unidentifiable) and
numerically unstable. The honest counts reported alongside: number of shift-pair
rows, number of (state, year) domains, and **number of independent geographic
clusters (states)** — the last is the real N for geographic generalization, and
the temporal axis adds within-cluster replication, not new clusters.

> **Spec note (pre-data, 2026-06-24):** the blueprint phrased this as "task and
> domain as random effects." A smoke test on *synthetic* rows (no real results
> seen) showed the crossed task+state variance-components estimator fails to
> converge (Hessian not positive-definite). Because task/model have <5 levels,
> the fix is to keep `state` random and make task/model fixed. This refinement was
> made before any real-data run and is logged here for transparency.

Two pre-committed readouts:
1. **Coefficient on `expl_z`**, one-sided test of `H1: β_expl > 0` (more explanation
   shift ⇒ larger accuracy loss). Wrong sign ⇒ fails.
2. **Likelihood-ratio test** comparing the full model vs. the same model **without**
   `expl_z` — does explanation shift add fit *beyond the two baselines*?

Uncertainty: a **cluster bootstrap over states** (resample states with
replacement, relabel each draw as a distinct cluster, refit) gives a 95% CI on
`β_expl`. States are the independent geographic units, so they are the correct
bootstrap cluster. Effect size (β and the LRT improvement) is reported with the
p-value, not instead of it.

## 4. Decision rule (pre-committed)

- **GO** (→ Q1-reachable "early warning works" paper) **iff** all hold:
  `β_expl > 0`, the LRT for adding `expl_z` is significant at α = 0.05, **and** the
  bootstrap 95% CI for `β_expl` excludes 0 on the positive side.
- **NO-GO** (→ Q2 "trade-off map + explanation shift is false-alarm-prone" paper)
  **otherwise**. The null is reported as a finding, never spun as a positive.

α = 0.05. Where multiple comparisons enter (per-task, per-model breakdowns), they
are **secondary** and corrected (Holm); the primary test above is single and
uncorrected.

## 5. Pre-specified secondary / exploratory analyses (labeled as such)

- Per-task and per-model Spearman of each signal vs `acc_drop`.
- Separation by shift type (temporal-only / geographic-only / combined).
- Shifted-reference SHAP variant.
- Synthetic-tier appropriate-vs-spurious decomposition (Part A, already done).
- Local explanation shift (per-feature Wasserstein) — only if global signal is
  promising.

## 6. What would falsify the early-warning claim

If `expl_shift` does **not** beat the baselines under §4, the claim is false and we
say so. Week-1 (N=8) already leaned that way (`expl_shift` Spearman −0.333, wrong
sign, underpowered); this powered run exists to settle it, not to rescue it.
