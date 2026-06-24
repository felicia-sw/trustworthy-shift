# Week-4 findings — strong label-free baselines (peer-review critical #1)

Run: `week4_baselines.py` (2026-06-24). Tables: `week4_baselines.csv`,
`week4_baselines_joined.csv`. Same grid / frozen models / sampling as the early-warning run
(sample-alignment check vs `week2_rows_full.csv`: max |Δacc_drop| = 0.0000).

## Why
Reviewers noted the early-warning test compared explanation shift only against two *cheap* drift
statistics (covariate shift, prediction shift), a low bar. We add the standard **unsupervised
accuracy estimators** as strong label-free competitors:
- **ATC** (Average Thresholded Confidence): threshold source confidence at the source error rate;
  estimate target accuracy as the fraction of target points above it → a label-free *accuracy-drop
  estimate*.
- **DoC** (Difference of Confidence): mean source confidence − mean target confidence.

## Result — explanation shift loses to strong baselines and adds nothing
Spearman correlation with the realized accuracy drop (pooled, N=531):

| signal | ρ vs accuracy drop |
|--------|-------------------:|
| **ATC accuracy-drop estimate (strong)** | **+0.346** |
| difference-of-confidence (strong) | +0.236 |
| prediction shift (cheap) | +0.171 |
| covariate shift (cheap) | +0.119 |
| **explanation shift (ours)** | **−0.267** |

ATC and DoC are the strongest label-free predictors; explanation shift is the **only** signal
negatively associated with accuracy drop (the pooled negative is the Simpson artifact discussed in
the early-warning findings; the point here is that it is nowhere near competitive).

Pre-registered incremental test, now against the **strong** baseline set {covariate, prediction,
DoC, ATC}: explanation shift's coefficient is negative (β = −0.0041, one-sided p(β>0) = 0.995); the
likelihood-ratio test is "significant" only in the wrong direction (p = 0.0098). **Explanation
shift adds no positive early-warning value beyond strong baselines.**

## Takeaway (sharpens the paper)
The reviewer concern is answered and the conclusion is strengthened: for label-free monitoring of
accuracy under shift, use a dedicated accuracy estimator (ATC is best here); SHAP-based explanation
shift is not competitive and is not worth its compute. This converts "explanation shift doesn't
beat cheap stats" into the stronger "explanation shift loses even to the proper label-free
accuracy estimators."

## Caveat
The multivariate incremental coefficients among the confidence-based signals (prediction, DoC, ATC)
are affected by collinearity, so we report each signal's standalone predictive strength (Spearman)
as the primary comparison and treat the incremental coefficients as secondary. ATC/DoC are added as
*comparators*, not as a new contribution. References to add to the draft: Garg et al. (ATC) and
Guillory et al. (DoC) — to be verified at citation check.
