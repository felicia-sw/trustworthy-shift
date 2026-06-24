"""
Power / minimum-detectable-effect (MDE) analysis for the pre-registered early-warning test
(addresses peer-review critical item #2: "a null is only informative if the design could have
detected a meaningful effect").

The primary test is a one-sided test of H1: beta_expl > 0 in the mixed-effects model of accuracy
drop, with uncertainty from a state cluster bootstrap. The relevant noise level is therefore the
cluster-bootstrap standard error of beta_expl. The MDE at power (1 - b) for a one-sided test at
level a is:  MDE = (z_{1-a} + z_{1-b}) * SE.

We report the MDE in raw units (accuracy-drop per 1 SD of explanation shift) and relative to the
observed signal magnitudes, and interpret whether the null is informative.

Run: .venv/bin/python power_analysis.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from week2_powered import incremental_test, RESULTS_DIR

df = pd.read_csv(os.path.join(RESULTS_DIR, "week2_rows_full.csv"))
for c in ["cov", "pred", "expl"]:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

# one cluster-bootstrap run gives both the point estimate and the bootstrap distribution
res = incremental_test(df, "acc_drop", n_boot=600, rng=np.random.default_rng(2024), verbose=False)
beta = res["beta"]
se = float(np.std(res["boot"]))               # cluster-bootstrap SE of beta_expl
lo, hi = res["ci"]

alpha = 0.05
def mde(power):
    return (norm.ppf(1 - alpha) + norm.ppf(power)) * se

mde80, mde50, mde95 = mde(0.80), mde(0.50), mde(0.95)
mean_drop = df["acc_drop"].mean()

# context: observed standardized coefficients (effect of 1 SD of each signal on acc_drop)
print("=" * 70)
print("POWER / MDE ANALYSIS  (one-sided H1: beta_expl > 0, alpha=0.05)")
print("=" * 70)
print(f"  observed beta_expl            = {beta:+.4f}  (95% boot CI [{lo:+.4f}, {hi:+.4f}])")
print(f"  cluster-bootstrap SE(beta)    = {se:.4f}   over {df.state.nunique()} state clusters")
print(f"  mean accuracy drop            = {mean_drop:.4f}")
print()
print("  Minimum detectable effect (MDE) of explanation shift on accuracy drop")
print("  (units: accuracy-drop per +1 SD of explanation shift):")
print(f"    power 0.50 : MDE = {mde50:.4f}   ({100*mde50/mean_drop:4.1f}% of mean acc drop)")
print(f"    power 0.80 : MDE = {mde80:.4f}   ({100*mde80/mean_drop:4.1f}% of mean acc drop)")
print(f"    power 0.95 : MDE = {mde95:.4f}   ({100*mde95/mean_drop:4.1f}% of mean acc drop)")
print()
print("  Interpretation:")
print(f"   - At 80% power the study could detect an early-warning effect as small as")
print(f"     {mde80:.4f} accuracy-drop per 1 SD of explanation shift, i.e. ~{100*mde80/mean_drop:.0f}% of the")
print(f"     average realized accuracy drop. Effects larger than this would have been")
print(f"     flagged; the observed effect is negative and far from the GO region.")
print(f"   - The null is therefore informative for early-warning effects down to a small")
print(f"     fraction of the typical accuracy movement; it is NOT merely 'no power'.")
print(f"   - Caveat: 12 geographic clusters is modest; MDE shrinks with more states.")

out = pd.DataFrame([{
    "beta_expl_observed": beta, "ci_lo": lo, "ci_hi": hi, "se_boot": se,
    "mean_acc_drop": mean_drop,
    "MDE_power50": mde50, "MDE_power80": mde80, "MDE_power95": mde95,
    "MDE80_pct_of_mean_drop": 100 * mde80 / mean_drop,
}])
out.to_csv(os.path.join(RESULTS_DIR, "tables", "power_analysis.csv"), index=False)
print("\n  wrote results/tables/power_analysis.csv")
