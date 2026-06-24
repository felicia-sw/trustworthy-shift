"""
Week-3 FOUR-PROPERTY AUDIT (roadmap step: the trade-off map).

Given the pre-registered NO-GO on the early-warning signal (see results/WEEK2_FINDINGS.md),
the paper's main empirical contribution becomes a *shift-aware audit*: under temporal and
geographic shift, do the four trustworthiness properties degrade together, or trade off?

  Properties measured here, per domain (in-distribution source AND every shifted target):
    1. Predictive performance : accuracy, ROC-AUC, PR-AUC
    2. Calibration            : adaptive ECE (equal-count bins), Brier score,
                                adaptive ECE after source-fit isotonic recalibration
    3. Group fairness         : demographic-parity gap and equalized-odds gap,
                                for SEX (M/F) and RACE (White / non-White)
    (4. Explanation stability  : the SHAP explanation-shift metric `expl`, already computed
                                in week2_powered.py -> results/week2_rows_full.csv; joined in
                                the analysis notebook rather than recomputed here.)

Design is IDENTICAL to week2_powered.py --scale full (same source CA-2014, same 12 states ×
2014-2018 grid, same frozen models, same split, same scaler, same row subsampling order),
so this table joins cleanly to the early-warning table on (task, model, state, year).
No SHAP here -> fast (~10 min on cached data).

Run:
    .venv/bin/python week3_audit.py
Output: results/week3_audit_full.csv  (one row per task × model × target domain)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

from week2_powered import build_tasks, get_frozen_models, safe_auc, shift_type, RESULTS_DIR
from week1_demo import have
from sklearn.metrics import roc_auc_score, average_precision_score

# Same grid as week2_powered.py --scale full (keep in lock-step).
SOURCE_STATE, SOURCE_YEAR = "CA", "2014"
STATES = ["CA", "TX", "NY", "FL", "PA", "OH", "IL", "GA", "NC", "WA", "AZ", "MA"]
YEARS = ["2014", "2015", "2016", "2017", "2018"]
SOURCE_CAP, TARGET_CAP = 20000, 8000


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def adaptive_ece(p, y, n_bins=15):
    """Adaptive (equal-count) Expected Calibration Error: robust to binning choice.
    Sort by predicted prob, split into equal-size bins, average |confidence - accuracy|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p)
    p, y = p[order], y[order]
    e = 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b) == 0:
            continue
        e += (len(b) / len(p)) * abs(p[b].mean() - y[b].mean())
    return e


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# --------------------------------------------------------------------------- #
# Group-fairness metrics (binary protected group g in {0,1})
# --------------------------------------------------------------------------- #
def fairness_gaps(yhat, y, g):
    """Demographic-parity gap and equalized-odds gap between the two groups.
    DP gap  = |P(yhat=1|g=1) - P(yhat=1|g=0)|.
    EO gap  = mean(|TPR diff|, |FPR diff|).  NaN-safe when a cell is empty."""
    yhat, y, g = np.asarray(yhat), np.asarray(y), np.asarray(g)

    def rate(mask):
        return yhat[mask].mean() if mask.any() else np.nan

    dp = abs(rate(g == 1) - rate(g == 0))
    tpr1, tpr0 = rate((g == 1) & (y == 1)), rate((g == 0) & (y == 1))
    fpr1, fpr0 = rate((g == 1) & (y == 0)), rate((g == 0) & (y == 0))
    eo = np.nanmean([abs(tpr1 - tpr0), abs(fpr1 - fpr0)])
    return float(dp), float(eo)


# --------------------------------------------------------------------------- #
# Data loading: X, y, and the protected attributes, sampled IDENTICALLY to week2
# --------------------------------------------------------------------------- #
def load_with_prot(problem, state, year, cap, rng):
    from folktables import ACSDataSource
    ds = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person")
    df = ds.get_data(states=[state], download=True)
    X, y, _ = problem.df_to_numpy(df)
    fdf = problem._preprocess(df)                 # same filtered rows df_to_numpy uses
    X, y = X.astype(np.float32), y.astype(int)
    sex = (fdf["SEX"].to_numpy() == 2).astype(int)        # 1 = Female
    rac = (fdf["RAC1P"].to_numpy() != 1).astype(int)      # 1 = non-White
    if len(X) > cap:
        s = rng.choice(len(X), cap, replace=False)        # SAME call as week2 load_raw
        X, y, sex, rac = X[s], y[s], sex[s], rac[s]
    return X, y, sex, rac


def props(p, yhat, y, sex, rac):
    """All shift-sensitive properties for one domain, given predicted probs p."""
    dp_s, eo_s = fairness_gaps(yhat, y, sex)
    dp_r, eo_r = fairness_gaps(yhat, y, rac)
    return dict(
        acc=accuracy_score(y, yhat), auc=safe_auc(y, p, roc_auc_score),
        ap=safe_auc(y, p, average_precision_score),
        brier=brier(p, y), ace=adaptive_ece(p, y),
        dp_sex=dp_s, eo_sex=eo_s, dp_rac=dp_r, eo_rac=eo_r,
    )


# --------------------------------------------------------------------------- #
def run():
    tasks = build_tasks()
    rng = np.random.default_rng(0)        # same seed/order as week2 -> identical samples
    rows = []

    for tname, problem in tasks.items():
        print(f"\n=== TASK {tname} ===")
        Xs_raw, ys, sex_s, rac_s = load_with_prot(problem, SOURCE_STATE, SOURCE_YEAR,
                                                  SOURCE_CAP, rng)
        sc = StandardScaler().fit(Xs_raw[: int(0.8 * len(Xs_raw))])
        Xs = sc.transform(Xs_raw)
        cut = int(0.8 * len(Xs))
        Xtr, ytr = Xs[:cut], ys[:cut]
        Xval, yval = Xs[cut:], ys[cut:]
        sex_val, rac_val = sex_s[cut:], rac_s[cut:]

        # load every target once (reuse across models); same order as week2
        targets = {}
        for st in STATES:
            for yr in YEARS:
                if st == SOURCE_STATE and yr == SOURCE_YEAR:
                    continue
                Xt_raw, yt, sex_t, rac_t = load_with_prot(problem, st, yr, TARGET_CAP, rng)
                targets[(st, yr)] = (sc.transform(Xt_raw), yt, sex_t, rac_t)
        print(f"  source val n={len(yval)}, target domains={len(targets)}")

        for mname, model in get_frozen_models().items():
            model.fit(Xtr, ytr)
            p_val = model.predict_proba(Xval)[:, 1]
            src = props(p_val, (p_val >= 0.5).astype(int), yval, sex_val, rac_val)

            # recalibrator learned on SOURCE val only (no leakage), applied to targets
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, yval)

            for (st, yr), (Xt, yt, sex_t, rac_t) in targets.items():
                p_t = model.predict_proba(Xt)[:, 1]
                tgt = props(p_t, (p_t >= 0.5).astype(int), yt, sex_t, rac_t)
                ace_iso_tgt = adaptive_ece(iso.transform(p_t), yt)
                rows.append(dict(
                    task=tname, model=mname, state=st, year=int(yr),
                    shift=shift_type(st, yr, SOURCE_STATE, SOURCE_YEAR),
                    n_target=len(yt), ace_iso_tgt=ace_iso_tgt,
                    **{f"{k}_src": src[k] for k in src},
                    **{f"{k}_tgt": tgt[k] for k in tgt},
                ))
            print(f"  [{mname}] src acc={src['acc']:.3f} ace={src['ace']:.3f} "
                  f"dp_sex={src['dp_sex']:.3f} -> scored {len(targets)} domains")

    return pd.DataFrame(rows)


def main():
    if not have("folktables"):
        print("needs folktables. pip install -r requirements.txt")
        return
    df = run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "week3_audit_full.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows, {len(df.columns)} columns)")

    # quick honest peek: how do the four properties move under shift (mean change)?
    chg = pd.DataFrame({
        "accuracy drop":   (df["acc_src"] - df["acc_tgt"]),
        "ECE change":      (df["ace_tgt"] - df["ace_src"]),
        "DP-gap change (sex)": (df["dp_sex_tgt"] - df["dp_sex_src"]),
        "DP-gap change (race)":(df["dp_rac_tgt"] - df["dp_rac_src"]),
    })
    print("\nMean change under shift (positive = degraded):")
    print(chg.mean().round(4).to_string())


if __name__ == "__main__":
    main()
