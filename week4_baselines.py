"""
Week-4 STRONG label-free baselines (peer-review critical item #1).

The early-warning test originally compared explanation shift only against two cheap drift
statistics (covariate shift, prediction shift). Reviewers correctly note these are a low bar.
The real label-free competitors are *unsupervised accuracy estimators*:

  - ATC  (Average Thresholded Confidence, Garg et al. 2022): pick a confidence threshold on the
          source so that the fraction below it equals the source error rate; estimate target
          accuracy as the fraction of target points above that threshold.
  - DoC  (Difference of Confidence, Guillory et al. 2021): mean source confidence minus mean
          target confidence.

Both yield a label-free *estimate of the accuracy drop* and are far stronger than drift stats.
We add them as signals and re-run the pre-registered incremental-value test: does explanation
shift add anything beyond the strong baselines? (Same grid / frozen models / sampling order as
week2_powered.py + week3_audit.py, so it joins on (task, model, state, year).)

Run: .venv/bin/python week4_baselines.py
Output: results/week4_baselines.csv + a console comparison.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from week2_powered import build_tasks, get_frozen_models, shift_type, incremental_test, RESULTS_DIR
from week3_audit import load_with_prot, SOURCE_STATE, SOURCE_YEAR, STATES, YEARS, SOURCE_CAP, TARGET_CAP


def confidence(p):
    """Binary confidence score = max(p, 1-p)."""
    return np.maximum(p, 1 - p)


def atc_threshold(conf_val, err_rate):
    """Threshold t such that P(conf < t) on source val == source error rate."""
    return np.quantile(conf_val, err_rate) if err_rate > 0 else conf_val.min() - 1e-9


def run():
    tasks = build_tasks()
    rng = np.random.default_rng(0)          # same order/seed as week3 -> identical samples
    rows = []
    for tname, problem in tasks.items():
        Xs_raw, ys, _, _ = load_with_prot(problem, SOURCE_STATE, SOURCE_YEAR, SOURCE_CAP, rng)
        sc = StandardScaler().fit(Xs_raw[: int(0.8 * len(Xs_raw))])
        Xs = sc.transform(Xs_raw); cut = int(0.8 * len(Xs))
        Xtr, ytr, Xval, yval = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]
        targets = {}
        for st in STATES:
            for yr in YEARS:
                if st == SOURCE_STATE and yr == SOURCE_YEAR:
                    continue
                Xt_raw, yt, _, _ = load_with_prot(problem, st, yr, TARGET_CAP, rng)
                targets[(st, yr)] = (sc.transform(Xt_raw), yt)
        for mname, model in get_frozen_models().items():
            model.fit(Xtr, ytr)
            p_val = model.predict_proba(Xval)[:, 1]
            acc_src = accuracy_score(yval, model.predict(Xval))
            conf_val = confidence(p_val)
            t = atc_threshold(conf_val, 1 - acc_src)        # ATC threshold on source val
            for (st, yr), (Xt, yt) in targets.items():
                p_t = model.predict_proba(Xt)[:, 1]
                conf_t = confidence(p_t)
                est_acc = np.mean(conf_t >= t)              # ATC estimated target accuracy
                atc = acc_src - est_acc                      # ATC estimated accuracy DROP
                doc = conf_val.mean() - conf_t.mean()        # difference of confidence
                acc_drop = acc_src - accuracy_score(yt, model.predict(Xt))
                rows.append(dict(task=tname, model=mname, state=st, year=int(yr),
                                 shift=shift_type(st, yr, SOURCE_STATE, SOURCE_YEAR),
                                 atc=atc, doc=doc, acc_drop_check=acc_drop))
    return pd.DataFrame(rows)


def analyze(df):
    from scipy.stats import spearmanr
    early = pd.read_csv(os.path.join(RESULTS_DIR, "week2_rows_full.csv"))
    d = early.merge(df[["task", "model", "state", "year", "atc", "doc"]],
                    on=["task", "model", "state", "year"], how="left")
    for c in ["cov", "pred", "expl", "atc", "doc"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()

    print("\n" + "=" * 70)
    print("SIGNAL STRENGTH vs realized accuracy drop (Spearman, pooled)")
    print("=" * 70)
    for c, lbl in [("cov", "covariate shift (cheap)"), ("pred", "prediction shift (cheap)"),
                   ("doc", "difference-of-confidence (STRONG)"),
                   ("atc", "ATC accuracy-drop estimate (STRONG)"),
                   ("expl", "explanation shift (ours)")]:
        rho, p = spearmanr(d[c], d["acc_drop"])
        print(f"  {lbl:38s} rho={rho:+.3f}  p={p:.4f}")

    # Does ATC add value beyond the cheap baselines? (validates it as a strong baseline)
    print("\n  --- Does the STRONG baseline (ATC) beat the cheap ones? ---")
    r_atc = _incr(d, "atc_z", ["cov_z", "pred_z"])
    print(f"    ATC beyond covariate+prediction: beta={r_atc['beta']:+.4f}  LRT p={r_atc['p_lrt']:.4f}")

    # Pre-registered question, now against the STRONG baseline set
    print("\n  --- Does EXPLANATION SHIFT add value beyond the STRONG baselines? ---")
    r_expl = _incr(d, "expl_z", ["cov_z", "pred_z", "doc_z", "atc_z"])
    print(f"    explanation shift beyond {{cov,pred,doc,atc}}: beta={r_expl['beta']:+.4f}  "
          f"one-sided p(>0)={r_expl['p_one']:.4f}  LRT p={r_expl['p_lrt']:.4f}")
    verdict = ("explanation shift adds NOTHING beyond strong baselines"
               if not (r_expl["beta"] > 0 and r_expl["p_lrt"] < 0.05)
               else "explanation shift adds value")
    print(f"\n  ==> {verdict}.")
    d.to_csv(os.path.join(RESULTS_DIR, "week4_baselines_joined.csv"), index=False)
    return r_atc, r_expl


def _incr(d, target_signal, base_signals):
    """LRT + one-sided test for adding `target_signal` beyond `base_signals` (state RE; task,model fixed)."""
    import statsmodels.formula.api as smf, warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from scipy.stats import chi2, norm
    rhs = " + ".join(base_signals + [target_signal]) + " + C(task) + C(model)"
    rhs0 = " + ".join(base_signals) + " + C(task) + C(model)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        full = smf.mixedlm(f"acc_drop ~ {rhs}", d, groups=d["state"]).fit(reml=False)
        base = smf.mixedlm(f"acc_drop ~ {rhs0}", d, groups=d["state"]).fit(reml=False)
    beta, se = full.params[target_signal], full.bse[target_signal]
    return dict(beta=beta, p_one=norm.sf(beta / se),
                p_lrt=chi2.sf(max(2 * (full.llf - base.llf), 0), df=1))


if __name__ == "__main__":
    df = run()
    df.to_csv(os.path.join(RESULTS_DIR, "week4_baselines.csv"), index=False)
    print(f"wrote results/week4_baselines.csv ({len(df)} rows)")
    # sanity: our recomputed acc_drop should match week2's
    early = pd.read_csv(os.path.join(RESULTS_DIR, "week2_rows_full.csv"))
    m = df.merge(early[["task", "model", "state", "year", "acc_drop"]],
                 on=["task", "model", "state", "year"])
    print(f"sample-alignment check: max |acc_drop diff vs week2| = "
          f"{(m['acc_drop_check'] - m['acc_drop']).abs().max():.4f}  (should be ~0)")
    analyze(df)
