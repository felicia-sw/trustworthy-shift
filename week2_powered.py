"""
Week-2 STRENGTHENED early-warning test (roadmap step 5).

Week-1's real-data go/no-go was underpowered: 8 states, one year, one task, tiny
accuracy drops -> a single Spearman over 8 points (explanation_shift rho = -0.333,
wrong sign, not significant). This script fixes the power problem and runs the
PRE-REGISTERED primary test (see PREREGISTRATION.md). Nothing here is tuned to the
result; the design and decision rule were frozen first.

What changes vs week1_demo.py:
  - 3 tasks (ACSIncome, ACSPublicCoverage, ACSMobility), each restricted to a
    small fixed feature set, with their proper population filters.
  - Temporal x geographic domains: train on one (state, year); test on a grid of
    (state, year) -> temporal-only, geographic-only, and combined shift pairs.
    N goes from 8 to dozens-hundreds of independent shift pairs.
  - Primary analysis is a MIXED-EFFECTS model (task & state as crossed random
    intercepts) testing whether explanation shift adds predictive value for the
    accuracy drop BEYOND the two cheap baselines -- not a raw Spearman.
  - Cluster bootstrap over domains for a 95% CI on the explanation-shift effect.
  - Fixed SHAP background across all domains (reference-distribution control).

Run:
    .venv/bin/python week2_powered.py --scale pilot   # fast, validates pipeline
    .venv/bin/python week2_powered.py --scale full    # powered; downloads census

The metric functions (shap_importance, explanation_shift, covariate_shift,
prediction_shift) are imported unchanged from week1_demo.py -- one source of truth.
"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# Reuse the Week-1 metric implementations verbatim (no reinventing).
from week1_demo import (
    shap_importance, explanation_shift, covariate_shift, prediction_shift, have,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --------------------------------------------------------------------------- #
# Frozen models (fixed a priori; NOT tuned per shift -- see PREREGISTRATION.md)
# --------------------------------------------------------------------------- #
def get_frozen_models():
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     random_state=0, n_jobs=-1),
    }
    if have("xgboost"):
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
            eval_metric="logloss", random_state=0,
        )
    else:
        print("  (xgboost not installed -> skipping XGB)")
    return models


# --------------------------------------------------------------------------- #
# Restricted tasks: small fixed feature sets + correct population filters.
# Built lazily inside main() so importing folktables is not required to read this.
# --------------------------------------------------------------------------- #
def build_tasks():
    from folktables import BasicProblem, adult_filter

    def public_coverage_filter(df):
        df = df[df["AGEP"] < 65]
        df = df[df["PINCP"] <= 30000]
        return df

    def mobility_filter(df):
        return df.drop(df.loc[(df["AGEP"] <= 18) | (df["AGEP"] >= 35)].index)

    nan = lambda x: np.nan_to_num(x, nan=-1)
    return {
        # income > $50k; restricted to the 5 project features.
        "income": BasicProblem(
            features=["AGEP", "SCHL", "WKHP", "COW", "MAR"],
            target="PINCP", target_transform=lambda x: x > 50000,
            group="RAC1P", preprocess=adult_filter, postprocess=nan),
        # on public health insurance; low-income, under-65 population.
        "pubcov": BasicProblem(
            features=["AGEP", "SCHL", "MAR", "DIS", "ESR"],
            target="PUBCOV", target_transform=lambda x: x == 1,
            group="RAC1P", preprocess=public_coverage_filter, postprocess=nan),
        # moved address in the last year; young-adult population.
        "mobility": BasicProblem(
            features=["AGEP", "SCHL", "MAR", "WKHP", "COW"],
            target="MIG", target_transform=lambda x: x == 1,
            group="RAC1P", preprocess=mobility_filter, postprocess=nan),
    }


# --------------------------------------------------------------------------- #
# Data loading (one full census file per (state, year), capped after load)
# --------------------------------------------------------------------------- #
def load_raw(problem, state, year, cap, rng):
    from folktables import ACSDataSource
    ds = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person")
    df = ds.get_data(states=[state], download=True)
    X, y, _ = problem.df_to_numpy(df)
    X, y = X.astype(np.float32), y.astype(int)
    if len(X) > cap:
        s = rng.choice(len(X), cap, replace=False)
        X, y = X[s], y[s]
    return X, y


def safe_auc(y, p, fn):
    """ROC-AUC / average-precision, robust to a degenerate single-class domain."""
    try:
        return fn(y, p)
    except ValueError:
        return np.nan


def shift_type(state, year, src_state, src_year):
    same_state = state == src_state
    same_year = str(year) == str(src_year)
    if same_state and not same_year:
        return "temporal"
    if not same_state and same_year:
        return "geographic"
    return "combined"


# --------------------------------------------------------------------------- #
# Build the per-shift-pair table (the rows fed to the mixed model)
# --------------------------------------------------------------------------- #
def run_grid(cfg):
    tasks = build_tasks()
    if cfg.tasks:
        tasks = {k: tasks[k] for k in cfg.tasks if k in tasks}
    rng = np.random.default_rng(0)
    rows = []

    for tname, problem in tasks.items():
        print(f"\n{'='*70}\nTASK: {tname}\n{'='*70}")
        # ---- source domain: train/val split, scaler fit on TRAIN only ----
        Xs_raw, ys = load_raw(problem, cfg.source_state, cfg.source_year,
                              cfg.source_cap, rng)
        sc = StandardScaler().fit(Xs_raw[: int(0.8 * len(Xs_raw))])
        Xs = sc.transform(Xs_raw)
        cut = int(0.8 * len(Xs))
        Xtr, ytr, Xval, yval = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]
        bg = Xtr[:200]  # FIXED SHAP background, identical for every domain
        print(f"  source = ({cfg.source_state}, {cfg.source_year})  "
              f"train={len(Xtr)} val={len(Xval)}  pos-rate={ytr.mean():.3f}")

        # ---- precompute target domains once (reused across all models) ----
        targets = {}
        cov_cache = {}  # covariate shift is model-independent -> compute once
        for st in cfg.states:
            for yr in cfg.years:
                if st == cfg.source_state and str(yr) == str(cfg.source_year):
                    continue  # that's the source, not a shift pair
                try:
                    Xt_raw, yt = load_raw(problem, st, yr, cfg.target_cap, rng)
                except Exception as e:
                    print(f"  ! skip ({st},{yr}): {e}")
                    continue
                Xt = sc.transform(Xt_raw)
                targets[(st, yr)] = (Xt, yt)
                cov_cache[(st, yr)] = covariate_shift(Xtr[:4000], Xt[:4000])
        print(f"  target domains loaded: {len(targets)}")

        # ---- per model: freeze, then score every target domain ----
        for mname, model in get_frozen_models().items():
            model.fit(Xtr, ytr)
            imp_src = shap_importance(model, bg, Xval[:300])
            p_val = model.predict_proba(Xval)[:, 1]
            acc_src = accuracy_score(yval, model.predict(Xval))
            auc_src = safe_auc(yval, p_val, roc_auc_score)
            ap_src = safe_auc(yval, p_val, average_precision_score)
            for (st, yr), (Xt, yt) in targets.items():
                p_t = model.predict_proba(Xt)[:, 1]
                acc_drop = acc_src - accuracy_score(yt, model.predict(Xt))
                # higher-is-worse drops: positive = the property degraded on target
                auc_drop = auc_src - safe_auc(yt, p_t, roc_auc_score)
                ap_drop = ap_src - safe_auc(yt, p_t, average_precision_score)
                pred = prediction_shift(model, Xval, Xt)
                expl = explanation_shift(imp_src, shap_importance(model, bg, Xt[:300]))
                rows.append(dict(
                    task=tname, model=mname, state=st, year=int(yr),
                    shift=shift_type(st, yr, cfg.source_state, cfg.source_year),
                    acc_src=acc_src, acc_drop=acc_drop,
                    auc_drop=auc_drop, ap_drop=ap_drop,
                    cov=cov_cache[(st, yr)], pred=pred, expl=expl, n_target=len(yt),
                ))
            print(f"  [{mname}] in-dist acc={acc_src:.3f} auc={auc_src:.3f}  "
                  f"scored {len(targets)} domains")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Primary analysis: mixed-effects, incremental value of explanation shift
# --------------------------------------------------------------------------- #
def fit_mixed(df, outcome="acc_drop"):
    """Mixed model: random intercept for STATE (geographic clustering / repeated
    measurement of the same state across years), with task and model as fixed
    effects. Task/model have only 3 levels each -> fixed is both more stable and
    statistically more appropriate than treating them as random. Returns the full
    fit and the no-explanation-shift baseline fit (for the incremental-value LRT)."""
    import statsmodels.formula.api as smf
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        full = smf.mixedlm(f"{outcome} ~ cov_z + pred_z + expl_z + C(task) + C(model)",
                           df, groups=df["state"]).fit(reml=False)
        base = smf.mixedlm(f"{outcome} ~ cov_z + pred_z + C(task) + C(model)",
                           df, groups=df["state"]).fit(reml=False)
    return full, base


def incremental_test(df, outcome, n_boot, rng, verbose=True):
    """Does explanation shift add predictive value for `outcome` beyond the two
    baselines? Returns beta_expl, one-sided p, LRT p, and a state cluster-bootstrap
    95% CI on beta_expl. Drops rows where the outcome is undefined (NaN AUC)."""
    from scipy.stats import chi2, norm
    d = df.dropna(subset=[outcome]).copy()

    full, base = fit_mixed(d, outcome)
    beta, se = full.params["expl_z"], full.bse["expl_z"]
    z = beta / se; p_one = norm.sf(z)                  # one-sided H1: beta_expl > 0
    lr = 2 * (full.llf - base.llf); p_lrt = chi2.sf(max(lr, 0), df=1)

    if verbose:
        print(f"\n  Fixed-effect coefficients (outcome = {outcome}; signals z-scored):")
        for c in ["cov_z", "pred_z", "expl_z"]:
            print(f"    {c:8s}  beta={full.params[c]:+.4f}  se={full.bse[c]:.4f}  "
                  f"p(2-sided)={full.pvalues[c]:.4f}")
        print(f"  explanation shift, one-sided H1 (beta_expl > 0): "
              f"beta={beta:+.4f}  z={z:+.2f}  p={p_one:.4f}")
        print(f"  LRT add expl_z beyond baselines: chi2(1)={lr:.2f}  p={p_lrt:.4f}")

    # ---- cluster bootstrap over STATES (the independent geographic units) ----
    state_rows = {s: d.index[d.state == s].to_numpy() for s in d.state.unique()}
    states = list(state_rows)
    betas = []
    for _ in range(n_boot):
        pick = rng.choice(states, size=len(states), replace=True)
        frames = []
        for k, s in enumerate(pick):
            sub = d.loc[state_rows[s]].copy()
            sub["state"] = f"bs{k}"          # distinct cluster id per draw
            frames.append(sub)
        bs = pd.concat(frames, ignore_index=True)
        try:
            f, _ = fit_mixed(bs, outcome)
            betas.append(f.params["expl_z"])
        except Exception:
            continue
    if betas:
        lo, hi = np.percentile(betas, [2.5, 97.5])
        if verbose:
            print(f"  cluster bootstrap ({len(betas)}/{n_boot} ok) 95% CI beta_expl: "
                  f"[{lo:+.4f}, {hi:+.4f}]")
    else:
        lo = hi = float("nan")
        if verbose:
            print("  cluster bootstrap: no successful refits")
    return dict(outcome=outcome, n=len(d), beta=beta, p_one=p_one, p_lrt=p_lrt,
                ci=(lo, hi), boot=np.asarray(betas))


def analyze(df, n_boot, rng):
    from scipy.stats import spearmanr

    # standardize the three signals so coefficients are comparable
    for c in ["cov", "pred", "expl"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-12)

    n_pairs = len(df)
    n_domains = df.groupby(["state", "year"]).ngroups
    n_states = df.state.nunique()
    print(f"\n{'='*70}\nPRIMARY TEST (pre-registered)  outcome = acc_drop\n{'='*70}")
    print(f"  shift-pair rows N = {n_pairs}   (state,year) domains = {n_domains}   "
          f"independent geographic clusters (states) = {n_states}")
    print(f"  tasks = {df.task.nunique()}  models = {df.model.nunique()}")

    res = incremental_test(df, "acc_drop", n_boot, rng)
    beta, p_lrt, (lo, hi) = res["beta"], res["p_lrt"], res["ci"]

    # ---- secondary: per-signal Spearman (context only) ----
    print("\n  Secondary (exploratory) -- pooled Spearman vs acc_drop:")
    for c, lbl in [("cov", "covariate_shift (baseline)"),
                   ("pred", "prediction_shift (baseline)"),
                   ("expl", "explanation_shift (ours)")]:
        rho, p = spearmanr(df[c], df["acc_drop"])
        print(f"    {lbl:32s} rho={rho:+.3f}  p={p:.4f}")

    # ---- robustness: same incremental test on AUC- and PR-AUC-drop (secondary) ----
    print(f"\n{'-'*70}\n  ROBUSTNESS (secondary) -- does expl add value for AUC / PR-AUC drop?")
    print(f"{'-'*70}")
    robustness = {}
    for outc in ["auc_drop", "ap_drop"]:
        if outc in df.columns:
            robustness[outc] = incremental_test(df, outc, n_boot, rng)

    # ---- pre-registered GO / NO-GO (on the PRIMARY outcome only) ----
    ci_ok = np.isfinite(lo)
    go = (beta > 0) and (p_lrt < 0.05) and ci_ok and (lo > 0)
    print(f"\n{'='*70}")
    print("PRE-REGISTERED DECISION (primary outcome = acc_drop)")
    print(f"{'='*70}")
    print("  Rule: GO iff  beta_expl>0  AND  LRT p<0.05  AND  bootstrap CI lower>0.")
    print(f"  -> beta_expl>0: {beta>0}   LRT p<0.05: {p_lrt<0.05}   "
          f"CI_lo>0: {(lo>0) if ci_ok else 'N/A (bootstrap failed)'}")
    if not ci_ok:
        print("\n  ===> INCONCLUSIVE: bootstrap CI unavailable; do not call GO/NO-GO.")
    else:
        print(f"\n  ===> {'GO  (early warning beats baselines)' if go else 'NO-GO  (report the null; pivot to trade-off map)'}")
    return dict(n_pairs=n_pairs, n_domains=int(n_domains), n_states=int(n_states),
                primary=res, robustness=robustness, go=bool(go))


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--source-state", default="CA")
    ap.add_argument("--source-year", default=None)
    ap.add_argument("--states", default=None, help="comma list, overrides scale")
    ap.add_argument("--years", default=None, help="comma list, overrides scale")
    ap.add_argument("--tasks", default=None, help="comma list subset of income,pubcov,mobility")
    ap.add_argument("--source-cap", type=int, default=20000)
    ap.add_argument("--target-cap", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    cfg = ap.parse_args()

    if cfg.scale == "pilot":
        states = ["CA", "TX", "NY", "FL"]
        years = ["2017", "2018"]
        cfg.source_year = cfg.source_year or "2017"
        cfg.target_cap = cfg.target_cap or 6000
        cfg.bootstrap = cfg.bootstrap if cfg.bootstrap is not None else 50
    else:  # full
        states = ["CA", "TX", "NY", "FL", "PA", "OH", "IL", "GA", "NC", "WA", "AZ", "MA"]
        years = ["2014", "2015", "2016", "2017", "2018"]
        cfg.source_year = cfg.source_year or "2014"
        cfg.target_cap = cfg.target_cap or 8000
        cfg.bootstrap = cfg.bootstrap if cfg.bootstrap is not None else 300

    cfg.states = cfg.states.split(",") if cfg.states else states
    cfg.years = cfg.years.split(",") if cfg.years else years
    cfg.tasks = cfg.tasks.split(",") if cfg.tasks else None
    return cfg


def main():
    cfg = parse_args()
    if not (have("folktables") and have("shap") and have("statsmodels")):
        print("Needs folktables, shap, statsmodels. `pip install -r requirements.txt`.")
        return
    print(f"SCALE={cfg.scale}  source=({cfg.source_state},{cfg.source_year})  "
          f"states={cfg.states}  years={cfg.years}")

    df = run_grid(cfg)
    if df.empty:
        print("No rows produced (all domains skipped?).")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"week2_rows_{cfg.scale}.csv")
    df.to_csv(out, index=False)
    print(f"\n  per-shift-pair rows written: {out}")

    analyze(df, cfg.bootstrap, np.random.default_rng(1))


if __name__ == "__main__":
    main()
