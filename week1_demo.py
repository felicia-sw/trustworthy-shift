"""
Week-1 go/no-go demo for:
  "Trustworthiness Auditing of Tabular Classifiers Under Distribution Shift"

Two parts, one file:

  PART A  Synthetic ground-truth check.
          We WRITE the data-generating rule, so we know the truth:
            - covariate shift  -> inputs move, rule unchanged -> explanations SHOULD stay
            - concept  shift   -> rule moves                  -> explanations SHOULD change
          This proves the explanation-shift metric measures what we claim, and shows
          the hard case (concept shift can crater accuracy with little observable signal).

  PART B  Real-data decisive experiment (Folktables ACSIncome, CA -> other states).
          The make-or-break question from the reviewer panel:
            does explanation shift predict the accuracy drop BETTER than the cheap,
            label-free baselines (covariate shift, prediction shift)?
          GO if yes. NO-GO (still publishable as an honest null) if no.

Variables are spelled out on purpose:
  - synthetic: age, income_k, education_yrs, debt_ratio, employment_yrs
               (each with a known effect, so the "true reasoning" is explicit)
  - census:    a small subset only (AGEP age, SCHL education, WKHP hours/week,
               COW worker class, MAR marital status), not all columns, to keep
               it light.

Run:
    pip install -r requirements.txt
    python week1_demo.py

Nothing here needs a GPU, credentials, or ethics approval. Folktables downloads
public census CSVs the first time (a few MB per state).
"""

import importlib
import numpy as np
from scipy.stats import spearmanr, ks_2samp
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Variables for the synthetic "credit scoring" world (Part A) -- named on purpose,
# each on a real-world scale with a known effect on the label.
SYNTH_VARS = ["age", "income_k", "education_yrs", "debt_ratio", "employment_yrs"]
SYNTH_MEAN = np.array([40., 50., 14., 0.35, 8.])    # natural-scale means
SYNTH_STD  = np.array([10., 15.,  3., 0.15, 5.])    # natural-scale spreads

# Restricted census feature set (Part B) -- a small, clear subset, NOT all columns,
# so it stays light and interpretable.
CENSUS_FEATURES = ["AGEP", "SCHL", "WKHP", "COW", "MAR"]  # age, education, hours/wk, worker class, marital status


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def have(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def get_models():
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=8,
                                     random_state=0, n_jobs=-1),
    }
    if have("xgboost"):
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                      subsample=0.8, eval_metric="logloss", random_state=0)
    else:
        print("  (xgboost not installed -> skipping XGB)")
    return models


def shap_importance(model, X_bg, X_ex):
    """Global feature importance = mean |SHAP|, computed against a FIXED background.
    The fixed background is the reviewer-requested control so we don't mistake a
    change in the reference data for a change in the model's reasoning.
    Written to be robust to shap version / model-type quirks."""
    import shap
    if type(model).__name__ == "LogisticRegression":
        sv = shap.LinearExplainer(model, X_bg).shap_values(X_ex)
    else:
        sv = shap.TreeExplainer(model, data=X_bg,
                                feature_perturbation="interventional").shap_values(
                                    X_ex, check_additivity=False)
    if isinstance(sv, list):        # some versions return [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                # (n, features, classes)
        sv = sv[..., -1]
    return np.abs(sv).mean(axis=0)


def explanation_shift(imp_a, imp_b):
    """Total-variation distance between the two normalised importance profiles.
    Magnitude-sensitive (unlike rank correlation, which is too coarse with few
    features): 0 = identical importance profile, 1 = completely different."""
    a = imp_a / (imp_a.sum() + 1e-12)
    b = imp_b / (imp_b.sum() + 1e-12)
    return 0.5 * np.abs(a - b).sum()


def covariate_shift(Xs, Xt):
    """Label-free covariate drift = domain-classifier AUC, rescaled.
    0 = indistinguishable, 1 = perfectly separable."""
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    idx = rng.permutation(len(X)); cut = len(X) // 2
    clf = LogisticRegression(max_iter=1000).fit(X[idx[:cut]], y[idx[:cut]])
    p = clf.predict_proba(X[idx[cut:]])[:, 1]
    return abs(roc_auc_score(y[idx[cut:]], p) - 0.5) * 2


def prediction_shift(model, Xs, Xt):
    """Label-free prediction drift = KS distance between predicted-prob distributions."""
    ps = model.predict_proba(Xs)[:, 1]
    pt = model.predict_proba(Xt)[:, 1]
    return ks_2samp(ps, pt)[0]


# --------------------------------------------------------------------------- #
# PART A  -  synthetic ground-truth check
# --------------------------------------------------------------------------- #
def part_A_synthetic():
    print("\n" + "=" * 70)
    print("PART A: synthetic ground-truth check")
    print("=" * 70)
    if not have("shap"):
        print("  shap not installed -> `pip install shap`. Skipping Part A.")
        return

    # clearly defined variables: each named feature has a known effect on the label,
    # so the "true reasoning" is explicit (income & education help, debt hurts, ...).
    names = SYNTH_VARS
    w_src = np.array([0.2, 1.5, 0.8, -1.2, 0.5])    # source rule
    w_con = np.array([0.1, 0.3, 0.4, -2.2, 0.3])    # concept shift: a regime where debt dominates

    def make(n, mean, w):
        raw = rng.normal(mean, SYNTH_STD, size=(n, 5))
        z = (raw - SYNTH_MEAN) / SYNTH_STD          # standardise vs the FIXED source reference
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(z @ w))))
        return z.astype(np.float32), y

    mean_richer = SYNTH_MEAN + np.array([10., 20., 0., 0., 0.])  # older, higher-income applicants

    Xtr, ytr = make(6000, SYNTH_MEAN, w_src)        # training distribution
    Xid, yid = make(3000, SYNTH_MEAN, w_src)        # in-distribution test
    Xcov, ycov = make(3000, mean_richer, w_src)     # covariate shift: population moves, rule unchanged
    Xcon, ycon = make(3000, SYNTH_MEAN, w_con)      # concept shift:   rule changes

    for name, model in get_models().items():
        model.fit(Xtr, ytr)
        bg = Xtr[:200]
        imp_src = shap_importance(model, bg, Xid[:300])
        acc_id = accuracy_score(yid, model.predict(Xid))
        top = ", ".join(names[i] for i in np.argsort(imp_src)[::-1])
        print(f"\n  [{name}]  in-distribution acc = {acc_id:.3f}")
        print(f"  features by importance: {top}")
        print(f"  {'shift':10s}{'acc_drop':>10s}{'expl_shift(obs)':>17s}{'expl_change(true)':>19s}")
        for tag, Xt, yt in [("covariate", Xcov, ycov), ("concept", Xcon, ycon)]:
            acc_drop = acc_id - accuracy_score(yt, model.predict(Xt))
            # what monitoring can see WITHOUT labels (fixed model, source vs target inputs):
            expl_obs = explanation_shift(imp_src, shap_importance(model, bg, Xt[:300]))
            # ground truth: did the RIGHT explanation actually change? (oracle retrained on target)
            oracle = clone(model).fit(Xt, yt)
            expl_true = explanation_shift(imp_src, shap_importance(oracle, Xt[:200], Xt[:300]))
            print(f"  {tag:10s}{acc_drop:>+10.3f}{expl_obs:>17.3f}{expl_true:>19.3f}")

    print("\n  Read it like this:")
    print("   - covariate: expl_change(true) ~ low (rule unchanged); any expl_shift(obs) is SPURIOUS.")
    print("   - concept:   expl_change(true) HIGH (rule changed) and acc_drop HIGH, but a fixed model's")
    print("                expl_shift(obs) can stay low -> the dangerous blind spot.")
    print("  This is exactly why Part B benchmarks the label-free signal against cheap baselines.")


# --------------------------------------------------------------------------- #
# PART B  -  real-data go / no-go
# --------------------------------------------------------------------------- #
def part_B_folktables():
    print("\n" + "=" * 70)
    print("PART B: real-data go / no-go  (Folktables ACSIncome)")
    print("=" * 70)
    if not (have("folktables") and have("shap")):
        print("  needs `pip install folktables shap`. Skipping Part B.")
        return
    from folktables import ACSDataSource, BasicProblem, adult_filter

    # Restricted income task: only the 5 clear features in CENSUS_FEATURES,
    # not all ~10 default columns -> lighter to run, easier to interpret.
    income_small = BasicProblem(
        features=CENSUS_FEATURES,
        target="PINCP",
        target_transform=lambda x: x > 50000,
        group="RAC1P",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    source_state = "CA"
    target_states = ["TX", "NY", "FL", "PA", "OH", "MI", "GA", "NC"]
    print(f"  features used ({len(CENSUS_FEATURES)}): {', '.join(CENSUS_FEATURES)}\n")

    def load(state, cap):
        df = ds.get_data(states=[state], download=True)
        X, y, _ = income_small.df_to_numpy(df)
        X, y = X.astype(np.float32), y.astype(int)
        if len(X) > cap:
            s = rng.choice(len(X), cap, replace=False)
            X, y = X[s], y[s]
        return X, y

    Xs, ys = load(source_state, cap=20000)
    sc = StandardScaler().fit(Xs)
    Xs = sc.transform(Xs)
    cut = int(0.8 * len(Xs))
    Xtr, ytr, Xval, yval = Xs[:cut], ys[:cut], Xs[cut:], ys[cut:]

    model = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=0, n_jobs=-1).fit(Xtr, ytr)
    bg = Xtr[:200]
    imp_src = shap_importance(model, bg, Xval[:300])
    acc_src = accuracy_score(yval, model.predict(Xval))
    print(f"  trained RF on {source_state}: in-distribution acc = {acc_src:.3f}\n")

    print(f"  {'state':6s}{'acc_drop':>10s}{'cov_shift':>11s}{'pred_shift':>12s}{'expl_shift':>12s}")
    rows = []
    for st in target_states:
        Xt, yt = load(st, cap=8000)
        Xt = sc.transform(Xt)
        acc_drop = acc_src - accuracy_score(yt, model.predict(Xt))
        cov = covariate_shift(Xtr[:4000], Xt[:4000])
        pred = prediction_shift(model, Xval, Xt)
        expl = explanation_shift(imp_src, shap_importance(model, bg, Xt[:300]))
        rows.append((acc_drop, cov, pred, expl))
        print(f"  {st:6s}{acc_drop:>+10.3f}{cov:>11.3f}{pred:>12.3f}{expl:>12.3f}")

    rows = np.array(rows)
    print("\n  --- GO / NO-GO: which signal best predicts the accuracy drop? ---")
    print("      (Spearman correlation of each label-free signal vs the realized acc_drop)")
    for j, label in [(1, "covariate_shift  (cheap baseline)"),
                     (2, "prediction_shift (cheap baseline)"),
                     (3, "explanation_shift (ours)")]:
        rho = spearmanr(rows[:, 0], rows[:, j])[0]
        rho = 0.0 if np.isnan(rho) else rho
        print(f"        {label:36s} rho = {rho:+.3f}")
    print("\n  GO    if explanation_shift clearly beats both baselines -> Q1-reachable claim.")
    print("  NO-GO if it does not -> pivot to the trade-off map and report the null honestly.")
    print("  Either way you know in week 1, not month 3.")


if __name__ == "__main__":
    part_A_synthetic()
    part_B_folktables()
