"""
Generate the remaining manuscript figures (Fig 1, 2, 3, 5) into results/figures/.
Figs 2/3/5 come straight from the saved result tables; Fig 1 (reliability) retrains a
single frozen model (cheap) to get per-bin calibration curves. Figs 4 and 6 are produced
by notebooks 00 and 03.

Run: .venv/bin/python make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from week2_powered import RESULTS_DIR
FIG = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

early = pd.read_csv(os.path.join(RESULTS_DIR, "week2_rows_full.csv"))
audit = pd.read_csv(os.path.join(RESULTS_DIR, "week3_audit_full.csv"))
COL = {"logreg": "#4a90d9", "rf": "#5cb85c", "xgb": "#d9534f"}


# ---- Figure 1: reliability diagrams under shift (retrain one frozen model) ----
def fig1_reliability():
    from sklearn.preprocessing import StandardScaler
    from week2_powered import build_tasks, get_frozen_models
    from week3_audit import load_with_prot
    rng = np.random.default_rng(0)
    problem = build_tasks()["income"]
    Xs, ys, _, _ = load_with_prot(problem, "CA", "2014", 20000, rng)
    sc = StandardScaler().fit(Xs[: int(0.8 * len(Xs))]); Xs = sc.transform(Xs)
    cut = int(0.8 * len(Xs))
    model = get_frozen_models()["xgb"].fit(Xs[:cut], ys[:cut])
    Xv, yv = Xs[cut:], ys[cut:]
    Xt_raw, yt, _, _ = load_with_prot(problem, "TX", "2018", 8000, rng)
    Xt = sc.transform(Xt_raw)

    def curve(X, y, bins=10):
        p = model.predict_proba(X)[:, 1]
        edges = np.linspace(0, 1, bins + 1); idx = np.digitize(p, edges) - 1
        xs, ys_ = [], []
        for b in range(bins):
            m = idx == b
            if m.sum() > 20:
                xs.append(p[m].mean()); ys_.append(y[m].mean())
        return np.array(xs), np.array(ys_)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    xs, ys_ = curve(Xv, yv); ax.plot(xs, ys_, "o-", color="#4a90d9", label="in-distribution (CA-2014)")
    xt, yt_ = curve(Xt, yt); ax.plot(xt, yt_, "s-", color="#d9534f", label="shifted (TX-2018)")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Fig 1. Reliability degrades under shift (XGBoost, income)")
    ax.legend(frameon=False); plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_reliability_under_shift.png"), dpi=200, bbox_inches="tight")
    print("wrote fig1_reliability_under_shift.png")


# ---- Figure 2: SHAP importance stability vs shift magnitude, by model ----
def fig2_shap_stability():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m, g in early.groupby("model"):
        ax.scatter(g["cov"], g["expl"], s=14, alpha=0.5, color=COL[m], label=m)
    ax.set_xlabel("covariate-shift magnitude (domain-classifier AUC)")
    ax.set_ylabel("explanation shift (TV distance of SHAP profiles)")
    ax.set_title("Fig 2. SHAP attributions move with covariate shift, by model")
    ax.legend(frameon=False, title="model"); plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_shap_stability.png"), dpi=200, bbox_inches="tight")
    print("wrote fig2_shap_stability.png")


# ---- Figure 3: fairness-gap trajectories under shift ----
def fig3_fairness_trajectories():
    # join covariate-shift magnitude to the audit table
    d = audit.merge(early[["task", "model", "state", "year", "cov"]],
                    on=["task", "model", "state", "year"], how="left")
    d["mag_bin"] = pd.qcut(d["cov"], 5, labels=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for ax, (col, name) in zip(axes, [("dp_sex_tgt", "DP gap (sex)"), ("dp_rac_tgt", "DP gap (race)")]):
        for t, g in d.groupby("task"):
            traj = g.groupby("mag_bin")[col].mean()
            ax.plot(traj.index, traj.values, "o-", label=t)
        ax.set_xlabel("covariate-shift magnitude (quintile)")
        ax.set_title(name)
    axes[0].set_ylabel("fairness gap on shifted domain")
    axes[0].legend(frameon=False, title="task")
    fig.suptitle("Fig 3. Fairness gaps widen with shift magnitude", y=1.03, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_fairness_trajectories.png"), dpi=200, bbox_inches="tight")
    print("wrote fig3_fairness_trajectories.png")


# ---- Figure 5: incremental early-warning value (signals vs accuracy drop) ----
def fig5_early_warning():
    outcomes = ["acc_drop", "auc_drop", "ap_drop"]
    sigs = {"cov": "covariate (cheap)", "pred": "prediction (cheap)", "expl": "explanation (ours)"}
    M = np.array([[spearmanr(early[s], early[o]).statistic for o in outcomes] for s in sigs])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(outcomes)); w = 0.26
    for i, (k, name) in enumerate(sigs.items()):
        ax.bar(x + (i - 1) * w, M[i], w, label=name, color=["#999999", "#4a90d9", "#d9534f"][i])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["accuracy drop", "ROC-AUC drop", "PR-AUC drop"])
    ax.set_ylabel("Spearman correlation\n(positive = useful warning)")
    ax.set_title("Fig 5. Explanation shift does not beat the cheap baselines")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_incremental_value.png"), dpi=200, bbox_inches="tight")
    print("wrote fig5_incremental_value.png")


if __name__ == "__main__":
    fig2_shap_stability()
    fig3_fairness_trajectories()
    fig5_early_warning()
    fig1_reliability()   # last: needs data load + model fit
    print("all figures ->", FIG)
