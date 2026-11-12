"""All manuscript figures (consistent typography, line widths and dark academic colours)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ----------------------------------------------------------------- style
NAVY, CRIMSON, BLACK, FOREST, PLUM, GREY = "#1f3a93", "#a31515", "#111111", "#1e6b3a", "#5b2a86", "#6b6b6b"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
    "lines.linewidth": 1.6, "lines.markersize": 5, "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "legend.frameon": False,
})
W1, W2 = 3.5, 7.2          # single / double column widths (inches)
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
POL_ORDER = ["Fixed extended", "Fixed prime-boost", "Clinician rule", "Fixed q3w",
             "FQI (model-free)", "k-tree, delta=0.05", "k-tree (proposed)"]
POL_LABEL = {"Fixed extended": "Fixed extended (0,8,16)", "Fixed prime-boost": "Fixed prime-boost (0,4,8,16)",
             "Clinician rule": "Clinician-style rule", "Fixed q3w": "Fixed q3w (8 doses)",
             "FQI (model-free)": "Fitted Q-iteration", "k-tree, delta=0.05": r"k-tree, $\delta=0.05$",
             "k-tree (proposed)": r"k-tree, $\delta=0.10$ (proposed)"}
POL_COL = {"Fixed extended": GREY, "Fixed prime-boost": GREY, "Clinician rule": PLUM, "Fixed q3w": BLACK,
           "FQI (model-free)": FOREST, "k-tree, delta=0.05": NAVY, "k-tree (proposed)": CRIMSON}
POL_MK = {"Fixed extended": "v", "Fixed prime-boost": "^", "Clinician rule": "D", "Fixed q3w": "s",
          "FQI (model-free)": "P", "k-tree, delta=0.05": "o", "k-tree (proposed)": "o"}


def ci(x):
    x = np.asarray(x, float)
    h = 2.093 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), h


def panel(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


# --------------------------------------------------------- Fig 1: PITCH validation
def fig_pitch():
    tr = pd.read_csv("results/pitch_train_pred.csv")
    te = pd.read_csv("results/pitch_test_pred.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.8))
    visits = [("V1plus28", "D1+28"), ("preV2", "pre-D2"), ("V2plus28", "D2+28"), ("V2plus90", "D2+90")]
    for ax, assay, lab in [(axes[0], "IgG", r"Anti-spike IgG, $\log_{10}$ AU"),
                           (axes[1], "T", r"ELISpot, $\log_{10}$(SFU+10)")]:
        for k, (df, name, col, off) in enumerate([(tr, "Long interval (fitting set)", NAVY, -0.12),
                                                  (te, "Short interval (held out)", CRIMSON, 0.12)]):
            xs, med, lo, hi, pm = [], [], [], [], []
            for i, (v, _) in enumerate(visits):
                g = df[(df.assay == assay) & (df.visit == v)]
                if len(g) < 5:
                    continue
                xs.append(i + off); med.append(g.y.median()); lo.append(g.y.quantile(.25))
                hi.append(g.y.quantile(.75)); pm.append(g.yhat.median())
            xs = np.array(xs)
            ax.errorbar(xs, med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt="o", color=col,
                        capsize=2.5, lw=1.2, label=f"{name}: observed median (IQR)")
            ax.plot(xs, pm, ls="none", marker="x", ms=7, mew=1.6, color=col,
                    label=f"{name}: model prediction")
        ax.set_xticks(range(len(visits)))
        ax.set_xticklabels([v[1] for v in visits])
        ax.set_ylabel(lab)
        ax.set_xlabel("Sampling time point")
    panel(axes[0], "(a)"); panel(axes[1], "(b)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig2_PITCH_validation.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 2: platform kinetics
def fig_platform_kinetics():
    d = pd.read_csv("results/platform_kinetics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4))
    for ax, key, lab in [(axes[0], "E", r"ELISpot, $\log_{10}$(SFU/$10^6$+10)"),
                         (axes[1], "H", r"IgG, $\log_{10}$ AU"),
                         (axes[2], "X", "Log tumour burden change")]:
        for plat, col, ls in [("circRNA", CRIMSON, "-"), ("mRNA", NAVY, "--")]:
            g = d[d.platform == plat]
            ax.plot(g.week, g[f"{key}_med"], color=col, ls=ls, label=plat)
            ax.fill_between(g.week, g[f"{key}_lo"], g[f"{key}_hi"], color=col, alpha=0.12, lw=0)
        for w in range(0, 24, 3):
            ax.axvline(w, color=GREY, lw=0.4, ls=":")
        ax.set_xlabel("Week"); ax.set_ylabel(lab)
    axes[0].legend(loc="lower right")
    for a, s in zip(axes, "abc"):
        panel(a, f"({s})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig3_Platform_kinetics.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 3: main comparison
def fig_main():
    d = pd.read_csv("results/main_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    for i, p in enumerate(POL_ORDER):
        m, h = ci(d[d.policy == p].V)
        ax.errorbar(m, i, xerr=h, fmt=POL_MK[p], color=POL_COL[p], capsize=2.5, ms=6)
    ax.set_yticks(range(len(POL_ORDER)))
    ax.set_yticklabels([POL_LABEL[p] for p in POL_ORDER])
    ax.set_xlabel("Clinical value $V$ (mean, 95% CI over 20 runs)")
    ax.grid(axis="y", visible=False)
    panel(ax, "(a)")
    ax = axes[1]
    for p in POL_ORDER:
        g = d[d.policy == p]
        mx, hx = ci(g.any_g3 * 100); my, hy = ci(g.V)
        ax.errorbar(mx, my, xerr=hx, yerr=hy, fmt=POL_MK[p], color=POL_COL[p], capsize=2, ms=6,
                    label=POL_LABEL[p])
    ax.set_xlabel(r"Patients with any grade $\geq$3 event (%)")
    ax.set_ylabel("Clinical value $V$")
    panel(ax, "(b)")
    for p in POL_ORDER:
        g = d[d.policy == p]
        ax.annotate(POL_LABEL[p].split(" (")[0].replace("Fitted Q-iteration", "FQI"),
                    (g.any_g3.mean() * 100, g.V.mean()), xytext=(4, -9), textcoords="offset points",
                    fontsize=6.5, color=BLACK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig4_Policy_comparison.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 4: frontier
def fig_frontier():
    d = pd.read_csv("results/frontier.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    series = [("k-tree (ucb)", "k-tree, credible bound", CRIMSON, "o", "-"),
              ("k-tree (mean)", "k-tree, point estimate", NAVY, "s", "--"),
              ("FQI", "Fitted Q-iteration (no constraint)", FOREST, "P", ":")]
    deltas = sorted(d.delta_set.unique())
    xt = np.arange(len(deltas))
    for name, lab, col, mk, ls in series:
        g = d[d.policy == name]
        mV = [ci(g[g.delta_set == x].V) for x in deltas]
        mR = [ci(100 * g[g.delta_set == x].viol_rate) for x in deltas]
        axes[0].errorbar(xt, [a for a, _ in mV], yerr=[b for _, b in mV], color=col, marker=mk, ls=ls,
                         capsize=2, label=lab)
        axes[1].errorbar(xt, [a for a, _ in mR], yerr=[b for _, b in mR], color=col, marker=mk, ls=ls,
                         capsize=2, label=lab)
    for ax in axes:
        ax.set_xticks(xt)
        ax.set_xticklabels(["1.0" if x == 1 else f"{x:g}" for x in deltas])
        ax.set_xlabel(r"Toxicity tolerance $\delta$")
    axes[0].set_ylabel("Clinical value $V$")
    axes[1].set_ylabel(r"Boosts with true risk $>\delta$ (%)")
    axes[1].legend(loc="upper right")
    panel(axes[0], "(a)"); panel(axes[1], "(b)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig5_Safety_frontier.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 5: sensitivity + data size
def fig_sensitivity():
    s = pd.read_csv("results/sensitivity.csv")
    ds = pd.read_csv("results/data_size.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))
    ax = axes[0]
    piv = s.pivot_table(index="L", columns="k", values="V", aggfunc="mean")
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_xlabel("In-degree bound $k$"); ax.set_ylabel("Readout bins $L$")
    ax.grid(False)
    vmax = piv.values.max(); vmin = piv.values.min()
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > vmin + 0.6 * (vmax - vmin) else BLACK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Mean clinical value $V$")
    panel(ax, "(a)")
    ax = axes[1]
    for name, lab, col, mk, ls in [("k-tree (proposed)", "k-tree with mechanistic prior", CRIMSON, "o", "-"),
                                   ("k-tree, no prior", "k-tree without prior", NAVY, "s", "--"),
                                   ("FQI (model-free)", "Fitted Q-iteration", FOREST, "P", ":")]:
        g = ds[ds.policy == name]
        ns = sorted(g.n_train.unique())
        m = [ci(g[g.n_train == n].V) for n in ns]
        ax.errorbar(ns, [a for a, _ in m], yerr=[b for _, b in m], color=col, marker=mk, ls=ls,
                    capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.minorticks_off()
    ax.set_xlabel("Training patients"); ax.set_ylabel("Clinical value $V$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7)
    panel(ax, "(b)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig7_Sensitivity_and_data_size.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 6: learned structure
def fig_structure():
    S = json.load(open("results/structures.json"))
    nodes = ["C", "H", "U", "Tox", "R", "B", "S", "N"]
    lab = {"C": "T-cell C", "H": "Antibody H", "U": "Tumour U", "Tox": "Toxicity Tox",
           "R": "Reactogenicity R", "B": "Baseline risk B", "S": "Weeks since dose S", "N": "Doses N"}
    learned = ["C", "H", "U", "Tox", "R"]
    M = np.zeros((len(learned), len(nodes)))
    for s_ in S:
        for i, child in enumerate(learned):
            for p in s_["parents"][child]:
                M[i, nodes.index(p)] += 1
    M /= len(S)
    fig, ax = plt.subplots(figsize=(W2 * 0.78, 2.6))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if learned[i] == nodes[j]:
                ax.text(j, i, "self", ha="center", va="center", fontsize=7, color=GREY)
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 else BLACK)
    ax.set_xticks(range(len(nodes))); ax.set_xticklabels([lab[n] + r"$_t$" for n in nodes], rotation=35, ha="right")
    ax.set_yticks(range(len(learned))); ax.set_yticklabels([lab[n] + r"$_{t+1}$" for n in learned])
    ax.set_xlabel("Candidate parent (current slice)"); ax.set_ylabel("Child (next slice)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Selection frequency (20 runs)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig8_Learned_structure.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 7: calibration
def fig_calibration():
    d = pd.read_csv("results/calibration.csv")
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    for col_name, lab, col, mk in [("p_mean", "Posterior mean", NAVY, "s"), ("p_ucb", "90% upper credible bound", CRIMSON, "o")]:
        q = pd.qcut(d[col_name].rank(method="first"), 8, labels=False)
        g = d.groupby(q).agg(p=(col_name, "mean"), y=("y", "mean"), n=("y", "size"))
        se = np.sqrt(g.y * (1 - g.y) / g.n)
        ax.errorbar(g.p * 100, g.y * 100, yerr=1.96 * se * 100, color=col, marker=mk, capsize=2, label=lab)
    lim = 16
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel(r"Predicted P(grade $\geq$3 | boost) (%)")
    ax.set_ylabel("Observed frequency (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/Fig6_Toxicity_calibration.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 8: case studies
def fig_cases(delta=0.10):
    d = pd.read_csv("results/case_studies.csv")
    fig, axes = plt.subplots(3, 2, figsize=(W2, 5.2), sharex=True)
    for j, (case, g) in enumerate(d.groupby("case")):
        g = g.sort_values("week")
        ax = axes[0, j]
        ax.plot(g.week, g.logE, color=CRIMSON, marker="o", ms=3)
        ax.set_ylabel(r"ELISpot ($\log_{10}$)")
        ax.set_title(f"{case}: baseline risk {'high' if g.brisk.max() else 'low'}")
        ax = axes[1, j]
        ax.plot(g.week, g.X, color=BLACK, marker="o", ms=3)
        ax.axhline(np.log(0.7), color=GREY, ls=":", lw=1)
        ax.set_ylabel("Log tumour change")
        ax = axes[2, j]
        ax.step(g.week, 100 * g.ucb_full, where="mid", color=CRIMSON, label="Credible bound, full boost")
        ax.plot(g.week, 100 * g.true_risk_full, color=GREY, ls="--", lw=1, label="True risk (simulator)")
        ax.axhline(100 * delta, color=BLACK, ls=":", lw=1, label=r"Tolerance $\delta$")
        for _, r in g.iterrows():
            if r.action == 1:
                ax.scatter(r.week, 1, marker="^", color=FOREST, s=30, zorder=3)
            elif r.action == 2:
                ax.scatter(r.week, 1, marker="v", color=NAVY, s=30, zorder=3)
        ax.scatter([], [], marker="^", color=FOREST, label="Full boost given")
        ax.scatter([], [], marker="v", color=NAVY, label="Reduced boost given")
        ax.set_ylabel(r"P(grade $\geq$3) (%)")
        ax.set_xlabel("Week")
    for a, s in zip(axes.T.ravel(), "abcdef"):
        panel(a, f"({s})")
    h, l = axes[2, 1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(f"{OUT}/Fig9_Case_studies.png")
    plt.close(fig)


# --------------------------------------------------------- Fig 9: platform policy
def fig_platform_policy():
    d = pd.read_csv("results/platform.csv")
    pols = ["Fixed extended", "Fixed prime-boost", "Fixed q3w", "k-tree (proposed)"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    x = np.arange(len(pols))
    for k, (plat, col, hatch) in enumerate([("circRNA", CRIMSON, None), ("mRNA", NAVY, "//")]):
        for ax, met, lab in [(axes[0], "V", "Clinical value $V$"), (axes[1], "resp30", r"Patients with $\geq$30% shrinkage (%)")]:
            vals = [ci(d[(d.platform == plat) & (d.policy == p)][met] * (100 if met == "resp30" else 1)) for p in pols]
            ax.bar(x + (k - 0.5) * 0.36, [a for a, _ in vals], 0.34, yerr=[b for _, b in vals],
                   color=col, alpha=0.85, hatch=hatch, edgecolor="white", capsize=2, label=plat)
            ax.set_ylabel(lab)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["Extended", "Prime-boost", "q3w", "k-tree"])
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    panel(axes[0], "(a)"); panel(axes[1], "(b)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/Extra_Platform_policies.png")  # not used in the manuscript
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["pitch", "platform_kinetics", "main", "frontier", "sensitivity",
                             "structure", "calibration", "cases"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("done", w)
