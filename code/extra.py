"""Supplementary analyses: PITCH interval-effect bootstrap, simulated kinetics by
platform, and two illustrative patient trajectories under the learned policy."""
import json
import numpy as np
import pandas as pd
from core import logged_data, prior_data, fit_policy, POP
from sim import rollout, Cohort, DOSE
from baselines import Q3W
from ktree import NODES


def pitch_interval_effect(B=2000, seed=0):
    tr = pd.read_csv("results/pitch_train_pred.csv")
    te = pd.read_csv("results/pitch_test_pred.csv")
    rng = np.random.default_rng(seed)
    rows = []
    for assay, visit in [("IgG", "V2plus28"), ("T", "V2plus28")]:
        a = tr[(tr.assay == assay) & (tr.visit == visit)]
        b = te[(te.assay == assay) & (te.visit == visit)]
        obs = np.median(b.y) - np.median(a.y)
        boots = [np.median(rng.choice(b.y.values, len(b))) - np.median(rng.choice(a.y.values, len(a)))
                 for _ in range(B)]
        pred = np.median(b.yhat) - np.median(a.yhat)
        rows.append(dict(assay=assay, visit=visit, n_long=len(a), n_short=len(b),
                         obs_diff=obs, ci_lo=np.quantile(boots, 0.025),
                         ci_hi=np.quantile(boots, 0.975), pred_diff=pred))
    pd.DataFrame(rows).to_csv("results/pitch_interval_effect.csv", index=False)
    print(pd.DataFrame(rows).round(3))


def platform_kinetics():
    out = []
    for plat in ["circRNA", "mRNA"]:
        co, ret, traj = rollout(Q3W, 2000, 4242, POP, platform=plat, record=True)
        for tr in traj:
            o = tr["obs"]
            out.append(dict(platform=plat, week=int(o["t"][0]),
                            E_med=float(np.median(o["logE"])), E_lo=float(np.quantile(o["logE"], 0.25)),
                            E_hi=float(np.quantile(o["logE"], 0.75)),
                            H_med=float(np.median(o["logH"])), H_lo=float(np.quantile(o["logH"], 0.25)),
                            H_hi=float(np.quantile(o["logH"], 0.75)),
                            X_med=float(np.median(o["X"])), X_lo=float(np.quantile(o["X"], 0.25)),
                            X_hi=float(np.quantile(o["X"], 0.75))))
    pd.DataFrame(out).to_csv("results/platform_kinetics.csv", index=False)


def case_studies(seed=3, delta=0.10):
    tr = logged_data(seed, 400)
    pr = prior_data(seed)
    pol, info = fit_policy(tr, pr, delta=delta)
    pl, disc = info["plan"], info["disc"]
    st = pl.strides()
    co, ret, traj = rollout(pol, 300, 31337, POP, record=True)
    rows = []
    for k, step in enumerate(traj):
        o = step["obs"]
        z = disc.transform(o)
        idx = z @ st
        t = int(o["t"][0])
        Q = pl.Q[min(t, pl.T - 1)][idx]
        rows.append(pd.DataFrame(dict(pid=np.arange(co.n), week=t, logE=o["logE"], logH=o["logH"],
                                      X=o["X"], tox=o["tox"], maxtox=o["maxtox"], brisk=o["brisk"],
                                      action=step["a"], ucb_full=pl.tox_ucb_tab[idx, 1],
                                      mean_full=pl.tox_mean_tab[idx, 1],
                                      allowed_full=pl.allowed[idx, 1],
                                      Q_defer=Q[:, 0], Q_full=Q[:, 1], Q_red=Q[:, 2],
                                      true_risk_full=step["risk_full"])))
    df = pd.concat(rows)
    # select a low-risk responder and a high-risk patient with a masked boost
    final = df[df.week == df.week.max()]
    low = final[(final.brisk == 0) & (final.maxtox <= 1)].sort_values("X").pid.values
    masked = df[(df.allowed_full == False) & (df.brisk == 1)].pid.unique()
    high = [p for p in masked if p in final[final.brisk == 1].pid.values]
    pick = [int(low[len(low) // 4]), int(high[0]) if len(high) else int(final[final.brisk == 1].pid.values[0])]
    sel = df[df.pid.isin(pick)].copy()
    sel["case"] = np.where(sel.pid == pick[0], "Case 1", "Case 2")
    sel.to_csv("results/case_studies.csv", index=False)
    json.dump(dict(parents={NODES[i]: [NODES[j] for j in pa] for i, pa in info["model"].parents.items()},
                   ess={NODES[i]: e for i, e in info["model"].ess.items()},
                   cuts_C=list(map(float, disc.eC)), cuts_H=list(map(float, disc.eH)), picked=pick),
              open("results/case_model.json", "w"), indent=1)
    print(sel.groupby("case").agg(n_boost=("action", lambda a: int((a > 0).sum())),
                                  brisk=("brisk", "max"), maxtox=("maxtox", "max")))


if __name__ == "__main__":
    pitch_interval_effect()
    platform_kinetics()
    case_studies()
