"""Run every in-silico experiment reported in the manuscript.
Outputs CSV files in results/. Usage: python code/run_all.py [n_seeds]"""
import sys
import time
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from core import (logged_data, prior_data, fit_policy, evaluate, metrics, POP, E_THR,
                  PRIOR_SPEC)
from baselines import Q3W, PRIME_BOOST, EXTENDED, clinician_rule
from fqi import fit_fqi
from ktree import NODES

N_TRAIN = 400
N_TEST = 2000
DELTA = 0.10
SEEDS = list(range(1, int(sys.argv[1]) + 1)) if len(sys.argv) > 1 else list(range(1, 21))


def test_seed(s):
    return 10_000 + s


def ev(pol, s, name, delta=DELTA, platform="circRNA", keep_ret=False, **extra):
    m, r, co, traj = evaluate(pol, test_seed(s), n=N_TEST, platform=platform, delta=delta)
    m.update(policy=name, seed=s, platform=platform, **extra)
    return (m, r) if keep_ret else m


# --------------------------------------------------------------- experiment 2
def main_comparison(s):
    tr = logged_data(s, N_TRAIN)
    pr = prior_data(s)
    rows, rets = [], {}
    pols = [("Fixed q3w", Q3W), ("Fixed prime-boost", PRIME_BOOST),
            ("Fixed extended", EXTENDED), ("Clinician rule", clinician_rule(E_THR))]
    t0 = time.time()
    pols.append(("FQI (model-free)", fit_fqi(tr, seed=s)))
    t_fqi = time.time() - t0
    t0 = time.time()
    kt, info = fit_policy(tr, pr, delta=DELTA)
    t_kt = time.time() - t0
    kt05, _ = fit_policy(tr, pr, delta=0.05)
    pols += [("k-tree (proposed)", kt), ("k-tree, delta=0.05", kt05)]
    for name, p in pols:
        m, r = ev(p, s, name, keep_ret=True)
        rows.append(m)
        rets[name] = r
    ref = rets["k-tree (proposed)"]
    paired = [dict(seed=s, policy=k, dV_mean=float((ref - v).mean()))
              for k, v in rets.items() if k != "k-tree (proposed)"]
    par = {NODES[i]: [NODES[j] for j in pa] for i, pa in info["model"].parents.items()}
    return rows, paired, dict(seed=s, parents=par, t_fit_ktree=t_kt, t_fit_fqi=t_fqi)


# --------------------------------------------------------------- experiment 3
def frontier(s):
    tr = logged_data(s, N_TRAIN)
    pr = prior_data(s)
    rows = []
    for d in [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 1.0]:
        for cons in ["ucb", "mean"]:
            p, _ = fit_policy(tr, pr, delta=d, constraint=cons)
            rows.append(ev(p, s, f"k-tree ({cons})", delta=d, delta_set=d, constraint=cons))
    fq = fit_fqi(tr, seed=s)
    for d in [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 1.0]:
        rows.append(ev(fq, s, "FQI", delta=d, delta_set=d, constraint="none"))
    return rows


# --------------------------------------------------------------- experiment 4
def ablation(s):
    tr = logged_data(s, N_TRAIN)
    pr = prior_data(s)
    cfg = [("Full model", dict()),
           ("No mechanistic prior", dict(prior=False)),
           ("k = 0 (no cross-links)", dict(k=0)),
           ("k = 1", dict(k=1)),
           ("k = 3", dict(k=3)),
           ("Saturated (k = m-1)", dict(k=7)),
           ("Point-estimate constraint", dict(constraint="mean")),
           ("No safety constraint", dict(delta=1.0))]
    rows = []
    for name, c in cfg:
        c = dict(c)
        use_prior = c.pop("prior", True)
        t0 = time.time()
        p, info = fit_policy(tr, pr if use_prior else None, **c)
        rows.append(ev(p, s, name, fit_sec=time.time() - t0,
                       n_params=int(sum(v["counts"].size for v in info["model"].cpt.values()))))
    return rows


# --------------------------------------------------------------- experiment 5
def sensitivity(s):
    tr = logged_data(s, N_TRAIN)
    pr = prior_data(s)
    rows = []
    for L in [3, 4, 5, 6, 7]:
        for k in [0, 1, 2, 3]:
            t0 = time.time()
            p, info = fit_policy(tr, pr, L=L, k=k)
            rows.append(ev(p, s, "k-tree", L=L, k=k, fit_sec=time.time() - t0))
    return rows


def data_size(s):
    rows = []
    for n in [100, 200, 400, 800, 1600]:
        tr = logged_data(s, n)
        pr = prior_data(s)
        p, _ = fit_policy(tr, pr)
        rows.append(ev(p, s, "k-tree (proposed)", n_train=n))
        p, _ = fit_policy(tr, None)
        rows.append(ev(p, s, "k-tree, no prior", n_train=n))
        rows.append(ev(fit_fqi(tr, seed=s), s, "FQI (model-free)", n_train=n))
    return rows


# --------------------------------------------------------------- experiment 6
def platform(s):
    rows = []
    for plat in ["circRNA", "mRNA"]:
        tr = logged_data(s, N_TRAIN, platform=plat)
        pr = prior_data(s, platform=plat)
        p, _ = fit_policy(tr, pr)
        rows.append(ev(p, s, "k-tree (proposed)", platform=plat))
        for name, q in [("Fixed q3w", Q3W), ("Fixed prime-boost", PRIME_BOOST),
                        ("Fixed extended", EXTENDED)]:
            rows.append(ev(q, s, name, platform=plat))
    return rows


# --------------------------------------------------------------- experiment 7
def calibration(s):
    """Predicted vs observed severe-toxicity probability after boosts (test cohort,
    logged with the behaviour policy so that all actions occur)."""
    from baselines import behaviour_policy
    from sim import rollout
    from ktree import collect
    tr = logged_data(s, N_TRAIN)
    pr = prior_data(s)
    _, info = fit_policy(tr, pr)
    te = logged_data(test_seed(s), 1000)
    D = collect(te, info["disc"])
    ucb, mean = info["model"].tox_ucb(D["Z"], 0.90)
    b = D["A"] > 0
    pm = mean[np.arange(len(D["A"])), D["A"]][b]
    pu = ucb[np.arange(len(D["A"])), D["A"]][b]
    y = (D["Z2"][b, 3] == 1).astype(int)
    return [dict(seed=s, p_mean=float(a), p_ucb=float(u), y=int(v)) for a, u, v in zip(pm, pu, y)]


# --------------------------------------------------------------- experiment 9
def prior_robustness(s):
    rows = []
    variants = {"Nominal prior": dict(),
                "Immunogenicity x0.5": dict(imm_scale=0.10),
                "Immunogenicity x2": dict(imm_scale=0.40),
                "Tumour kill x0.5": dict(kill_scale=0.4),
                "Tumour kill x1.5": dict(kill_scale=1.2),
                "mRNA kinetics as prior": dict(kR_factor=0.35 / 0.12)}
    for n in [200, 400]:
        tr = logged_data(s, n)
        for name, v in variants.items():
            spec = dict(PRIOR_SPEC)
            spec.update(v)
            pr = prior_data(s, spec=spec)
            p, _ = fit_policy(tr, pr)
            rows.append(ev(p, s, name, n_train=n))
        p, _ = fit_policy(tr, None)
        rows.append(ev(p, s, "No prior", n_train=n))
    return rows


def runtime(s):
    """Planning cost of factored (variable elimination) vs saturated model."""
    tr = logged_data(s, N_TRAIN)
    pr = prior_data(s)
    rows = []
    for k in [0, 1, 2, 3, 7]:
        t0 = time.time()
        _, info = fit_policy(tr, pr, k=k)
        t_total = time.time() - t0
        pl = info["plan"]
        t0 = time.time()
        pl.solve()
        rows.append(dict(seed=s, k=k, plan_sec=time.time() - t0, fit_total_sec=t_total,
                         n_params=int(sum(v["counts"].size for v in info["model"].cpt.values()))))
    return rows


def flat(list_of_lists):
    return [x for l in list_of_lists for x in l]


if __name__ == "__main__":
    t_all = time.time()
    with Pool(2) as pool:
        res = pool.map(main_comparison, SEEDS)
        pd.DataFrame(flat(r[0] for r in res)).to_csv("results/main_comparison.csv", index=False)
        pd.DataFrame(flat(r[1] for r in res)).to_csv("results/main_paired.csv", index=False)
        json.dump([r[2] for r in res], open("results/structures.json", "w"), indent=1)
        print("main", time.time() - t_all, flush=True)
        for fn, out in [(frontier, "frontier"), (ablation, "ablation"),
                        (sensitivity, "sensitivity"), (data_size, "data_size"),
                        (platform, "platform"), (calibration, "calibration"),
                        (prior_robustness, "prior_robustness"), (runtime, "runtime")]:
            r = pool.map(fn, SEEDS)
            pd.DataFrame(flat(r)).to_csv(f"results/{out}.csv", index=False)
            print(out, time.time() - t_all, flush=True)
