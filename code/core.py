"""Shared pipeline: data generation, model fitting, policy evaluation, metrics."""
import numpy as np
from sim import Cohort, rollout, load_pop, W, DOSE, PLATFORM_KR, T_WEEKS
from baselines import Q3W, PRIME_BOOST, EXTENDED, clinician_rule, behaviour_policy
from ktree import Discretizer, KTreeModel, Planner, collect

POP = load_pop()
E_THR = 2.3      # clinician-rule threshold on log10(ELISpot+10)

PRIOR_SPEC = dict(kR_factor=1.25, imm_scale=0.20, kill_scale=0.8, het=0.5,
                  simple_tox=True)


def logged_data(seed, n, platform="circRNA", eps=0.15, **ckw):
    rng = np.random.default_rng(seed + 7777)
    pol = behaviour_policy(rng, E_THR, eps=eps)
    co, ret, traj = rollout(pol, n, seed, POP, platform=platform, record=True, **ckw)
    return traj


def prior_data(seed, platform="circRNA", n=1500, spec=None):
    spec = dict(PRIOR_SPEC if spec is None else spec)
    kR = PLATFORM_KR[platform] * spec.pop("kR_factor")
    rng = np.random.default_rng(seed + 99)
    pol = behaviour_policy(rng, E_THR, eps=0.5)
    co, ret, traj = rollout(pol, n, seed + 50_000, POP, platform=platform,
                            record=True, kR=kR, **spec)
    return traj


def fit_policy(traj, prior_traj=None, L=5, k=2, prior_ess="cv", delta=0.10,
               level=0.90, constraint="ucb", disc=None):
    obs0 = traj[0]["obs"]
    allE = np.concatenate([tr["obs"]["logE"] for tr in traj])
    allH = np.concatenate([tr["obs"]["logH"] for tr in traj])
    disc = disc or Discretizer(L).fit(allE, allH)
    D = collect(traj, disc)
    P = collect(prior_traj, disc) if prior_traj is not None else None
    model = KTreeModel(disc.card, k=k, prior_ess=prior_ess)
    model.fit(D, prior=P)
    plan = Planner(model, D, W, delta=delta, level=level, constraint=constraint).solve()
    return plan.policy(disc), dict(model=model, plan=plan, disc=disc, D=D)


def metrics(co, ret, traj, delta=0.10):
    gave = co.dose_hist > 0
    g3 = co.tox_hist >= 3
    risk = co.true_risk_hist
    L = co.latent()
    boosts = gave.copy()
    boosts[:, 0] = False                     # prime is common to all policies
    nb = boosts.sum()
    out = dict(
        V=float(ret.mean()),
        doses=float(gave.sum(1).mean()),
        tumour_logchange=float(np.median(co.X)),
        resp30=float((co.X <= np.log(0.7)).mean()),
        prog20=float((co.X >= np.log(1.2)).mean()),
        logE24=float(np.median(np.log10(L["ELISpot"] + 10))),
        logH24=float(np.median(np.log10(L["IgG"]))),
        g3_per_boost=float(g3[boosts].mean()) if nb else 0.0,
        any_g3=float(g3.any(1).mean()),
        true_risk_per_boost=float(risk[boosts].mean()) if nb else 0.0,
        viol_rate=float((risk[boosts] > delta).mean()) if nb else 0.0,
        p95_risk=float(np.quantile(risk[boosts], 0.95)) if nb else 0.0,
    )
    # mean interval between consecutive doses (weeks)
    iv = []
    for row in gave:
        w = np.where(row)[0]
        if len(w) > 1:
            iv.extend(np.diff(w))
    out["mean_interval"] = float(np.mean(iv)) if iv else np.nan
    return out, ret


def evaluate(policy, seed, n=1000, platform="circRNA", delta=0.10, **ckw):
    co, ret, traj = rollout(policy, n, seed, POP, platform=platform, record=True, **ckw)
    m, r = metrics(co, ret, traj, delta)
    return m, r, co, traj
