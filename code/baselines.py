"""Reference schedules and a clinician-style rule."""
import numpy as np


def fixed_schedule(weeks, dose_action=1):
    weeks = set(weeks)

    def pol(obs, co):
        t = int(obs["t"][0])
        return np.full(len(obs["t"]), dose_action if t in weeks else 0)
    return pol


Q3W = fixed_schedule(range(3, 24, 3))          # weeks 0,3,...,21 (8 doses)
PRIME_BOOST = fixed_schedule([4, 8, 16])        # weeks 0,4,8,16 (4 doses)
EXTENDED = fixed_schedule([8, 16])              # weeks 0,8,16 (3 doses)


def clinician_rule(E_thr):
    """Boost when the T-cell readout is below a threshold, at least 3 weeks after
    the last dose; use the reduced dose after a grade-2 event or when the
    baseline risk marker is high; hold after grade >= 3."""
    def pol(obs, co):
        a = np.zeros(len(obs["t"]), int)
        ok = (obs["wsd"] >= 2) & (obs["ndoses"] < 8)
        low = obs["logE"] < E_thr
        a[ok & low] = 1
        a[ok & low & ((obs["tox"] == 2) | (obs["brisk"] == 1))] = 2
        a[obs["tox"] >= 3] = 0
        return a
    return pol


def behaviour_policy(rng, E_thr, eps=0.15):
    """Logging policy for the training cohort: each patient follows one of the
    four reference strategies, with eps-random weekly deviations (as in a
    multi-arm study with protocol deviations)."""
    pols = [Q3W, PRIME_BOOST, EXTENDED, clinician_rule(E_thr)]
    assign = None

    def pol(obs, co):
        nonlocal assign
        n = len(obs["t"])
        if assign is None:
            assign = rng.integers(0, 4, n)
        a = np.zeros(n, int)
        for j, p in enumerate(pols):
            m = assign == j
            if m.any():
                a[m] = np.asarray(p(obs, co))[m]
        flip = rng.random(n) < eps
        a[flip] = rng.integers(0, 3, flip.sum())
        return a
    return pol
