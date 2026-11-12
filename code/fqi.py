"""Model-free offline RL baseline: finite-horizon fitted Q-iteration (extra trees).
Uses the same logged data and an observed-reward proxy with a toxicity penalty
(soft safety only, no chance constraint)."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sim import W, DOSE, T_WEEKS

NA = 3


def feats(o):
    return np.c_[o["logE"], o["logH"], o["X"], o["tox"], o["maxtox"], o["brisk"],
                 np.minimum(o["wsd"], 10), o["ndoses"], o["t"]]


def build(traj):
    S, A, R, S2, done = [], [], [], [], []
    for k in range(len(traj) - 1):
        o, o2, a = traj[k]["obs"], traj[k + 1]["obs"], traj[k]["a"]
        r = (W["cell"] * o2["logE"] / 3 + W["hum"] * o2["logH"] / 6
             + W["tum"] * 10 * (-(o2["X"] - o["X"])) - W["tox"] * (o2["tox"] >= 3)
             - W["dose"] * DOSE[a])
        S.append(feats(o)); A.append(a); R.append(r); S2.append(feats(o2))
        done.append(np.full(len(a), k == len(traj) - 2))
    return [np.concatenate(x) for x in (S, A, R, S2, done)]


def feasible(F):
    ok = np.ones((len(F), NA), bool)
    ok[F[:, 6] < 1, 1:] = False     # dosed last week
    ok[F[:, 7] >= 8, 1:] = False
    return ok


def fit_fqi(traj, iters=20, gamma=0.98, seed=0):
    S, A, R, S2, done = build(traj)
    X = np.c_[S, np.eye(NA)[A]]
    ok2 = feasible(S2)
    model = None
    y = R.copy()
    for it in range(iters):
        model = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=10,
                                    random_state=seed, n_jobs=1).fit(X, y)
        q2 = np.stack([model.predict(np.c_[S2, np.tile(np.eye(NA)[a], (len(S2), 1))])
                       for a in range(NA)], 1)
        q2 = np.where(ok2, q2, -np.inf).max(1)
        y = R + gamma * np.where(done, 0.0, q2)

    def pol(obs, co):
        F = feats(obs)
        q = np.stack([model.predict(np.c_[F, np.tile(np.eye(NA)[a], (len(F), 1))])
                      for a in range(NA)], 1)
        q = np.where(feasible(F), q, -np.inf)
        return q.argmax(1)
    return pol
