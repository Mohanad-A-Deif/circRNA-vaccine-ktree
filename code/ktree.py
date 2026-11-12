"""Hybrid mechanistic-prior Markov k-tree transition model and safety-constrained planner.

Discrete state z = (C, H, U, Tox, R, S, N):
  C   T-cell readout bin (log10 ELISpot, quantile bins, L levels)
  H   antibody readout bin (log10 IgG, quantile bins, L levels)
  U   tumour-burden change from baseline (shrinking / stable / growing)
  Tox severe toxicity in the last week (grade >= 3: no / yes)
  R   reactogenicity history (any grade >= 2 observed so far: no / yes)
  B   baseline risk marker (low / high; static)
  S   weeks since last dose ({0}, {1}, {2-3}, {>=4})
  N   doses received ({1}, {2-3}, {4-7}, {8})

Two-slice transition model: every next-state variable depends on its own past
value, the action, and at most k further current-state variables (its parent
set). Each family therefore spans at most k+2 variables, and the learned
structure is a bounded in-degree (Markov k-tree-type) dynamic Bayesian network.
Parent sets are chosen by exhaustive search maximising patient-grouped
cross-validated predictive log-likelihood; conditional probability tables
(CPTs) are Dirichlet posteriors whose prior pseudo-counts come from a
mechanistic simulation (hybridisation). Policies are computed by finite-horizon
backward induction with a chance constraint on severe toxicity enforced through
the posterior upper credible bound of P(Tox'=3 | z, boost).
"""
import itertools
import numpy as np
from scipy.stats import beta as beta_dist

NODES = ["C", "H", "U", "Tox", "R", "B", "S", "N"]
NA = 3
DOSE = np.array([0.0, 1.0, 0.5])
U_CUTS = np.array([-0.15, 0.15])


def s_bin(wsd):
    return np.digitize(wsd, [1, 2, 4])             # 0,1,2-3,4+


def n_bin(nd):
    return np.digitize(nd, [2, 4, 8])              # 1,2-3,4-7,8


def tox_bin(g):
    return (np.asarray(g) >= 3).astype(int)        # grade >= 3


class Discretizer:
    def __init__(self, L=5):
        self.L = L

    def fit(self, logE, logH):
        q = np.linspace(0, 1, self.L + 1)[1:-1]
        self.eC = np.quantile(logE, q)
        self.eH = np.quantile(logH, q)
        return self

    @property
    def card(self):
        return [self.L, self.L, 3, 2, 2, 2, 4, 4]

    def transform(self, obs):
        return np.stack([np.digitize(obs["logE"], self.eC),
                         np.digitize(obs["logH"], self.eH),
                         np.digitize(obs["X"], U_CUTS),
                         tox_bin(obs["tox"]),
                         (obs["maxtox"] >= 2).astype(int),
                         obs["brisk"].astype(int),
                         s_bin(obs["wsd"]),
                         n_bin(obs["ndoses"])], axis=1)


def collect(traj, disc):
    """Turn recorded rollouts into transition arrays (z, a, z', features)."""
    Z, A, Z2, F = [], [], [], []
    for k in range(len(traj) - 1):
        z = disc.transform(traj[k]["obs"])
        z2 = disc.transform(traj[k + 1]["obs"])
        o, o2 = traj[k]["obs"], traj[k + 1]["obs"]
        Z.append(z); A.append(traj[k]["a"]); Z2.append(z2)
        F.append(dict(logE2=o2["logE"], logH2=o2["logH"], dX=o2["X"] - o["X"],
                      tox2=o2["tox"], t=o["t"], pid=np.arange(len(z))))
    out = dict(Z=np.concatenate(Z), A=np.concatenate(A), Z2=np.concatenate(Z2))
    for key in F[0]:
        out[key] = np.concatenate([f[key] for f in F])
    return out


class KTreeModel:
    ESS_GRID = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0)

    def __init__(self, card, k=2, prior_ess="cv", alpha0=0.5, fixed=("B", "S", "N")):
        self.card = list(card)
        self.k = k
        self.prior_ess = prior_ess
        self.alpha0 = alpha0
        self.fixed = set(fixed)
        self.parents = {}
        self.ess = {}
        self.cpt = {}
        self.score = {}

    # ---------------------------------------------------------------- utils
    def _cfg_index(self, Z, cols):
        idx = np.zeros(len(Z), np.int64)
        for c in cols:
            idx = idx * self.card[c] + Z[:, c]
        return idx

    def _ncfg(self, cols):
        return int(np.prod([self.card[c] for c in cols])) if cols else 1

    def _counts(self, Z, A, Z2, child, pa):
        cols = [child] + list(pa)
        cfg = self._cfg_index(Z, cols) * NA + A
        ncfg = self._ncfg(cols) * NA
        Lc = self.card[child]
        cnt = np.bincount(cfg * Lc + Z2[:, child], minlength=ncfg * Lc).reshape(ncfg, Lc)
        return cnt.astype(float)

    # ------------------------------------------------------------- learning
    def _pseudo(self, prior, i, pa, ess=None):
        ess = self.prior_ess if ess is None else ess
        if prior is None or ess <= 0:
            return self.alpha0
        pc = self._counts(prior["Z"], prior["A"], prior["Z2"], i, pa)
        tot = pc.sum(1, keepdims=True)
        pprob = np.where(tot > 0, pc / np.maximum(tot, 1), 1.0 / self.card[i])
        return ess * pprob + self.alpha0

    def cv_loglik(self, D, prior, i, pa, folds=5, seed=0, ess=None):
        """Patient-grouped K-fold held-out log-likelihood of child i given pa."""
        Z, A, Z2 = D["Z"], D["A"], D["Z2"]
        pid = D["pid"]
        rng = np.random.default_rng(seed)
        f_of = rng.integers(0, folds, pid.max() + 1)[pid]
        pseudo = self._pseudo(prior, i, pa, ess)
        cols = [i] + list(pa)
        cfg = self._cfg_index(Z, cols) * NA + A
        tot = 0.0
        for f in range(folds):
            tr = f_of != f
            cnt = self._counts(Z[tr], A[tr], Z2[tr], i, pa)
            th = (cnt + pseudo) / (cnt + pseudo).sum(1, keepdims=True)
            te = ~tr
            tot += np.log(th[cfg[te], Z2[te, i]]).sum()
        return tot

    def fit(self, D, prior=None, candidates=None):
        Z, A, Z2 = D["Z"], D["A"], D["Z2"]
        m = len(NODES)
        for i, name in enumerate(NODES):
            if name in self.fixed:
                options = [()]
            else:
                others = [j for j in range(m) if j != i] if candidates is None else candidates
                if self.k >= len(others):        # saturated: condition on the full state
                    options = [tuple(others)]
                else:
                    options = [pa for r in range(self.k + 1)
                               for pa in itertools.combinations(others, r)]
            # parent set and prior strength chosen jointly by held-out likelihood;
            # the data decide how far to trust the mechanistic prior
            if prior is None:
                grid = (0.0,)
            elif self.prior_ess == "cv":
                grid = self.ESS_GRID
            else:
                grid = (float(self.prior_ess),)
            scores = {(pa, e): self.cv_loglik(D, prior, i, pa, ess=e)
                      for pa in options for e in grid}
            best, ess = max(scores, key=scores.get)
            self.parents[i] = best
            self.ess[i] = ess
            self.score[i] = scores
            cnt = self._counts(Z, A, Z2, i, best)
            self.cpt[i] = dict(counts=cnt, pseudo=self._pseudo(prior, i, best, ess))
        return self

    def probs(self, i, Zall):
        """P(child_i' | z, a) for every state row in Zall -> (nz, NA, L_i)."""
        pa = self.parents[i]
        cols = [i] + list(pa)
        base = self._cfg_index(Zall, cols) * NA
        post = self.cpt[i]["counts"] + self.cpt[i]["pseudo"]
        post = post / post.sum(1, keepdims=True)
        return np.stack([post[base + a] for a in range(NA)], axis=1)

    def tox_ucb(self, Zall, level=0.90):
        """Posterior upper credible bound of P(Tox'=grade>=3 | z, a)."""
        i = NODES.index("Tox")
        cols = [i] + list(self.parents[i])
        base = self._cfg_index(Zall, cols) * NA
        a_ = self.cpt[i]["counts"] + self.cpt[i]["pseudo"]
        a3 = a_[:, 1]
        rest = a_.sum(1) - a3
        ucb = beta_dist.ppf(level, a3, rest)
        mean = a3 / (a3 + rest)
        return (np.stack([ucb[base + a] for a in range(NA)], axis=1),
                np.stack([mean[base + a] for a in range(NA)], axis=1))


def all_states(card):
    grids = np.meshgrid(*[np.arange(c) for c in card], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


class Planner:
    """Finite-horizon safety-constrained backward induction on the factored model."""

    def __init__(self, model, D, W, gamma=0.98, T=24, delta=0.10, level=0.90,
                 constraint="ucb"):
        self.m = model
        self.card = model.card
        self.gamma, self.T = gamma, T
        self.delta, self.level, self.constraint = delta, level, constraint
        self.W = W
        self.Zs = all_states(self.card)
        self._reward_tables(D)

    def _reward_tables(self, D):
        L = self.card
        cbar = np.array([np.mean(D["logE2"][D["Z2"][:, 0] == b]) if (D["Z2"][:, 0] == b).any()
                         else np.nan for b in range(L[0])])
        hbar = np.array([np.mean(D["logH2"][D["Z2"][:, 1] == b]) if (D["Z2"][:, 1] == b).any()
                         else np.nan for b in range(L[1])])
        cbar = np.where(np.isnan(cbar), np.nanmean(cbar), cbar)
        hbar = np.where(np.isnan(hbar), np.nanmean(hbar), hbar)
        dx = np.zeros((3, 3))
        for u in range(3):
            for u2 in range(3):
                mk = (D["Z"][:, 2] == u) & (D["Z2"][:, 2] == u2)
                dx[u, u2] = D["dX"][mk].mean() if mk.sum() >= 5 else 0.15 * (u2 - u)
        self.cbar, self.hbar, self.dx = cbar, hbar, dx

    def solve(self):
        Zs, card = self.Zs, self.card
        nz = len(Zs)
        P = [self.m.probs(i, Zs) for i in range(len(NODES))]   # each (nz, NA, Li)
        W = self.W
        # immediate expected reward (additive over nodes)
        r = (W["cell"] * (P[0] @ (self.cbar / 3.0))
             + W["hum"] * (P[1] @ (self.hbar / 6.0))
             + W["tum"] * 10 * (-(P[2] * self.dx[Zs[:, 2]][:, None, :]).sum(-1))
             - W["tox"] * P[3][:, :, 1]
             - W["dose"] * DOSE[None, :])
        # feasibility and safety masks
        feas = np.ones((nz, NA), bool)
        feas[Zs[:, 6] == 0, 1:] = False          # dosed last week
        feas[Zs[:, 7] == 3, 1:] = False          # 8 doses reached
        ucb, mean = self.m.tox_ucb(Zs, self.level)
        self.tox_ucb_tab, self.tox_mean_tab = ucb, mean
        risk = ucb if self.constraint == "ucb" else mean
        safe = np.ones((nz, NA), bool)
        if self.delta < 1:
            safe[:, 1:] = risk[:, 1:] <= self.delta
        allowed = feas & safe
        allowed[:, 0] = True
        self.allowed = allowed
        V = np.zeros(card)
        self.pi = np.zeros((self.T, nz), int)
        self.Q = np.zeros((self.T, nz, NA))
        factors = self._factors()
        for t in range(self.T - 1, 0, -1):
            EV = self._expect(V, factors).reshape(nz, NA)
            Q = r + self.gamma * EV
            Q = np.where(allowed, Q, -np.inf)
            self.Q[t] = Q
            self.pi[t] = Q.argmax(1)
            V = Q.max(1).reshape(card)
        self.r = r
        return self

    def _factors(self):
        """CPTs as tensors over (family current vars..., action, child next var)."""
        fac = []
        for i in range(len(NODES)):
            cols = [i] + list(self.m.parents[i])
            post = self.m.cpt[i]["counts"] + self.m.cpt[i]["pseudo"]
            post = post / post.sum(1, keepdims=True)
            shape = [self.card[c] for c in cols] + [NA, self.card[i]]
            fac.append((cols, post.reshape(shape)))
        return fac

    def _expect(self, V, factors):
        """E[V(z') | z, a] by sequential variable elimination over next-state nodes.
        Returns array of shape card + (NA,)."""
        letters = "abcdefghijklmnopqrstuvw"
        cur = [letters[j] for j in range(len(NODES))]            # current vars
        nxt = [letters[8 + j] for j in range(len(NODES))]        # next vars
        act = "y"
        T, axes = V, list(nxt)                                    # tensor over next vars
        # eliminate nodes with the smallest families first
        order = sorted(range(len(NODES)), key=lambda i: len(factors[i][0]))
        for i in order:
            cols, F = factors[i]
            f_axes = [cur[c] for c in cols] + [act, nxt[i]]
            out = [ax for ax in axes if ax != nxt[i]]
            for ax in f_axes[:-1]:
                if ax not in out:
                    out.append(ax)
            T = np.einsum("".join(axes) + "," + "".join(f_axes) + "->" + "".join(out),
                          T, F, optimize=True)
            axes = out
        # broadcast to the full current state space and order axes
        full = cur + [act]
        missing = [ax for ax in full if ax not in axes]
        for ax in missing:
            T = np.expand_dims(T, -1)
            axes = axes + [ax]
        T = np.einsum("".join(axes) + "->" + "".join(full), T)
        return np.broadcast_to(T, tuple(self.card) + (NA,))

    def strides(self):
        s = np.ones(len(self.card), np.int64)
        for j in range(len(self.card) - 2, -1, -1):
            s[j] = s[j + 1] * self.card[j + 1]
        return s

    def policy(self, disc):
        st = self.strides()

        def pol(obs, co):
            z = disc.transform(obs)
            idx = z @ st
            t = int(obs["t"][0])
            return self.pi[min(t, self.T - 1)][idx]
        return pol
