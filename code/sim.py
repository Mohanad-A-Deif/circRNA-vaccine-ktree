"""Virtual-patient simulator for therapeutic RNA cancer-vaccine boosting (ground truth).

Kinetics: mech.py structure with population parameters calibrated on PITCH
(results/pitch_fit.json); between-patient heterogeneity is log-normal.
Platform: circRNA templates persist longer than linear mRNA (smaller kR).
Tumour: log-burden X grows at rate g and is reduced by effector T cells.
Toxicity: weekly CTCAE-like grade from a cumulative-logit model driven by the
dose given that week, recent dosing density, patient frailty and recent
toxicity (sensitisation). The decision maker never sees these equations.

Decisions are weekly: 0 = Defer, 1 = Boost (full dose), 2 = Boost (reduced dose).
"""
import json
import numpy as np

ACTIONS = ["Defer", "BoostFull", "BoostReduced"]
DOSE = np.array([0.0, 1.0, 0.5])
T_WEEKS = 24
DT = 0.05  # days (explicit Euler; stable for the fitted rate constants)


def load_pop(path="results/pitch_fit.json"):
    f = json.load(open(path))
    p = dict(f["params"])
    p["cH"], p["cE"] = f["cH"], f["cE"]
    return p


PLATFORM_KR = {"mRNA": 0.35, "circRNA": 0.12}


class Cohort:
    """Vectorised cohort of n virtual patients."""
    IMM_SCALE = 0.15

    def __init__(self, n, rng, pop, platform="circRNA", kR=None, het=1.0,
                 tox_shift=0.0, imm_scale=None, kill_scale=1.0, simple_tox=False):
        self.n = n
        if imm_scale is not None:
            self.IMM_SCALE = imm_scale
        self.simple_tox = simple_tox
        self.rng = rng
        lg = lambda cv: np.exp(rng.normal(0, cv * het, n) - 0.5 * (cv * het) ** 2)
        p = {k: np.full(n, float(v)) for k, v in pop.items()
             if k not in ("cH", "cE")}
        kr = PLATFORM_KR[platform] if kR is None else kR
        p["kR"] = kr * lg(0.25)
        # tumour neoantigens are less immunogenic than the viral spike used for
        # calibration: responsiveness is scaled by IMM_SCALE (sensitivity-tested)
        p["s"] = p["s"] * self.IMM_SCALE * lg(0.8)   # T-cell responsiveness
        p["p"] = p["p"] * self.IMM_SCALE * lg(0.7)   # B-cell responsiveness
        p["kT"] = p["kT"] * lg(0.4)
        p["kB"] = p["kB"] * lg(0.4)
        p["dH"] = p["dH"] * lg(0.25)
        self.p = p
        self.cH, self.cE = pop["cH"], pop["cE"]
        # tumour
        self.g = 0.004 * lg(0.5)            # per day
        self.kill = 0.010 * kill_scale * lg(0.6)   # per day at saturation
        self.KE = 120.0                      # SFU/1e6 at half-maximal kill
        # toxicity
        self.frail = rng.normal(0, 1, n)     # latent frailty
        # observable baseline risk marker (e.g. performance status / inflammation),
        # imperfectly correlated with latent frailty
        self.brisk = ((0.8 * self.frail + 0.6 * rng.normal(0, 1, n)) > 0.5).astype(int)
        self.tox_shift = tox_shift
        self.platform = platform
        # observation noise (log10 units) and missingness
        self.sd_E, self.sd_H, self.sd_X = 0.15, 0.10, 0.10
        self.p_miss = 0.10
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        n = self.n
        self.x = {k: np.zeros(n) for k in ["R", "A", "G", "Bm", "P", "H", "E", "M"]}
        self.X = np.zeros(n)            # log tumour burden relative to baseline
        self.t = 0
        self.last_dose_week = np.full(n, -99)
        self.n_doses = np.zeros(n, int)
        self.dose_hist = np.zeros((n, T_WEEKS))
        self.tox = np.zeros(n, int)     # grade observed in the last week
        self.tox_hist = np.zeros((n, T_WEEKS), int)
        self.true_risk_hist = np.full((n, T_WEEKS), np.nan)

    # --------------------------------------------------------------- toxicity
    TOX_CUTS = np.array([-0.3, 1.7, 3.6])   # cumulative-logit cut-points for >=1, >=2, >=3

    def tox_eta(self, dose):
        recent = self.dose_hist[:, max(0, self.t - 6):self.t].sum(axis=1)
        if self.simple_tox:   # population-average reactogenicity only
            return -2.6 + 2.4 * dose + self.tox_shift
        eta = (-3.0 + 2.4 * dose + 0.40 * recent * (dose > 0)
               + 1.1 * self.frail + 0.9 * (self.tox >= 2) + self.tox_shift)
        return eta

    def p_grade_ge3(self, dose):
        """True probability of grade>=3 this week if `dose` is given now."""
        return 1 / (1 + np.exp(-(self.tox_eta(dose) - self.TOX_CUTS[2])))

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply weekly action (array of ints), advance 7 days, return obs."""
        action = np.asarray(action)
        dose = DOSE[action]
        t = self.t
        self.true_risk_hist[:, t] = self.p_grade_ge3(np.where(dose > 0, dose, 0.0))
        eta = self.tox_eta(dose)
        u = self.rng.random(self.n)
        pge = 1 / (1 + np.exp(-(eta[:, None] - self.TOX_CUTS[None, :])))
        grade = (u[:, None] < pge).sum(axis=1)
        self.dose_hist[:, t] = dose
        self.tox = grade
        self.tox_hist[:, t] = grade
        gave = dose > 0
        self.last_dose_week = np.where(gave, t, self.last_dose_week)
        self.n_doses = self.n_doses + gave
        # kinetics over 7 days
        x, p = self.x, self.p
        x["R"] = x["R"] + dose
        Esum = np.zeros(self.n)
        steps = int(7 / DT)
        for _ in range(steps):
            R, A, G, Bm, Pc, H, E, M = (x[k] for k in ["R", "A", "G", "Bm", "P", "H", "E", "M"])
            x["R"] = R + DT * (-p["kR"] * R)
            x["A"] = A + DT * (R - p["kA"] * A)
            x["G"] = G + DT * (p["g"] * A - p["dG"] * G)
            x["Bm"] = Bm + DT * (p["m"] * G - p["dBm"] * Bm)
            x["P"] = Pc + DT * (p["p"] * A * (1 + p["kB"] * Bm) - p["dP"] * Pc)
            x["H"] = H + DT * (Pc - p["dH"] * H)
            x["E"] = E + DT * (p["s"] * A * (1 + p["kT"] * M) - p["dE"] * E)
            x["M"] = M + DT * (p["f"] * A - p["dM"] * M)
            Esfu = self.cE * np.maximum(x["E"], 0)
            Esum += Esfu
            self.X = self.X + DT * (self.g - self.kill * Esfu / (Esfu + self.KE))
        self.X = np.clip(self.X, -3.0, 2.0)
        self.t += 1
        return self.observe()

    def latent(self):
        return dict(ELISpot=self.cE * np.maximum(self.x["E"], 0) + 2.0,
                    IgG=self.cH * np.maximum(self.x["H"], 0) + 60.0,
                    X=self.X.copy())

    def observe(self):
        L = self.latent()
        r = self.rng
        obs = dict(
            logE=np.log10(L["ELISpot"] + 10) + r.normal(0, self.sd_E, self.n),
            logH=np.log10(L["IgG"]) + r.normal(0, self.sd_H, self.n),
            X=L["X"] + r.normal(0, self.sd_X, self.n),
            tox=self.tox.copy(),
            wsd=self.t - 1 - self.last_dose_week,   # weeks since last dose (at decision time)
            ndoses=self.n_doses.copy(),
            brisk=self.brisk.copy(),
            maxtox=self.tox_hist[:, :self.t].max(axis=1) if self.t > 0 else np.zeros(self.n, int),
            t=np.full(self.n, self.t))
        miss = r.random(self.n) < self.p_miss
        obs["miss"] = miss
        return obs


# ------------------------------------------------------------------ reward
W = dict(cell=0.30, hum=0.15, tum=0.60, tox=2.0, dose=0.03)


def reward_from_latent(L, X_prev, tox, dose):
    """True per-week clinical utility (used for evaluation only)."""
    c = np.log10(L["ELISpot"] + 10) / 3.0
    h = np.log10(L["IgG"]) / 6.0
    shrink = -(L["X"] - X_prev)          # tumour reduction this week (ln units)
    return (W["cell"] * c + W["hum"] * h + W["tum"] * 10 * shrink
            - W["tox"] * (tox >= 3) - W["dose"] * dose)


# ------------------------------------------------------------- rollout
def rollout(policy, n, seed, pop, platform="circRNA", record=False, **ckw):
    """Run a policy on a fresh cohort. policy(obs_hist_dict, cohort) -> actions.
    Week 0: prime dose (full) for every patient. Decisions at weeks 1..23."""
    rng = np.random.default_rng(seed)
    co = Cohort(n, rng, pop, platform=platform, **ckw)
    gamma = 0.98
    ret = np.zeros(n)
    traj = []
    L0 = co.latent()
    X_prev = L0["X"]
    obs = co.step(np.ones(n, int))          # prime at week 0
    L = co.latent()
    ret += reward_from_latent(L, X_prev, co.tox, 1.0)
    X_prev = L["X"]
    last = {k: v.copy() for k, v in obs.items()}
    for t in range(1, T_WEEKS):
        # carry forward biomarkers when the visit is missed
        cur = {k: v.copy() for k, v in obs.items()}
        for k in ("logE", "logH", "X"):
            cur[k] = np.where(obs["miss"], last[k], obs[k])
        last = cur
        a = np.asarray(policy(cur, co))
        # feasibility: minimum spacing 2 weeks, at most 8 doses
        too_soon = cur["wsd"] < 1
        maxed = cur["ndoses"] >= 8
        a = np.where((a > 0) & (too_soon | maxed), 0, a)
        if record:
            traj.append(dict(t=t, obs=cur, a=a.copy(), risk=co.p_grade_ge3(DOSE[a]),
                             risk_full=co.p_grade_ge3(np.ones(n))))
        obs = co.step(a)
        L = co.latent()
        r = reward_from_latent(L, X_prev, co.tox, DOSE[a])
        ret += gamma ** t * r
        X_prev = L["X"]
        if record:
            traj[-1].update(r=r, next_tox=co.tox.copy(), L=L)
    return co, ret, traj
