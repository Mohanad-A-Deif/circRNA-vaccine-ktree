"""Mechanistic immune-kinetics module (daily time scale).

States (per subject, vectorised over subjects):
  R   vaccine RNA template (decays with k_R; circRNA has a smaller k_R)
  A   expressed antigen
  G   germinal-centre activity
  Bm  matured memory-B pool
  P   antibody-secreting cells
  H   circulating IgG (arbitrary units)
  E   antigen-specific effector CD8+ T cells
  M   primed T-cell precursor pool (antigen-driven, contracts between doses)

Doses are impulses on R. The boost amplification terms (1+kB*Bm) and (1+kT*M)
produce the two interval effects seen in human data: a long interval lets the
memory-B pool mature (higher antibody boost), while the primed T-cell
precursor pool contracts (smaller T-cell boost).
"""
import numpy as np

PARAM_NAMES = ["kR", "kA", "g", "dG", "m", "dBm", "p", "kB", "dP", "dH",
               "s", "kT", "dE", "f", "dM"]

# reference values (per day). Fitted values are produced by pitch_fit.py
DEFAULT = dict(kR=0.35, kA=0.25, g=1.0, dG=0.12, m=0.05, dBm=0.01,
               p=1.0, kB=3.0, dP=0.08, dH=np.log(2) / 28.0,
               s=1.0, kT=2.0, dE=0.03, f=0.5, dM=0.05)


def simulate(params, dose_days, dose_amt, t_end, dt=0.25, record_days=None,
             x0=None):
    """Euler integration for n subjects.

    params     : dict name -> scalar or (n,) array
    dose_days  : (n, J) array of dose days (np.nan = no dose)
    dose_amt   : (n, J) array of dose amounts (1 = full dose)
    record_days: 1-D array of days at which to record states (default: daily)
    returns dict state -> (n, len(record_days))
    """
    dose_days = np.atleast_2d(np.asarray(dose_days, float))
    dose_amt = np.atleast_2d(np.asarray(dose_amt, float))
    n = dose_days.shape[0]
    P_ = {k: np.broadcast_to(np.asarray(params[k], float), (n,)).copy()
          for k in PARAM_NAMES}
    if record_days is None:
        record_days = np.arange(0, int(t_end) + 1)
    record_days = np.asarray(record_days, float)
    names = ["R", "A", "G", "Bm", "P", "H", "E", "M"]
    X = {k: np.zeros(n) for k in names} if x0 is None else {k: x0[k].copy() for k in names}
    out = {k: np.full((n, len(record_days)), np.nan) for k in names}
    steps = int(round(t_end / dt))
    rec_idx = np.round(record_days / dt).astype(int)
    rec_map = {}
    for j, s in enumerate(rec_idx):
        rec_map.setdefault(s, []).append(j)
    dose_step = np.round(np.nan_to_num(dose_days, nan=-1e9) / dt).astype(np.int64)
    for s in range(steps + 1):
        # dose impulses
        hit = dose_step == s
        if hit.any():
            X["R"] += (dose_amt * hit).sum(axis=1)
        if s in rec_map:
            for j in rec_map[s]:
                for k in names:
                    out[k][:, j] = X[k]
        R, A, G, Bm, Pc, H, E, M = (X[k] for k in names)
        dR = -P_["kR"] * R
        dA = R - P_["kA"] * A
        dG = P_["g"] * A - P_["dG"] * G
        dBm = P_["m"] * G - P_["dBm"] * Bm
        dP = P_["p"] * A * (1 + P_["kB"] * Bm) - P_["dP"] * Pc
        dH = Pc - P_["dH"] * H
        dE = P_["s"] * A * (1 + P_["kT"] * M) - P_["dE"] * E
        dM = P_["f"] * A - P_["dM"] * M
        X["R"] = R + dt * dR
        X["A"] = A + dt * dA
        X["G"] = G + dt * dG
        X["Bm"] = Bm + dt * dBm
        X["P"] = Pc + dt * dP
        X["H"] = H + dt * dH
        X["E"] = E + dt * dE
        X["M"] = M + dt * dM
    return out
