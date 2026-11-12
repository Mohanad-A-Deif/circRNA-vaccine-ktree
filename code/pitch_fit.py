"""Calibration and external validation of the mechanistic module on real human data.

Data: PITCH study (Payne et al., Cell 2021), de-identified Mendeley dataset
10.17632/fyp26zjgmj.1 (CC BY 4.0). BNT162b2 mRNA vaccine, two doses with a short
(17-35 d) or long (45-105 d) interval. We use infection-naive participants only.

Protocol: fit population parameters on LONG-interval participants only, then
predict SHORT-interval participants (never seen during fitting). This tests
whether the model captures how dose spacing changes humoral and cellular
responses.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from mech import simulate, DEFAULT

DATA = "data"
NOMINAL = {"V1plus28": 28, "V2plus28": 28, "V2plus90": 90}


def load():
    m = pd.read_csv(f"{DATA}/pitch_meta.csv", dtype={"ID": str})
    m["id"] = m.ID.astype(int)
    m["iv"] = pd.to_numeric(m.Vaccine_interval_days, errors="coerce")
    m = m[(m.Previous_infection == "N") & m.iv.notna()]
    e = pd.read_csv(f"{DATA}/pitch_elispot.csv", na_values=["NULL"])
    e["id"] = e.ID.astype(int)
    e = e[e.S1_PITCH_SFU.notna() | e.S2_PITCH_SFU.notna()]
    e["val"] = e.S1_PITCH_SFU.fillna(0) + e.S2_PITCH_SFU.fillna(0)
    e = e.rename(columns={"visit_reason": "visit", "exact_day": "xd"})
    e["assay"] = "T"
    a = pd.read_csv(f"{DATA}/pitch_msd.csv", na_values=["NULL"])
    a["id"] = a.ID.astype(int)
    a = a.rename(columns={"Visit_reason_MSD": "visit", "exact_days": "xd",
                          "Cov_2S_MSD": "val"})
    a["assay"] = "IgG"
    d = pd.concat([e[["id", "visit", "xd", "val", "assay"]],
                   a[["id", "visit", "xd", "val", "assay"]]])
    d = d.merge(m[["id", "iv", "Vaccine_interval_regminen"]], on="id")
    d = d.rename(columns={"Vaccine_interval_regminen": "regimen"})
    d = d[d.visit.isin(["V1plus28", "preV2", "V2plus28", "V2plus90"]) & d.val.notna()]
    d["xd"] = pd.to_numeric(d.xd, errors="coerce")
    # day since dose 1
    def day(r):
        x = r.xd
        if r.visit == "V1plus28":
            return x if np.isfinite(x) else 28
        if r.visit == "preV2":
            return x if np.isfinite(x) else r.iv - 1
        base = NOMINAL[r.visit]
        x = x if (np.isfinite(x) and 0 < x < 200) else base
        return r.iv + x
    d["t"] = d.apply(day, axis=1)
    # a sample labelled pre-dose-2 must precede dose 2
    d = d[~((d.visit == "preV2") & (d.t >= d.iv))]
    d = d[(d.visit != "V1plus28") | (d.t < d.iv)]
    d = d[d.t > 0]
    d = d.drop_duplicates(["id", "visit", "assay"])
    d["y"] = np.where(d.assay == "IgG", np.log10(d.val.clip(lower=1)),
                      np.log10(d.val.clip(lower=0) + 10))
    return d.reset_index(drop=True)


FREE = ["kA", "dG", "m", "dBm", "kB", "dP", "dH", "kT", "dE", "f", "dM"]


def unpack(theta):
    p = dict(DEFAULT)
    for k, v in zip(FREE, theta[:len(FREE)]):
        p[k] = float(np.exp(v))
    cH, cE = np.exp(theta[len(FREE)]), np.exp(theta[len(FREE) + 1])
    return p, cH, cE


def predict(theta, d, kR=None):
    p, cH, cE = unpack(theta)
    if kR is not None:
        p["kR"] = kR
    # one simulated trajectory per distinct interval, all intervals vectorised
    ivs = np.array(sorted(d.iv.unique()), float)
    days = np.array(sorted(d.t.astype(float).unique()))
    out = simulate(p, np.c_[np.zeros_like(ivs), ivs], np.ones((len(ivs), 2)),
                   t_end=days.max() + 1, dt=0.05, record_days=days)
    ri = np.searchsorted(ivs, d.iv.values.astype(float))
    ci = np.searchsorted(days, d.t.values.astype(float))
    H = out["H"][ri, ci]
    E = out["E"][ri, ci]
    isg = (d.assay.values == "IgG")
    yhat = np.where(isg, np.log10(cH * np.maximum(H, 0) + 60.0),
                    np.log10(cE * np.maximum(E, 0) + 2.0 + 10))
    return yhat


def fit(d):
    theta0 = np.r_[np.log([DEFAULT[k] for k in FREE]), np.log(2e3), np.log(2.0)]
    # scale residuals per assay by their spread so both assays count equally
    sd = d.groupby("assay").y.transform("std").values

    def res(th):
        return (predict(th, d) - d.y.values) / sd
    lo = theta0 - 6.0
    hi = theta0 + 6.0
    r = least_squares(res, theta0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0,
                      max_nfev=400, x_scale="jac")
    return r.x, r


def summarise(d, yhat, label):
    d = d.copy()
    d["yhat"] = yhat
    rows = []
    for (assay, visit), g in d.groupby(["assay", "visit"]):
        rows.append(dict(set=label, assay=assay, visit=visit, n=len(g),
                         obs_median=float(np.median(g.y)),
                         pred_median=float(np.median(g.yhat)),
                         rmse=float(np.sqrt(np.mean((g.y - g.yhat) ** 2))),
                         mae=float(np.mean(np.abs(g.y - g.yhat)))))
    return pd.DataFrame(rows)


def main():
    d = load()
    tr = d[d.regimen == "LONG"].copy()
    te = d[d.regimen == "SHORT"].copy()
    theta, r = fit(tr)
    yhat_tr = predict(theta, tr)
    yhat_te = predict(theta, te)
    # interval-agnostic reference: predict SHORT values with the LONG-arm visit medians
    ref = tr.groupby(["assay", "visit"]).y.median()
    yref = np.array([ref.get((a, v), np.nan) for a, v in zip(te.assay, te.visit)])
    s_tr = summarise(tr, yhat_tr, "train_LONG")
    s_te = summarise(te, yhat_te, "test_SHORT")
    s_ref = summarise(te, yref, "test_SHORT_reference")
    out = pd.concat([s_tr, s_te, s_ref])
    p, cH, cE = unpack(theta)
    json.dump(dict(params=p, cH=float(cH), cE=float(cE), theta=list(map(float, theta)),
                   n_train_obs=int(len(tr)), n_test_obs=int(len(te)),
                   n_train_subj=int(tr.id.nunique()), n_test_subj=int(te.id.nunique())),
              open("results/pitch_fit.json", "w"), indent=2)
    out.to_csv("results/pitch_validation.csv", index=False)
    d.to_csv("results/pitch_obs.csv", index=False)
    tr.assign(yhat=yhat_tr).to_csv("results/pitch_train_pred.csv", index=False)
    te.assign(yhat=yhat_te, yref=yref).to_csv("results/pitch_test_pred.csv", index=False)
    print(out.round(3).to_string())
    print(json.dumps(dict(params=p, cH=float(cH), cE=float(cE)), indent=1))


if __name__ == "__main__":
    main()
