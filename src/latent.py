"""
latent.py
=========
PERAN FILE
    Inspeksi ruang laten (Tujuan 1 proposal) dan bukti mekanisme H1.

ISI UTAMA
    - group_stats    : norma vektor, porsi nilai negatif, porsi nilai nyaris nol per
                       gender, usia, dan kelompok pekerjaan.
    - probe          : regresi logistik menebak atribut dari vektor laten pengguna
                       (AUC untuk gender, balanced accuracy untuk usia dan kelompok
                       pekerjaan), skor per lipatan CV disimpan untuk uji antar fold data.
    - baseline_probe : regresi logistik menebak KELOMPOK PEKERJAAN hanya dari one-hot
                       gender dan usia, dengan lipatan CV yang sama. Kebocoran hanya sah
                       diuji bila pekerjaan BUKAN input (alpha = 0 dan ablation gender +
                       usia); bila pekerjaan menjadi input, probe pasti tinggi.
    - pseudo_factors : cosinus antar vektor faktor pseudo-item (F vs M, antar usia).
                       Inti bukti mekanisme nilai negatif.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config as C
from src.features import item_masks


def user_frame(trainset, users):
    raw = np.array([trainset.to_raw_uid(i) for i in range(trainset.n_users)])
    return users.set_index("userId").loc[raw]


def group_stats(pu, U) -> dict:
    out = {}
    for attr in C.SENSITIVE_ATTRS:
        lab = U[attr].to_numpy()
        out[attr] = {str(g): {"n": int((lab == g).sum()),
                              "norm_mean": float(np.linalg.norm(pu[lab == g], axis=1).mean()),
                              "frac_negative": float((pu[lab == g] < 0).mean()),
                              "frac_near_zero": float((np.abs(pu[lab == g]) < 1e-3).mean())}
                     for g in pd.unique(lab)}
    out["overall"] = {"frac_negative": float((pu < 0).mean()),
                      "value_mean": float(pu.mean()), "value_std": float(pu.std())}
    return out


def _cv_scores(X, y, scoring, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=C.PROBE_FOLDS, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_score(clf, X, y, cv=skf, scoring=scoring)


def probe(pu, U, seed: int) -> dict:
    out = {}
    for attr in C.SENSITIVE_ATTRS:
        y = U[attr].astype(str).to_numpy()
        if pd.Series(y).value_counts().min() < C.PROBE_FOLDS:
            continue
        if attr == "gender":
            metric, chance = "roc_auc", 0.5
        else:
            metric, chance = "balanced_accuracy", 1.0 / len(np.unique(y))
        sc = _cv_scores(pu, y, metric, seed)
        out[attr] = {"metric": metric, "folds": sc.tolist(), "mean": float(sc.mean()),
                     "chance": chance}
    return out


def baseline_probe(U, seed: int) -> dict:
    X = pd.get_dummies(U[["gender", "age_group"]].astype(str)).to_numpy(float)
    y = U["occ_group"].astype(str).to_numpy()
    if pd.Series(y).value_counts().min() < C.PROBE_FOLDS:
        return {}
    sc = _cv_scores(X, y, "balanced_accuracy", seed)
    return {"occ_group": {"metric": "balanced_accuracy", "folds": sc.tolist(),
                           "mean": float(sc.mean()), "features": "one-hot gender + usia"}}


def pseudo_factors(algo, trainset) -> dict:
    _, pseudo = item_masks(trainset)
    if not pseudo:
        return {}
    qi = np.asarray(algo.qi)
    out = {}
    for attr in C.PROFILE_ATTRS:
        keys = [k for k in pseudo if k.startswith(f"D::{attr}::")]
        order = {"age_group": C.AGE_ORDER, "gender": C.GENDER_ORDER,
                 "occ_group": C.OCC_GROUP_ORDER}[attr]
        keys = sorted(keys, key=lambda k: order.index(k.split("::")[-1]))
        if not keys:
            continue
        Q = qi[[pseudo[k] for k in keys]]
        nrm = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
        cos = (Q / nrm) @ (Q / nrm).T
        out[attr] = {"labels": [k.split("::")[-1] for k in keys],
                     "norms": nrm.ravel().tolist(), "cosine": cos.tolist()}
    if "gender" in out and len(out["gender"]["labels"]) == 2:
        out["cos_F_M"] = float(out["gender"]["cosine"][0][1])
    return out


def inspect(algo, trainset, users, seed: int, run_probe: bool) -> dict:
    U = user_frame(trainset, users)
    pu = np.asarray(algo.pu)
    res = {"group_stats": group_stats(pu, U), "pseudo_factors": pseudo_factors(algo, trainset)}
    if run_probe:
        res["probe"] = probe(pu, U, seed)
        res["baseline_probe"] = baseline_probe(U, seed)
    return res
