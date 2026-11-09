"""
models.py
=========
PERAN FILE
    Pelatihan SVD dan NMF (Subbab 3.5), prediksi rating tervektorisasi, skor Top-N
    dengan ambang popularitas, dan fold-in profil untuk pengguna baru.

CATATAN
    - Hyperparameter berasal dari hasil tuning (tuning.py) untuk setiap kombinasi
      model x k; nilai tetap (biased) dari config.FIXED_PARAMS.
    - predict_ratings meniru aturan Surprise: SVD menambahkan suku bias yang
      diketahui; NMF tanpa bias hanya memprediksi bila pengguna dan film dikenal,
      selain itu memakai rata-rata global; hasil dipotong ke skala 1 sampai 5.
      Versi tervektorisasi ini jauh lebih cepat daripada algo.test.
    - Ambang popularitas (minimal 20 rating di data latih) diterapkan sama pada
      SVD dan NMF sebelum Top-N diambil.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from surprise import NMF, SVD

from src import config as C
from src.features import is_pseudo
from src.utils import Timer


def build_params(name: str, k: int, hp: dict, seed: int, biased: bool | None = None) -> dict:
    p = dict(C.FIXED_PARAMS[name])
    p.update(hp)
    if biased is not None:
        p["biased"] = biased
    p.update(n_factors=int(k), random_state=int(seed))
    return p


def train_model(name: str, trainset, params: dict):
    algo = SVD(**params) if name == "SVD" else NMF(**params)
    with Timer() as t:
        algo.fit(trainset)
    algo._train_seconds = t.seconds
    algo._name = name
    return algo


def predict_ratings(algo, trainset, uids, iids) -> np.ndarray:
    ru, ri = trainset._raw2inner_id_users, trainset._raw2inner_id_items
    u = np.array([ru.get(x, -1) for x in uids])
    i = np.array([ri.get(x, -1) for x in iids])
    ku, ki = u >= 0, i >= 0
    both = ku & ki
    mu = trainset.global_mean
    pu, qi = np.asarray(algo.pu), np.asarray(algo.qi)
    dot = np.zeros(len(u))
    dot[both] = np.einsum("ij,ij->i", pu[u[both]], qi[i[both]])
    biased = bool(getattr(algo, "biased", False))
    if algo._name == "SVD" or biased:
        if biased:
            est = np.full(len(u), mu)
            est[ku] += np.asarray(algo.bu)[u[ku]]
            est[ki] += np.asarray(algo.bi)[i[ki]]
            est += dot
        else:
            est = np.where(both, dot, mu)
    else:
        est = np.where(both, dot, mu)
    return np.clip(est, *C.RATING_SCALE)


def train_mae(algo, trainset, df, n: int = 50000, seed: int = 0):
    """(MAE latih pada sampel, MAE prediktor rata-rata global pada sampel yang sama).
    Model dianggap divergen bila nilai pertama lebih besar dari nilai kedua."""
    s = df.sample(n=min(n, len(df)), random_state=seed)
    est = predict_ratings(algo, trainset, s.userId.tolist(), s.movieId.tolist())
    r = s.rating.to_numpy()
    return float(np.abs(r - est).mean()), float(np.abs(r - trainset.global_mean).mean())


def eligible_items(trainset, rating_df) -> np.ndarray:
    """Mask item film yang boleh direkomendasikan (ambang popularitas)."""
    cnt = rating_df.groupby("movieId").size()
    mask = np.zeros(trainset.n_items, dtype=bool)
    for raw, inner in trainset._raw2inner_id_items.items():
        if not is_pseudo(raw) and cnt.get(raw, 0) >= C.POP_MIN_RATINGS:
            mask[inner] = True
    return mask


def movie_items(trainset) -> np.ndarray:
    mask = np.zeros(trainset.n_items, dtype=bool)
    for raw, inner in trainset._raw2inner_id_items.items():
        if not is_pseudo(raw):
            mask[inner] = True
    return mask


def score_matrix(algo, trainset) -> np.ndarray:
    S = np.asarray(algo.pu) @ np.asarray(algo.qi).T
    if bool(getattr(algo, "biased", False)):
        S += np.asarray(algo.bu)[:, None] + np.asarray(algo.bi)[None, :] + trainset.global_mean
    return S


def topn_from_scores(S: np.ndarray, allowed: np.ndarray, n: int, seen=None) -> np.ndarray:
    S = S.copy()
    S[:, ~allowed] = -np.inf
    if seen is not None:
        for row, cols in enumerate(seen):
            if len(cols):
                S[row, cols] = -np.inf
    n = min(n, int(allowed.sum()) - 1)
    part = np.argpartition(-S, n, axis=1)[:, :n]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, part], axis=1)
    return part[rows, order]


def seen_lists(trainset) -> list:
    return [np.array([i for (i, _) in trainset.ur[u]], dtype=int) for u in range(trainset.n_users)]


# --------------------------------------------------------------------------
# Fold-in pengguna baru (cold-start)
# --------------------------------------------------------------------------
def extract_params(algo, trainset, params: dict) -> dict:
    name = algo._name
    biased = bool(getattr(algo, "biased", False))
    P = {"name": name, "biased": biased, "qi": np.asarray(algo.qi),
         "mu": float(trainset.global_mean), "pu_mean": np.asarray(algo.pu).mean(0)}
    if biased:
        P["bi"] = np.asarray(algo.bi)
        P["bu_mean"] = float(np.asarray(algo.bu).mean())
    else:
        P["bi"] = np.zeros(trainset.n_items)
        P["bu_mean"] = 0.0
    P["lam"] = float(params.get("reg_all", params.get("reg_pu", 0.02)))
    return P


def fold_in(P: dict, obs, alpha: int):
    """Vektor laten pengguna baru dari pseudo-rating profilnya. Kembali (p, b_u).

    obs: list (indeks inner pseudo-item, nilai). Regularisasi mengikuti fungsi objektif
    pelatihan: pada SVD (SGD Surprise, Koren dkk. 2009) dan NMF Surprise, suku lambda
    dikenakan pada setiap rating, sehingga penalti efektif per pengguna = lambda x jumlah
    rating. Dengan alpha replikasi, jumlah rating pseudo = alpha x n_obs, sehingga

        min_z  alpha * ||y - X z||^2 + lambda * alpha * n_obs * ||z - z0||^2

    dan alpha saling meniadakan. z0 = rata-rata vektor (dan bias) pengguna latih, yaitu
    profil Default. Versi v3.1 memakai lambda tanpa dikali jumlah observasi, sehingga
    vektor pengguna baru kurang diregularisasi dan prediksinya meledak pada kelompok kecil
    (diperbaiki di v3.2).
    """
    pbar, lam = P["pu_mean"], P["lam"]
    if P["name"] == "SVD":
        z0 = np.append(pbar, P["bu_mean"])
        if not obs or alpha <= 0:
            return z0[:-1], float(z0[-1])
        idx = [i for i, _ in obs]
        X = np.hstack([P["qi"][idx], np.ones((len(idx), 1))])
        y = np.array([r for _, r in obs]) - P["mu"] - P["bi"][idx]
        A = X.T @ X + lam * len(idx) * np.eye(X.shape[1])
        z = z0 + np.linalg.solve(A, X.T @ (y - X @ z0))
        return z[:-1], float(z[-1])
    if not obs or alpha <= 0:
        return pbar, 0.0
    idx = [i for i, _ in obs]
    Q = P["qi"][idx]
    y = np.array([r for _, r in obs])
    if P["biased"]:
        y = y - P["mu"] - P["bi"][idx] - P["bu_mean"]
    lam_eff = lam * len(idx)
    A = np.vstack([Q, np.sqrt(lam_eff) * np.eye(Q.shape[1])])
    b = np.concatenate([y, np.sqrt(lam_eff) * pbar])
    p, _ = nnls(A, b, maxiter=5000)
    return p, (P["bu_mean"] if P["biased"] else 0.0)


def profile_scores(P: dict, p, bu, item_idx: np.ndarray) -> np.ndarray:
    s = P["qi"][item_idx] @ p
    if P["name"] == "SVD" or P["biased"]:
        s = s + P["mu"] + bu + P["bi"][item_idx]
    return s
