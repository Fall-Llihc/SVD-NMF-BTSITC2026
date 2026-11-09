"""
features.py
===========
PERAN FILE
    Membentuk representasi pengguna = fitur kolaboratif + fitur demografis
    (pseudo-item gender, usia, dan kelompok pekerjaan), termasuk tiga varian encoding untuk ablation mekanisme.

VARIAN ENCODING
    full       : setiap pengguna merating SEMUA pseudo-item atribut input; anggota
                 kelompok bernilai 5, bukan anggota bernilai 1 (encoding v2).
    member5    : pengguna hanya merating pseudo-item kelompoknya sendiri, nilai 5.
    membermean : pengguna hanya merating pseudo-item kelompoknya sendiri, nilai =
                 rata-rata rating film pengguna itu di data latih. Varian ini
                 menghindari pseudo-item bernilai konstan yang seluruhnya dapat
                 diserap bias item pada SVD.

BOBOT ALPHA
    alpha adalah jumlah replikasi setiap entri pseudo-rating. Karena fungsi objektif
    SVD dan NMF menjumlahkan galat setiap entri, replikasi setara dengan bobot
    suku demografis. alpha = 0 berarti tanpa pseudo-item (murni kolaboratif).
    Bobot relatif terhadap profil pengguna dilaporkan sebagai
    w_u = c * alpha / (n_u + c * alpha), c = jumlah pseudo-rating per pengguna.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from surprise import Dataset, Reader

from src import config as C


def pseudo_id(attr: str, value) -> str:
    return f"D::{attr}::{value}"


def is_pseudo(raw_iid) -> bool:
    return isinstance(raw_iid, str) and raw_iid.startswith("D::")


def attr_values(users: pd.DataFrame, attr: str) -> list:
    vals = users[attr].dropna().unique().tolist()
    if attr == "age_group":
        return [a for a in C.AGE_ORDER if a in vals]
    if attr == "occ_group":
        return [a for a in C.OCC_GROUP_ORDER if a in vals]
    return sorted(vals, key=str)


def pseudo_per_user(users: pd.DataFrame, attrs, encoding: str) -> int:
    """c pada rumus bobot relatif."""
    if encoding == "full":
        return int(sum(len(attr_values(users, a)) for a in attrs))
    return len(attrs)


def demographic_rows(users, rating_df, attrs, alpha: int, encoding: str) -> pd.DataFrame:
    if not attrs or alpha <= 0:
        return pd.DataFrame(columns=["userId", "movieId", "rating"])
    uids = rating_df.userId.unique()
    u = users[users.userId.isin(set(uids))]
    parts = []
    if encoding == "membermean":
        umean = rating_df.groupby("userId").rating.mean()
    for attr in attrs:
        for v in attr_values(users, attr):
            member = (u[attr] == v).to_numpy()
            if encoding == "full":
                r = np.where(member, C.DEMO_HIGH, C.DEMO_LOW)
                ids = u.userId.to_numpy()
            elif encoding == "member5":
                ids = u.userId.to_numpy()[member]
                r = np.full(len(ids), C.DEMO_HIGH)
            elif encoding == "membermean":
                ids = u.userId.to_numpy()[member]
                r = umean.reindex(ids).to_numpy()
            else:
                raise ValueError(f"encoding tidak dikenal: {encoding}")
            parts.append(pd.DataFrame({"userId": ids, "movieId": pseudo_id(attr, v), "rating": r}))
    rows = pd.concat(parts, ignore_index=True)
    return pd.concat([rows] * alpha, ignore_index=True) if alpha > 1 else rows


def build_trainset(rating_df, users, attrs, alpha: int, encoding: str = "full"):
    reader = Reader(rating_scale=C.RATING_SCALE)
    base = rating_df[["userId", "movieId", "rating"]]
    if not attrs or alpha <= 0:
        return Dataset.load_from_df(base, reader).build_full_trainset()
    pseudo = demographic_rows(users, base, attrs, alpha, encoding)
    df = pd.concat([base.astype(object), pseudo.astype(object)], ignore_index=True)
    df["rating"] = df["rating"].astype(float)
    return Dataset.load_from_df(df, reader).build_full_trainset()


def item_masks(trainset):
    """(mask boolean item film, dict raw pseudo-id -> indeks inner)."""
    movie_mask = np.ones(trainset.n_items, dtype=bool)
    pseudo = {}
    for raw, inner in trainset._raw2inner_id_items.items():
        if is_pseudo(raw):
            movie_mask[inner] = False
            pseudo[raw] = inner
    return movie_mask, pseudo


def relative_weight(rating_df, users, attrs, alpha: int, encoding: str) -> dict:
    if alpha <= 0 or not attrs:
        return {"c": 0, "median_w": 0.0, "q10_w": 0.0, "q90_w": 0.0}
    c = pseudo_per_user(users, attrs, encoding)
    n = rating_df.groupby("userId").size().to_numpy()
    w = c * alpha / (n + c * alpha)
    return {"c": c, "median_w": float(np.median(w)), "q10_w": float(np.quantile(w, .1)),
            "q90_w": float(np.quantile(w, .9))}


def preview_rows(users, rating_df, user_id: int, alpha: int, encoding: str):
    return demographic_rows(users, rating_df[rating_df.userId == user_id],
                            C.PROFILE_ATTRS, alpha, encoding)
