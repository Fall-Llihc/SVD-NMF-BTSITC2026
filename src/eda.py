"""
eda.py
======
PERAN FILE
    Analisis eksploratori data sesuai Subbab 3.3 proposal. Setiap fungsi
    mengembalikan dict yang langsung dicatat ke JSON dan dipakai oleh plots.py.

ISI UTAMA
    - demographic_distribution (3.3.1): jumlah dan proporsi pengguna per kelompok,
      rasio ketimpangan (kelompok terbesar dibagi terkecil), serta proporsi rating.
    - rating_by_group (3.3.2): rata-rata, median, simpangan baku rating per kelompok,
      jumlah rating per pengguna, dan uji Kruskal-Wallis antar kelompok.
    - sparsity (3.3.3): sparsity keseluruhan dan per kelompok demografis.
    - item_popularity (3.3.4): distribusi jumlah rating per film, Gini, porsi
      rating yang diserap film terpopuler, dan definisi item ekor panjang.
    - genre_preference: distribusi genre film yang disukai (rating >= 4) per
      kelompok. Ini menjadi acuan "selera historis" untuk metrik amplifikasi genre.
    - run_eda: menjalankan semuanya sekaligus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.data import eval_users


def _gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _order(attr, keys):
    if attr == "age_group":
        return [a for a in C.AGE_ORDER if a in keys]
    if attr == "gender":
        return [g for g in C.GENDER_ORDER if g in keys]
    return sorted(keys)


def demographic_distribution(ratings, users) -> dict:
    eu = eval_users(users)
    rpu = ratings.merge(eu[["userId"] + C.SENSITIVE_ATTRS], on="userId")
    out = {}
    for attr in C.SENSITIVE_ATTRS:
        cnt = eu[attr].value_counts()
        rc = rpu[attr].value_counts()
        keys = _order(attr, cnt.index.tolist())
        out[attr] = {
            "users": {k: int(cnt[k]) for k in keys},
            "user_share": {k: float(cnt[k] / cnt.sum()) for k in keys},
            "ratings": {k: int(rc.get(k, 0)) for k in keys},
            "rating_share": {k: float(rc.get(k, 0) / rc.sum()) for k in keys},
            "largest_group": str(cnt.idxmax()), "smallest_group": str(cnt.idxmin()),
            "imbalance_ratio": float(cnt.max() / cnt.min()),
        }
    return out


def rating_by_group(ratings, users) -> dict:
    eu = eval_users(users)
    d = ratings.merge(eu[["userId"] + C.SENSITIVE_ATTRS], on="userId")
    per_user = d.groupby("userId").size()
    out = {}
    for attr in C.SENSITIVE_ATTRS:
        g = d.groupby(attr)["rating"]
        keys = _order(attr, list(g.groups.keys()))
        upg = eu.set_index("userId")[attr]
        rpu = per_user.groupby(upg.reindex(per_user.index)).agg(["mean", "median"])
        dist = (d.groupby([attr, "rating"]).size().unstack(fill_value=0))
        dist = dist.div(dist.sum(axis=1), axis=0)
        kw = stats.kruskal(*[g.get_group(k).to_numpy() for k in keys])
        n = len(d)
        eps2 = (kw.statistic - len(keys) + 1) / (n - len(keys))   # efek ukuran
        out[attr] = {
            "per_group": {k: {"mean": float(g.get_group(k).mean()),
                              "median": float(g.get_group(k).median()),
                              "std": float(g.get_group(k).std()),
                              "n_ratings": int(g.get_group(k).size),
                              "ratings_per_user_mean": float(rpu.loc[k, "mean"]),
                              "ratings_per_user_median": float(rpu.loc[k, "median"]),
                              "rating_dist": {int(r): float(dist.loc[k, r])
                                              for r in dist.columns}}
                          for k in keys},
            "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
            "epsilon_squared": float(eps2),
            "significant_0_05": bool(kw.pvalue < 0.05),
        }
    return out


def sparsity(ratings, users) -> dict:
    eu = eval_users(users)
    n_items = ratings.movieId.nunique()
    d = ratings.merge(eu[["userId"] + C.SENSITIVE_ATTRS], on="userId")
    out = {"overall": float(1 - len(ratings) / (ratings.userId.nunique() * n_items)),
           "per_attribute": {}}
    for attr in C.SENSITIVE_ATTRS:
        nu = eu[attr].value_counts()
        nr = d[attr].value_counts()
        keys = _order(attr, nu.index.tolist())
        out["per_attribute"][attr] = {
            k: {"n_users": int(nu[k]), "n_ratings": int(nr.get(k, 0)),
                "density": float(nr.get(k, 0) / (nu[k] * n_items)),
                "sparsity": float(1 - nr.get(k, 0) / (nu[k] * n_items))}
            for k in keys}
    return out


def item_popularity(ratings, head_frac: float = 0.2) -> dict:
    cnt = ratings.groupby("movieId").size().sort_values(ascending=False)
    n = len(cnt)
    head_n = max(1, int(round(head_frac * n)))
    cum = cnt.cumsum() / cnt.sum()
    return {
        "n_items": int(n),
        "counts_sorted": cnt.to_numpy().tolist(),
        "gini": _gini(cnt.to_numpy()),
        "share_top_1pct": float(cnt.iloc[:max(1, n // 100)].sum() / cnt.sum()),
        "share_top_10pct": float(cnt.iloc[:max(1, n // 10)].sum() / cnt.sum()),
        "head_frac": head_frac, "head_items": int(head_n),
        "share_head": float(cnt.iloc[:head_n].sum() / cnt.sum()),
        "n_items_for_50pct_ratings": int((cum < 0.5).sum() + 1),
        "median_ratings_per_item": float(cnt.median()),
        "n_items_lt_10_ratings": int((cnt < 10).sum()),
        "head_item_ids": cnt.index[:head_n].tolist(),
    }


def popularity_by_group(ratings, users, head_ids) -> dict:
    """Proporsi rating tiap kelompok yang jatuh pada film populer (head)."""
    eu = eval_users(users)
    d = ratings.merge(eu[["userId"] + C.SENSITIVE_ATTRS], on="userId")
    d["is_head"] = d.movieId.isin(set(head_ids))
    out = {}
    for attr in C.SENSITIVE_ATTRS:
        s = d.groupby(attr)["is_head"].mean()
        out[attr] = {k: float(s[k]) for k in _order(attr, s.index.tolist())}
    return out


def genre_preference(ratings, users, movies) -> dict:
    eu = eval_users(users)
    liked = ratings[ratings.rating >= C.LIKE_THRESHOLD]
    g = movies.set_index("movieId").genres.str.split("|")
    rows = liked.merge(eu[["userId", "gender", "age_group"]], on="userId")
    rows = rows.assign(genre=rows.movieId.map(g)).explode("genre")
    out = {}
    for attr in ["gender", "age_group"]:
        t = rows.groupby([attr, "genre"]).size().unstack(fill_value=0)
        t = t.div(t.sum(axis=1), axis=0)
        out[attr] = {k: {gg: float(v) for gg, v in t.loc[k].items()}
                     for k in _order(attr, t.index.tolist())}
    return out


def run_eda(ratings, users, movies) -> dict:
    pop = item_popularity(ratings)
    return {
        "demographic_distribution": demographic_distribution(ratings, users),
        "rating_by_group": rating_by_group(ratings, users),
        "sparsity": sparsity(ratings, users),
        "item_popularity": {k: v for k, v in pop.items() if k != "head_item_ids"},
        "popularity_by_group": popularity_by_group(ratings, users, pop["head_item_ids"]),
        "genre_preference": genre_preference(ratings, users, movies),
    }
