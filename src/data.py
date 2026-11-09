"""
data.py
=======
PERAN FILE
    Memuat MovieLens 1M, menambah label demografis, dan membentuk seluruh pembagian
    data per seed (Subbab 3.2 dan 3.4 proposal).

SKEMA VALIDASI: 5-FOLD (sesuai proposal dan arahan pembimbing)
    - Rating: rating setiap pengguna diacak lalu dibagi rata ke 5 fold, sehingga setiap
      fold berproporsi 80:20 per pengguna dan setiap rating diuji tepat sekali.
    - Pengguna baru (cold-start): pengguna dipartisi ke 5 fold, distratifikasi gender x
      usia; pada fold f, pengguna di fold f dikeluarkan penuh dari pelatihan jalur
      cold-start. Setiap pengguna menjadi "pengguna baru" tepat sekali.
    - Validasi tuning: 10% rating train setiap pengguna pada fold 0.
    - Kontrol baseline: split lama 80:20 seed 42 (BaselineSplit), hanya untuk
      memeriksa reprodusibilitas terhadap hasil baseline.

KELOMPOK PEKERJAAN
    21 kategori MovieLens dipetakan ke 5 kelompok (config.OCC_GROUP) berdasarkan
    ISCO-08 skill level, status aktivitas menurut resolusi statistik ketenagakerjaan
    ILO (ICLS), dan ICSE-18 untuk status dalam pekerjaan.

ISI UTAMA
    load_movielens, add_demographic_labels, eval_users, merge_all, dataset_info,
    user_stratified_split, fold_assign, FoldSplit, BaselineSplit, split_summary, samples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


def load_movielens(data_dir: Path | str | None = None):
    d = Path(data_dir or C.DATA_DIR)
    for fn in ["ratings.dat", "users.dat", "movies.dat"]:
        if not (d / fn).exists():
            raise FileNotFoundError(f"{fn} tidak ditemukan di {d}. Letakkan ketiga berkas "
                                    "MovieLens 1M di data/ml-1m/ (lihat README).")
    ratings = pd.read_csv(d / "ratings.dat", sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1")
    users = pd.read_csv(d / "users.dat", sep="::", engine="python",
                        names=["userId", "gender", "age", "occupation", "zip"], encoding="latin-1")
    movies = pd.read_csv(d / "movies.dat", sep="::", engine="python",
                         names=["movieId", "title", "genres"], encoding="latin-1")
    return ratings, add_demographic_labels(users), movies


def add_demographic_labels(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()
    u["age_group"] = u["age"].map(C.AGE_LABELS)
    u["occupation_label"] = u["occupation"].map(C.OCCUPATION_LABELS)
    u["occ_group"] = u["occupation_label"].map(C.OCC_GROUP)
    return u


def eval_users(users: pd.DataFrame) -> pd.DataFrame:
    """Tabel pengguna untuk evaluasi: kolom `occupation` berisi label teks."""
    u = users.copy()
    u["occupation"] = u["occupation_label"]
    return u


def merge_all(ratings, users, movies):
    df = ratings.merge(users, on="userId", how="left").merge(movies, on="movieId", how="left")
    quality = {
        "n_rows": int(len(df)),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_user_movie_pairs": int(df.duplicated(["userId", "movieId"]).sum()),
        "ratings_outside_scale": int((~df.rating.between(*C.RATING_SCALE)).sum()),
    }
    return df, quality


def dataset_info(ratings, users, movies) -> dict:
    n_u, n_i = ratings.userId.nunique(), ratings.movieId.nunique()
    per_user = ratings.groupby("userId").size()
    return {
        "n_users": int(n_u), "n_movies_rated": int(n_i), "n_movies_catalog": int(len(movies)),
        "n_ratings": int(len(ratings)), "sparsity": float(1 - len(ratings) / (n_u * n_i)),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_mean": float(per_user.mean()),
        "rating_mean": float(ratings.rating.mean()),
        "n_users_per_gender": users.gender.value_counts().to_dict(),
        "n_users_per_age": users.age_group.value_counts().to_dict(),
        "n_users_per_occupation": users.occupation_label.value_counts().to_dict(),
        "n_users_per_occ_group": users.occ_group.value_counts().to_dict(),
        "occ_group_mapping": C.OCC_GROUP, "occ_group_basis": C.OCC_GROUP_BASIS,
    }


def user_stratified_split(ratings: pd.DataFrame, test_frac: float, seed: int):
    """Split per pengguna; identik dengan baseline untuk seed 42."""
    df = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_positions = []
    for _, g in df.groupby("userId", sort=False):
        n = len(g)
        k = max(1, int(round(n * test_frac))) if n > 1 else 0
        test_positions.extend(g.index[:k])
    mask = df.index.isin(test_positions)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def cold_user_split(users: pd.DataFrame, frac: float, seed: int) -> set:
    cold = (users.groupby(["gender", "age_group"], group_keys=False)
            .sample(frac=frac, random_state=seed))
    return set(cold.userId.tolist())


_FOLD_CACHE = {}


def fold_assign(ratings: pd.DataFrame, users: pd.DataFrame):
    """Indeks fold untuk setiap rating (per pengguna) dan setiap pengguna (stratified)."""
    key = (len(ratings), int(ratings.userId.sum() % 1_000_003))
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    df = ratings.sample(frac=1.0, random_state=C.BASE_SEED).reset_index(drop=True)
    df["fold"] = df.groupby("userId").cumcount() % C.N_FOLDS
    u = users.sample(frac=1.0, random_state=C.BASE_SEED).reset_index(drop=True)
    u["ufold"] = u.groupby(["gender", "age_group"]).cumcount() % C.N_FOLDS
    _FOLD_CACHE[key] = (df, dict(zip(u.userId, u.ufold)))
    return _FOLD_CACHE[key]


class FoldSplit:
    """Semua pembagian data untuk satu fold (atribut `seed` = indeks fold)."""

    def __init__(self, ratings: pd.DataFrame, users: pd.DataFrame, fold: int,
                 with_validation: bool = False):
        self.seed = self.fold = int(fold)
        df, ufold = fold_assign(ratings, users)
        te = df.fold.to_numpy() == fold
        cols = ["userId", "movieId", "rating", "timestamp"]
        self.train_df = df.loc[~te, cols].reset_index(drop=True)
        self.test_df = df.loc[te, cols].reset_index(drop=True)
        self.cold = {u for u, f in ufold.items() if f == fold}
        self.warm_r = ratings[~ratings.userId.isin(self.cold)].reset_index(drop=True)
        self.cold_r = ratings[ratings.userId.isin(self.cold)].reset_index(drop=True)
        if with_validation:
            self.fit_df, self.val_df = user_stratified_split(self.train_df, C.VAL_FRAC,
                                                             C.BASE_SEED + 1)

    def summary(self) -> dict:
        return {"fold": self.fold, "n_train": int(len(self.train_df)),
                "n_test": int(len(self.test_df)), "n_cold_users": int(len(self.cold)),
                "n_cold_ratings": int(len(self.cold_r)), "n_warm_ratings": int(len(self.warm_r))}


SeedSplit = FoldSplit


class BaselineSplit:
    """Split lama 80:20 seed 42 (identik baseline), khusus kontrol reprodusibilitas."""

    def __init__(self, ratings: pd.DataFrame, users: pd.DataFrame, seed: int = 42):
        self.seed = self.fold = 999   # penanda kontrol baseline
        self.train_df, self.test_df = user_stratified_split(ratings, C.TEST_FRAC, seed)
        self.cold = cold_user_split(users, C.COLD_FRAC, seed)
        self.warm_r = ratings[~ratings.userId.isin(self.cold)].reset_index(drop=True)
        self.cold_r = ratings[ratings.userId.isin(self.cold)].reset_index(drop=True)

    def summary(self) -> dict:
        return {"seed": self.seed, "n_train": int(len(self.train_df)),
                "n_test": int(len(self.test_df)), "n_cold_users": int(len(self.cold)),
                "n_cold_ratings": int(len(self.cold_r)), "n_warm_ratings": int(len(self.warm_r))}


def split_summary(train_df, test_df, users) -> dict:
    eu = eval_users(users)
    out = {}
    for attr in C.SENSITIVE_ATTRS:
        tr = train_df.merge(eu[["userId", attr]], on="userId")[attr].value_counts(normalize=True)
        te = test_df.merge(eu[["userId", attr]], on="userId")[attr].value_counts(normalize=True)
        out[attr] = float((tr - te).abs().max())
    return {"max_abs_share_diff_train_vs_test": out}


def samples(ratings, users, movies, n: int = 6) -> dict:
    rs = np.random.RandomState(C.BASE_SEED)
    pick = lambda df: df.iloc[rs.choice(len(df), size=min(n, len(df)), replace=False)]
    return {"ratings": pick(ratings).reset_index(drop=True),
            "users": pick(users).reset_index(drop=True),
            "movies": pick(movies).reset_index(drop=True)}
