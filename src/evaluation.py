"""
evaluation.py
=============
PERAN FILE
    Mengevaluasi SATU sel eksperimen (model x k x alpha x seed x encoding) pada jalur
    warm atau cold-start, lalu mengembalikan:
      - point : metrik titik (satu baris tabel hasil),
      - reps  : replikat bootstrap berpasangan untuk metrik kunci (dipakai uji
                antar konfigurasi tanpa menyimpan prediksi mentah),
      - extra : informasi pendukung (kurva Pickands, faktor laten, probe, ranking).

DUA JALUR
    warm : pengguna lama, split 80:20 per seed, representasi riwayat + pseudo-item.
    cold : 20% pengguna hold-out penuh; direpresentasikan hanya dari profil
           (8 kategori: Default dan semua kombinasi gender, usia, pekerjaan) lewat fold-in. Encoding utama saja.

METRIK UTAMA (Lubang 3 dan 5 protokol)
    gap_gender  = |MAE F - MAE M|
    gap_age_ref = maks_g (MAE_g - MAE 25-34)
    dxi_gender  = |xi_F - xi_M|, Pickands stabil pada eksposur Top-100
    relx_*      = excess gap relatif = (gap teramati - rata-rata gap permutasi) / MAE

PERUBAHAN v3.3
    - Bootstrap B = 9999 untuk metrik berbasis MAE dan amplifikasi; EVT memakai 999 baris
      pertama dari matriks bobot yang sama (tetap berpasangan).
    - sub_* : metrik jalur warm yang dihitung hanya pada pengguna fold cold-start dengan
      bobot bootstrap jalur cold, sehingga uji pengguna baru (F3) berpasangan pada pengguna
      yang sama.
    - Error per pengguna disimpan (extra.per_user) untuk analisis perancu aktivitas.
    - baseline_cell : prediktor BaselineOnly (Koren, 2010) per fold sebagai pembanding gap.
    - Encoding hanya-anggota juga dievaluasi pada jalur cold.
    - Penanda divergensi: MAE latih > MAE prediktor rata-rata global pada sampel yang sama.
    - Hill tidak lagi dibootstrap (hanya sah untuk xi > 0); Moment menjadi pembanding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src import config as C
from src import data as D
from src import evt as E
from src import features as F
from src import latent as L
from src import models as MD
from src import stats_tools as S


# ==========================================================================
# Konteks per seed (dihitung sekali)
# ==========================================================================
def genre_matrix(movies: pd.DataFrame):
    g = movies.set_index("movieId")["genres"].str.split("|")
    genres = sorted({x for lst in g for x in lst})
    col = {x: i for i, x in enumerate(genres)}
    mat = np.zeros((len(g), len(genres)))
    for r, lst in enumerate(g):
        for x in lst:
            mat[r, col[x]] = 1.0 / len(lst)
    return genres, pd.DataFrame(mat, index=g.index, columns=genres)


class SeedContext:
    """Konteks satu fold (atribut `seed` = indeks fold). baseline=True memakai split lama
    80:20 seed 42 untuk kontrol reprodusibilitas."""

    def __init__(self, ratings, users_raw, movies, seed: int, with_validation=False,
                 baseline: bool = False):
        self.seed = seed
        if baseline:
            self.split = D.BaselineSplit(ratings, users_raw)
            self.model_seed = C.BASE_SEED
        else:
            self.split = D.FoldSplit(ratings, users_raw, seed, with_validation)
            self.model_seed = C.BASE_SEED + int(seed)
        self.users = D.eval_users(users_raw)
        self.genres, self.gmat = genre_matrix(movies)
        self.titles = movies.set_index("movieId")[["title", "genres"]]
        ulab = self.users.set_index("userId")

        # ---- warm
        tr, te = self.split.train_df, self.split.test_df
        self.warm_units = np.sort(te.userId.unique())
        self.wlab = ulab.loc[self.warm_units]
        self.W_warm = S.boot_weights(len(self.warm_units), seed, "warm")
        self.W_warm_evt = self.W_warm[:C.N_BOOT_EVT]
        liked_tr = tr[tr.rating >= C.LIKE_THRESHOLD]
        self.hist_warm = self._user_genre(liked_tr, self.warm_units)
        self.rel_warm = (te[te.rating >= C.LIKE_THRESHOLD].groupby("userId")["movieId"]
                         .apply(set).to_dict())

        # ---- cold
        cr = self.split.cold_r
        self.cold_units = np.sort(cr.userId.unique())
        self.clab = ulab.loc[self.cold_units]
        self.W_cold = S.boot_weights(len(self.cold_units), seed, "cold")
        self.W_cold_evt = self.W_cold[:C.N_BOOT_EVT]
        # nilai pseudo-item membermean untuk pengguna baru = rata-rata dari rata-rata rating
        # pengguna latih (nilai harapan fitur itu di populasi latih)
        self.prior_user_mean = float(self.split.warm_r.groupby("userId").rating.mean().mean())
        liked_c = cr[cr.rating >= C.LIKE_THRESHOLD]
        self.hist_cold = self._user_genre(liked_c, self.cold_units)
        self.rel_cold = liked_c.groupby("userId")["movieId"].apply(set).to_dict()
        self.profiles = sorted(set(zip(self.clab.gender, self.clab.age_group, self.clab.occ_group)),
                               key=lambda t: (t[0], C.AGE_ORDER.index(t[1]),
                                              C.OCC_GROUP_ORDER.index(t[2])))

    def _user_genre(self, liked, units):
        gw = self.gmat.reindex(liked.movieId.to_numpy()).fillna(0.0).to_numpy()
        df = pd.DataFrame(gw, columns=self.genres)
        df["userId"] = liked.userId.to_numpy()
        return df.groupby("userId").sum().reindex(units).fillna(0.0).to_numpy()

    def item_genre(self, trainset) -> np.ndarray:
        out = np.zeros((trainset.n_items, len(self.genres)))
        for raw, inner in trainset._raw2inner_id_items.items():
            if not F.is_pseudo(raw) and raw in self.gmat.index:
                out[inner] = self.gmat.loc[raw].to_numpy()
        return out


# ==========================================================================
# Blok metrik
# ==========================================================================
def _per_user(df: pd.DataFrame, est: np.ndarray, units: np.ndarray):
    t = pd.DataFrame({"userId": df.userId.to_numpy(), "ae": np.abs(df.rating.to_numpy() - est),
                      "se": (df.rating.to_numpy() - est) ** 2})
    g = t.groupby("userId").agg(ae=("ae", "sum"), se=("se", "sum"), n=("ae", "size"))
    g = g.reindex(units).fillna(0.0)
    return g.ae.to_numpy(), g.se.to_numpy(), g.n.to_numpy()


def _gaps(gm: dict) -> tuple:
    """gm: {attr: {group: nilai atau array replikat}}."""
    gg = gm["gender"]
    gap_g = np.abs(gg["F"] - gg["M"])
    ref = gm["age_group"][C.AGE_REFERENCE]
    others = [v for k, v in gm["age_group"].items() if k != C.AGE_REFERENCE]
    gap_a = np.max(np.vstack([np.atleast_1d(v - ref) for v in others]), axis=0)
    occ = np.vstack([np.atleast_1d(v) for k, v in gm["occ_group"].items()
                     if k != C.OCC_UNCLASSIFIED])
    gap_o = np.nanmax(occ, 0) - np.nanmin(occ, 0)
    return gap_g, gap_a, gap_o


def accuracy_block(ae, se, n, lab: pd.DataFrame, W, seed: int, path: str):
    point, reps = {}, {}
    ones = n > 0
    point["mae"] = float(ae.sum() / n.sum())
    point["rmse"] = float(np.sqrt(se.sum() / n.sum()))
    reps["mae"] = S.group_ratio(W, ae, n, ones)
    gm_p, gm_r = {}, {}
    for attr in C.SENSITIVE_ATTRS:
        a = lab[attr].astype(str).to_numpy()
        gm_p[attr], gm_r[attr] = {}, {}
        for g in pd.unique(a):
            m = (a == g) & ones
            if m.sum() == 0:
                continue
            gm_p[attr][g] = float(ae[m].sum() / n[m].sum())
            gm_r[attr][g] = S.group_ratio(W, ae, n, m)
    gp, ap, op = (float(np.ravel(x)[0]) for x in _gaps(gm_p))
    gr, ar, orr = _gaps(gm_r)
    point.update(gap_gender=float(gp), gap_gender_signed=gm_p["gender"]["F"] - gm_p["gender"]["M"],
                 gap_age_ref=float(ap), gap_occ=float(op),
                 worst_age=max((k for k in gm_p["age_group"] if k != C.AGE_REFERENCE),
                               key=lambda k: gm_p["age_group"][k]))
    for attr, short in [("gender", "g"), ("age_group", "u")]:
        for gname, v in gm_p[attr].items():
            point[f"mae_{short}_{gname}"] = v
    point["mae_occ"] = gm_p["occ_group"]
    for gname, v in gm_p["occ_group"].items():
        point[f"mae_o_{gname}"] = v
    reps.update(gap_gender=gr, gap_age_ref=ar, gap_occ=orr)

    # ---- uji permutasi (excess gap), label pengguna diacak antar kelompok
    rng = np.random.default_rng(200_000 + 10 * seed + S.PATH_CODE[path])
    U = len(ae)
    perm = np.argsort(rng.random((C.N_PERM, U)), axis=1)
    for attr, key in [("gender", "gender"), ("age_group", "age"), ("occ_group", "occ")]:
        codes, cats = pd.factorize(lab[attr].astype(str).to_numpy())
        pc = codes[perm]
        gm_null = {}
        for ci_, g in enumerate(cats):
            M = ((pc == ci_) & ones[None, :]).astype(np.float32)
            gm_null[g] = (M @ ae) / np.maximum(M @ n, 1)
        if attr == "gender":
            null = np.abs(gm_null["F"] - gm_null["M"])
            obs, rep = gp, gr
        elif attr == "occ_group":
            V = np.vstack([v for k, v in gm_null.items() if k != C.OCC_UNCLASSIFIED])
            null = V.max(0) - V.min(0)
            obs, rep = op, orr
        else:
            ref = gm_null[C.AGE_REFERENCE]
            null = np.max(np.vstack([v - ref for k, v in gm_null.items()
                                     if k != C.AGE_REFERENCE]), axis=0)
            obs, rep = ap, ar
        nm = float(null.mean())
        point[f"null_{key}"] = nm
        point[f"p_perm_{key}"] = float((np.sum(null >= obs) + 1) / (len(null) + 1))
        point[f"excess_{key}"] = float(obs - nm)
        point[f"relx_{key}"] = float((obs - nm) / point["mae"])
        reps[f"relx_{key}"] = (rep - nm) / reps["mae"]
    return point, reps


def dp_block(est: np.ndarray, df: pd.DataFrame, users: pd.DataFrame) -> dict:
    lab = users.set_index("userId").loc[df.userId.to_numpy()]
    pos = est >= C.DP_THRESHOLD
    out = {}
    for attr, short in [("gender", "g"), ("age_group", "u"), ("occ_group", "o")]:
        r = pd.Series(pos).groupby(lab[attr].astype(str).to_numpy()).mean()
        out[f"dp_{short}"] = float(r.max() - r.min())
    return out


def _incidence(lists: np.ndarray, n_items: int):
    U, N = lists.shape
    return sparse.csr_matrix((np.ones(U * N, np.float32), (np.repeat(np.arange(U), N),
                                                          lists.ravel())), shape=(U, n_items))


def exposure_block(lists, lab, W, n_items, with_boot: bool, tag: str = ""):
    """Indeks ekor eksposur per gender (dengan bootstrap) dan per usia (titik)."""
    Lm = _incidence(lists, n_items)
    point, reps, extra = {}, {}, {}
    g = lab["gender"].astype(str).to_numpy()
    est = {}
    for grp in C.GENDER_ORDER:
        m = g == grp
        if m.sum() == 0:
            continue
        e = np.asarray(Lm[m].sum(0)).ravel()
        st = E.stable(e)
        est[grp] = st
        point[f"xi{tag}_{grp}"] = st["pickands"]
        point[f"hill{tag}_{grp}"] = st["hill"]
        point[f"moment{tag}_{grp}"] = st["moment"]
        point[f"xi{tag}_estimable_{grp}"] = st["estimable"]
        point[f"xi{tag}_frac_dropped_{grp}"] = st["frac_dropped"]
        point[f"n_items{tag}_{grp}"] = st["n"]
        if with_boot:
            Eb = (Lm[m].T @ W[:, m].T).T
            reps[f"evt_{grp}"] = E.stable_rows(np.asarray(Eb))
        if not tag:
            extra[f"curve_{grp}"] = E.curves(e)
    if "F" in est and "M" in est:
        point[f"dxi{tag}_gender"] = float(abs(est["F"]["pickands"] - est["M"]["pickands"])) \
            if est["F"]["estimable"] and est["M"]["estimable"] else np.nan
        point[f"dhill{tag}_gender"] = float(abs(est["F"]["hill"] - est["M"]["hill"]))
        point[f"dmoment{tag}_gender"] = float(abs(est["F"]["moment"] - est["M"]["moment"]))
        if with_boot:
            rF, rM = reps.pop("evt_F"), reps.pop("evt_M")
            reps[f"dxi{tag}_gender"] = np.abs(rF[:, 0] - rM[:, 0])
            reps[f"dmoment{tag}_gender"] = np.abs(rF[:, 2] - rM[:, 2])
            reps[f"xi{tag}_F"], reps[f"xi{tag}_M"] = rF[:, 0], rM[:, 0]
            reps[f"moment{tag}_F"], reps[f"moment{tag}_M"] = rF[:, 2], rM[:, 2]
    if not tag:
        # jumlah eksposur terurut disimpan agar EVT dapat dihitung ulang di analisis
        # (uji sensitivitas rentang k) tanpa melatih ulang model
        counts = {}
        for attr, short, order in [("gender", "g", C.GENDER_ORDER), ("age_group", "u", C.AGE_ORDER),
                                   ("occ_group", "o", C.OCC_GROUP_ORDER)]:
            a = lab[attr].astype(str).to_numpy()
            counts[attr] = {}
            for grp in order:
                m = a == grp
                if not m.sum():
                    continue
                e = np.asarray(Lm[m].sum(0)).ravel()
                nz = np.sort(e[e > 0])[::-1].astype(np.int32)
                counts[attr][grp] = nz
                if attr != "gender":
                    st = E.stable(nz)
                    point[f"xi_{short}_{grp}"] = st["pickands"]
                    point[f"moment_{short}_{grp}"] = st["moment"]
                    point[f"xi_{short}_estimable_{grp}"] = st["estimable"]
        extra["exposure_counts"] = counts
        if "F" in est and "M" in est:
            point["moment_negative_F"] = bool(est["F"]["moment"] < 0) if np.isfinite(est["F"]["moment"]) else None
            point["moment_negative_M"] = bool(est["M"]["moment"] < 0) if np.isfinite(est["M"]["moment"]) else None
    return point, reps, extra


def amplification_block(hist_raw, rec_raw, lab, W, with_boot=True):
    g = lab["gender"].astype(str).to_numpy()
    mF, mM = g == "F", g == "M"
    h32, r32 = np.asarray(hist_raw, np.float32), np.asarray(rec_raw, np.float32)
    def dists(Wm):
        HF = S.normalize_rows((Wm @ (h32 * mF[:, None])).astype(np.float64))
        HM = S.normalize_rows((Wm @ (h32 * mM[:, None])).astype(np.float64))
        RF = S.normalize_rows((Wm @ (r32 * mF[:, None])).astype(np.float64))
        RM = S.normalize_rows((Wm @ (r32 * mM[:, None])).astype(np.float64))
        return S.js2_rows(HF, HM), S.js2_rows(RF, RM)
    jh, jr = dists(np.ones((1, len(g)), np.float32))
    point = {"js_hist_gender": float(jh[0]), "js_rec_gender": float(jr[0]),
             "amp_gender": float(jr[0] - jh[0])}
    reps = {}
    if with_boot:
        jh_b, jr_b = dists(W)
        reps["amp_gender"] = jr_b - jh_b
    return point, reps


def calibration_block(hist_raw, rec_raw, lab, W):
    """Miscalibration genre per pengguna (KL ala Steck) dan selisih antar kelompok."""
    p = S.normalize_rows(hist_raw)
    q = S.normalize_rows(rec_raw)
    qt = (1 - C.STECK_ALPHA) * q + C.STECK_ALPHA * p
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * np.log(p / qt), 0.0).sum(1)
    has = hist_raw.sum(1) > 0
    one = has.astype(float)
    point, reps = {"misc_mean": float(kl[has].mean()) if has.any() else np.nan}, {}
    for attr, short in [("gender", "g"), ("age_group", "u")]:
        a = lab[attr].astype(str).to_numpy()
        gp, gr = {}, {}
        for grp in pd.unique(a):
            m = (a == grp) & has
            if m.sum():
                gp[grp] = float(kl[m].mean())
                gr[grp] = S.group_ratio(W, kl * one, one, m)
        v = np.array(list(gp.values()))
        point[f"misc_gap_{short}"] = float(v.max() - v.min())
        R = np.vstack(list(gr.values()))
        reps[f"misc_gap_{short}"] = np.nanmax(R, 0) - np.nanmin(R, 0)
        for k, val in gp.items():
            point[f"misc_{short}_{k}"] = val
    return point, reps


def precision_block(lists_raw: np.ndarray, units, relevant: dict, lab, prefix="p10"):
    n = lists_raw.shape[1]
    vals = np.full(len(units), np.nan)
    for r, u in enumerate(units):
        rel = relevant.get(int(u))
        if rel:
            vals[r] = len(set(lists_raw[r].tolist()) & rel) / n
    g = lab["gender"].astype(str).to_numpy()
    out = {prefix: float(np.nanmean(vals))}
    for grp in C.GENDER_ORDER:
        out[f"{prefix}_{grp}"] = float(np.nanmean(vals[g == grp])) if (g == grp).any() else np.nan
    return out


# ==========================================================================
# Sel warm
# ==========================================================================
def warm_cell(ctx: SeedContext, model: str, k: int, alpha: int, hp: dict,
              encoding: str = "full", biased=None, run_probe: bool = False,
              attrs_override=None) -> dict:
    attrs = list(attrs_override or C.PROFILE_ATTRS) if alpha > 0 else []
    tr, te = ctx.split.train_df, ctx.split.test_df
    ts = F.build_trainset(tr, ctx.users, attrs, alpha, encoding)
    params = MD.build_params(model, k, hp, ctx.model_seed, biased)
    algo = MD.train_model(model, ts, params)

    est = MD.predict_ratings(algo, ts, te.userId.tolist(), te.movieId.tolist())
    ae, se, n = _per_user(te, est, ctx.warm_units)
    point, reps = accuracy_block(ae, se, n, ctx.wlab, ctx.W_warm, ctx.seed, "warm")
    point.update(dp_block(est, te, ctx.users))
    # metrik warm pada pengguna fold cold-start, bobot bootstrap jalur cold (F3 berpasangan)
    ae_s, se_s, n_s = _per_user(te, est, ctx.cold_units)
    p_s, r_s = accuracy_block(ae_s, se_s, n_s, ctx.clab, ctx.W_cold, ctx.seed, "cold")
    point.update({f"sub_{k_}": v for k_, v in p_s.items()
                  if not isinstance(v, dict) and k_ != "worst_age"})
    reps.update({f"sub_{k_}": v for k_, v in r_s.items()})

    rows = np.array([ts._raw2inner_id_users[u] for u in ctx.warm_units])
    Sm = MD.score_matrix(algo, ts)[rows]
    all_seen = MD.seen_lists(ts)
    seen = [all_seen[r] for r in rows]
    raw_items = np.array([ts.to_raw_iid(i) if not F.is_pseudo(ts.to_raw_iid(i)) else -1
                          for i in range(ts.n_items)], dtype=object)
    elig = MD.eligible_items(ts, tr)
    top100 = MD.topn_from_scores(Sm, elig, C.TOP_N_MAIN, seen)
    top10 = top100[:, :C.TOP_N_SMALL]
    top10_nt = MD.topn_from_scores(Sm, MD.movie_items(ts), C.TOP_N_SMALL, seen)
    del Sm

    point.update(precision_block(raw_items[top10], ctx.warm_units, ctx.rel_warm, ctx.wlab))
    point.update(precision_block(raw_items[top10_nt], ctx.warm_units, ctx.rel_warm, ctx.wlab,
                                 "p10_nothr"))
    p_e, r_e, x_e = exposure_block(top100, ctx.wlab, ctx.W_warm_evt, ts.n_items, True)
    point.update(p_e); reps.update(r_e)
    p10e, _, _ = exposure_block(top10, ctx.wlab, None, ts.n_items, False, "_top10")
    p10n, _, _ = exposure_block(top10_nt, ctx.wlab, None, ts.n_items, False, "_top10nothr")
    point.update(p10e); point.update(p10n)

    ig = ctx.item_genre(ts)
    rec_raw = ig[top10].sum(1)
    p_a, r_a = amplification_block(ctx.hist_warm, rec_raw, ctx.wlab, ctx.W_warm)
    point.update(p_a); reps.update(r_a)

    lat = L.inspect(algo, ts, ctx.users, ctx.seed, run_probe)
    point["cos_F_M"] = lat["pseudo_factors"].get("cos_F_M", np.nan)
    point["frac_negative_pu"] = lat["group_stats"]["overall"]["frac_negative"]
    point["train_seconds"] = float(algo._train_seconds)
    point["train_mae_sample"], point["train_mae_global_mean"] = MD.train_mae(algo, ts, tr, seed=ctx.seed)
    point["diverged"] = bool(point["train_mae_sample"] > point["train_mae_global_mean"])
    point["n_pseudo_entries"] = int(ts.n_ratings - len(tr))
    point.update({f"w_{k_}": v for k_, v in
                  F.relative_weight(tr, ctx.users, attrs, alpha, encoding).items()})
    extra = {"latent": lat, "curves": x_e, "params": params,
             "pred_samples": _pred_samples(te, est, ctx),
             "per_user": {"ae": ae.astype(np.float32), "n": n.astype(np.int32)}}
    return {"point": point, "reps": {k_: np.asarray(v, np.float32) for k_, v in reps.items()},
            "extra": extra}


def _pred_samples(df, est, ctx):
    rs = np.random.RandomState(ctx.seed)
    idx = rs.choice(len(df), min(C.N_PRED_SAMPLES, len(df)), replace=False)
    s = df.iloc[idx][["userId", "movieId", "rating"]].copy()
    s["est"] = np.round(est[idx], 4)
    s["title"] = s.movieId.map(ctx.titles.title)
    return s.to_dict(orient="records")


# ==========================================================================
# Sel cold-start
# ==========================================================================
def profile_obs(profile: dict, attrs, users, pseudo: dict, encoding: str = "full",
                member_value: float | None = None):
    """Pseudo-rating profil pengguna baru, mengikuti encoding yang dipakai saat pelatihan:
    full = semua pseudo-item (anggota 5, bukan anggota 1); member5 = hanya kelompok sendiri
    bernilai 5; membermean = hanya kelompok sendiri bernilai member_value."""
    obs = []
    for attr in attrs:
        for v in F.attr_values(users, attr):
            key = F.pseudo_id(attr, v)
            if key not in pseudo:
                continue
            member = v == profile[attr]
            if encoding == "full":
                obs.append((pseudo[key], C.DEMO_HIGH if member else C.DEMO_LOW))
            elif member:
                obs.append((pseudo[key], C.DEMO_HIGH if encoding == "member5" else member_value))
    return obs


def cold_cell(ctx: SeedContext, model: str, k: int, alpha: int, hp: dict,
              encoding: str = "full") -> dict:
    attrs_train = C.PROFILE_ATTRS if alpha > 0 else []
    wr, cr = ctx.split.warm_r, ctx.split.cold_r
    ts = F.build_trainset(wr, ctx.users, attrs_train, alpha, encoding)
    params = MD.build_params(model, k, hp, ctx.model_seed)
    algo = MD.train_model(model, ts, params)
    tm, tmu = MD.train_mae(algo, ts, wr, seed=ctx.seed)
    P = MD.extract_params(algo, ts, params)
    _, pseudo = F.item_masks(ts)
    movie_mask = MD.movie_items(ts)
    elig = MD.eligible_items(ts, wr)
    all_idx = np.arange(ts.n_items)
    raw_items = np.array([ts.to_raw_iid(i) if movie_mask[i] else -1 for i in all_idx], dtype=object)
    ig = ctx.item_genre(ts)

    prof_row = {p: i for i, p in enumerate(ctx.profiles)}
    urow = np.array([prof_row[p] for p in zip(ctx.clab.gender, ctx.clab.age_group,
                                               ctx.clab.occ_group)])
    unit_row = {u: i for i, u in enumerate(ctx.cold_units)}
    r_user = cr.userId.map(unit_row).to_numpy()
    r_item = np.array([ts._raw2inner_id_items.get(m, -1) for m in cr.movieId])
    known = r_item >= 0

    scenarios = {"default": []} if alpha == 0 else dict(C.SCENARIOS)
    out = {"scenarios": {}, "meta": {"params": params, "train_seconds": float(algo._train_seconds),
                                     "n_train_entries": int(ts.n_ratings), "train_mae_sample": tm,
                                     "train_mae_global_mean": tmu, "diverged": bool(tm > tmu),
                                     "encoding": encoding}}
    keep_all = {"default", C.COLD_PRIMARY_SCENARIO}
    keep_some = {"mae", "gap_gender", "gap_age_ref", "gap_occ"}
    for sc, attrs in scenarios.items():
        Sp = np.vstack([MD.profile_scores(P, *MD.fold_in(
            P, profile_obs({"gender": g, "age_group": a, "occ_group": o}, attrs, ctx.users,
                           pseudo, encoding, ctx.prior_user_mean), alpha), all_idx)
            for g, a, o in ctx.profiles])
        est = np.full(len(cr), P["mu"])
        est[known] = Sp[urow[r_user[known]], r_item[known]]
        est = np.clip(est, *C.RATING_SCALE)
        ae, se, n = _per_user(cr, est, ctx.cold_units)
        point, reps = accuracy_block(ae, se, n, ctx.clab, ctx.W_cold, ctx.seed, "cold")
        point.update(dp_block(est, cr, ctx.users))
        top100_p = MD.topn_from_scores(Sp, elig, C.TOP_N_MAIN)
        top100 = top100_p[urow]
        top10 = top100[:, :C.TOP_N_SMALL]
        top10_nt = MD.topn_from_scores(Sp, movie_mask, C.TOP_N_SMALL)[urow]
        point.update(precision_block(raw_items[top10], ctx.cold_units, ctx.rel_cold, ctx.clab))
        point.update(precision_block(raw_items[top10_nt], ctx.cold_units, ctx.rel_cold, ctx.clab,
                                     "p10_nothr"))
        p_e, r_e, x_e = exposure_block(top100, ctx.clab, ctx.W_cold_evt, ts.n_items,
                                       sc in keep_all)
        point.update(p_e); reps.update(r_e)
        rec_raw = ig[top10].sum(1)
        p_a, _ = amplification_block(ctx.hist_cold, rec_raw, ctx.clab, ctx.W_cold, False)
        point.update({f"{k_}_descriptive": v for k_, v in p_a.items()})
        p_c, r_c = calibration_block(ctx.hist_cold, rec_raw, ctx.clab, ctx.W_cold)
        point.update(p_c); reps.update(r_c)
        lists = {p: set(top100_p[i, :C.TOP_N_SMALL].tolist()) for p, i in prof_row.items()}
        jac = [len(lists[("F", a, o)] & lists[("M", a, o)]) /
               len(lists[("F", a, o)] | lists[("M", a, o)])
               for a in C.AGE_ORDER for o in C.OCC_GROUP_ORDER
               if ("F", a, o) in lists and ("M", a, o) in lists]
        point["jaccard_FvsM_top10"] = float(np.mean(jac)) if jac else np.nan
        point["n_distinct_lists"] = int(len({tuple(r) for r in top100_p[:, :C.TOP_N_SMALL]}))
        rankings = {f"{g}|{a}|{o}": [int(raw_items[c]) for c in top100_p[i, :C.TOP_N_SMALL]]
                    for (g, a, o), i in prof_row.items()}
        point["diverged"] = bool(tm > tmu)
        if sc not in keep_all:   # skenario deskriptif: replikat metrik utama saja (hemat disk)
            reps = {k_: v for k_, v in reps.items() if k_ in keep_some}
        out["scenarios"][sc] = {"point": point,
                                "reps": {k_: np.asarray(v, np.float32) for k_, v in reps.items()},
                                "extra": {"rankings": rankings, "curves": x_e,
                                          "per_user": {"ae": ae.astype(np.float32),
                                                       "n": n.astype(np.int32)}}}
    return out


# ==========================================================================
# Prediktor pembanding (BaselineOnly) per fold
# ==========================================================================
def baseline_cell(ctx: SeedContext) -> dict:
    """Prediktor bias saja r = mu + b_u + b_i (BaselineOnly Surprise, ALS; Koren, 2010).
    Tidak memiliki faktor laten, sehingga gap kelompoknya mencerminkan perbedaan yang sudah
    ada pada pola rating (variasi rating, jumlah rating) dan bukan hasil faktorisasi.
    warm : dilatih pada train fold, diuji pada test fold (semua pengguna, dan sub-kelompok
           pengguna fold cold-start dengan bobot jalur cold).
    cold : dilatih pada data pengguna lama; pengguna baru tidak punya b_u sehingga prediksi
           = mu + b_i (setara skenario Default)."""
    from surprise import BaselineOnly
    tr, te = ctx.split.train_df, ctx.split.test_df
    out = {"key": {"path": "baseline", "seed": ctx.seed}}
    ts = F.build_trainset(tr, ctx.users, [], 0)
    algo = BaselineOnly(verbose=False)
    algo.fit(ts)
    ru, ri = ts._raw2inner_id_users, ts._raw2inner_id_items
    bu, bi, mu = np.asarray(algo.bu), np.asarray(algo.bi), ts.global_mean
    u = np.array([ru.get(x, -1) for x in te.userId]); i = np.array([ri.get(x, -1) for x in te.movieId])
    est = np.full(len(te), mu)
    est[u >= 0] += bu[u[u >= 0]]; est[i >= 0] += bi[i[i >= 0]]
    est = np.clip(est, *C.RATING_SCALE)
    ae, se, n = _per_user(te, est, ctx.warm_units)
    pw, rw = accuracy_block(ae, se, n, ctx.wlab, ctx.W_warm, ctx.seed, "warm")
    ae_s, se_s, n_s = _per_user(te, est, ctx.cold_units)
    ps, rs = accuracy_block(ae_s, se_s, n_s, ctx.clab, ctx.W_cold, ctx.seed, "cold")
    pw.update({f"sub_{k_}": v for k_, v in ps.items() if not isinstance(v, dict) and k_ != "worst_age"})
    rw.update({f"sub_{k_}": v for k_, v in rs.items()})
    n_train = tr.groupby("userId").size().reindex(ctx.warm_units).fillna(0).to_numpy(np.int32)
    sd_train = tr.groupby("userId").rating.std().reindex(ctx.warm_units).to_numpy(np.float32)
    out["warm"] = {"point": pw, "reps": {k_: np.asarray(v, np.float32) for k_, v in rw.items()},
                   "extra": {"per_user": {"ae": ae.astype(np.float32), "n": n.astype(np.int32),
                                          "n_train": n_train, "rating_sd_train": sd_train,
                                          "units": ctx.warm_units.astype(np.int32)}}}
    # cold: mu + b_i dari data pengguna lama
    wr, cr = ctx.split.warm_r, ctx.split.cold_r
    ts2 = F.build_trainset(wr, ctx.users, [], 0)
    algo2 = BaselineOnly(verbose=False)
    algo2.fit(ts2)
    ri2 = ts2._raw2inner_id_items
    i2 = np.array([ri2.get(x, -1) for x in cr.movieId])
    est2 = np.full(len(cr), ts2.global_mean)
    est2[i2 >= 0] += np.asarray(algo2.bi)[i2[i2 >= 0]]
    est2 = np.clip(est2, *C.RATING_SCALE)
    ae2, se2, n2 = _per_user(cr, est2, ctx.cold_units)
    pc, rc = accuracy_block(ae2, se2, n2, ctx.clab, ctx.W_cold, ctx.seed, "cold")
    out["cold"] = {"point": pc, "reps": {k_: np.asarray(v, np.float32) for k_, v in rc.items()},
                   "extra": {"per_user": {"ae": ae2.astype(np.float32), "n": n2.astype(np.int32),
                                          "units": ctx.cold_units.astype(np.int32)}}}
    return out
