"""
analysis.py
===========
PERAN FILE
    Membaca seluruh sel di cache lalu menerapkan SEMUA aturan keputusan protokol.
    Tidak ada pelatihan model di sini, sehingga analisis dapat diulang kapan saja.

ISI UTAMA (kunci JSON di bawah "analysis")
    table          : satu baris per sel (titik metrik)
    aggregate      : rata-rata dan simpangan baku antar seed per konfigurasi
    pareto         : front Pareto per jalur dan sumbu, dominasi kokoh (>= 4 dari 5 fold)
    h1             : keluarga F1 dan putusan Hipotesis Pertama (+ analisis sekunder)
    alpha_effect   : keluarga F2 (m(alpha) - m(0))
    safe_alpha     : bobot aman (non-inferioritas, margin 10%)
    new_users      : keluarga F3 (excess gap relatif cold vs warm, pengguna yang sama)
    new_user_profile: keluarga F3b (profil demografi vs Default pada pengguna baru)
    encoding_effect: keluarga F5b (encoding hanya-anggota vs full)
    confound       : keluarga F8 (gap model dikurangi gap BaselineOnly) + strata aktivitas
    *_evt          : keluarga uji indeks ekor (dipisah dari metrik berbasis MAE, v3.3)
    amplification  : keluarga F4 (A(alpha) - A(0), jalur warm)
    mechanism      : keluarga F5 (cos F vs M per encoding, ablation NMF dengan bias)
    h2             : keluarga F6 (Spearman akurasi vs disparitas di dalam algoritma)
    leakage        : probe pekerjaan laten vs baseline probe
    evt            : keterestimasian, kesesuaian Pickands/Hill/Moment, kurva Pickands
    popularity     : sensitivitas ambang popularitas
    deviations     : penyesuaian protokol yang dilakukan saat implementasi
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src import stats_tools as S

MAIN = dict(encoding="full", variant=None)


# ==========================================================================
# Indeks sel
# ==========================================================================
def add_relative(point: dict, reps: dict) -> None:
    """v3.4: gap relatif = gap / MAE (titik dan replikat), juga untuk metrik sub_*. Gap absolut
    ikut mengecil bila error keseluruhan turun, sehingga perbandingan antar prediktor dengan
    tingkat MAE berbeda (model vs BaselineOnly, profil vs Default) dibaca juga dalam skala relatif."""
    for pre in ["", "sub_"]:
        for met in C.GAP_METRICS:
            g, m = f"{pre}{met}", f"{pre}mae"
            if point.get(g) is not None and point.get(m):
                point[f"{pre}r{met}"] = point[g] / point[m]
            if reps is not None and g in reps and m in reps:
                with np.errstate(divide="ignore", invalid="ignore"):
                    reps[f"{pre}r{met}"] = (np.asarray(reps[g], float) /
                                            np.asarray(reps[m], float)).astype(np.float32)


class Cells:
    """Indeks seluruh sel. Kunci alpha = 0 selalu dinormalkan ke encoding 'full' dan
    variant '-' karena tanpa pseudo-item encoding/variant tidak berpengaruh (v3.3)."""

    def __init__(self, recs: list):
        self.rows, self.reps, self.extra, self.diverged = [], {}, {}, []
        for r in recs:
            k = r["key"]
            if k["path"] == "warm":
                self._add(k, None, r["point"], r["reps"], r["extra"])
            else:
                for sc, v in r["scenarios"].items():
                    self._add(k, sc, v["point"], v["reps"], v["extra"])
        self.df = pd.DataFrame(self.rows)

    def _add(self, k, sc, point, reps, extra):
        key = self.key(k["path"], k["model"], k["k"], k["alpha"], k["seed"], sc or "-",
                       k["encoding"], k["variant"] or "-")
        point = dict(point)
        reps = dict(reps)
        add_relative(point, reps)
        row = dict(path=k["path"], scenario=sc or "-", model=k["model"], k=k["k"],
                   alpha=k["alpha"], seed=k["seed"], encoding=key[6], variant=key[7],
                   seconds=k.get("seconds"))
        row.update({a: b for a, b in point.items() if not isinstance(b, (dict, list))})
        self.rows.append(row)
        if point.get("diverged"):
            # sel divergen dicatat tetapi replikatnya tidak dipakai uji
            self.diverged.append("|".join(map(str, key)))
            reps = {}
        self.reps[key] = reps
        self.extra[key] = extra

    def key(self, path, model, k, alpha, seed, scenario="-", encoding="full", variant="-"):
        if alpha == 0:
            encoding, variant = "full", "-"
        return (path, scenario, model, k, alpha, seed, encoding, variant)

    def rep(self, *a, metric, **kw):
        return self.reps.get(self.key(*a, **kw), {}).get(metric)

    def point(self, path, model, k, alpha, seed, metric, scenario="-", encoding="full",
              variant="-"):
        if alpha == 0:
            encoding, variant = "full", "-"
        d = self.df
        m = ((d.path == path) & (d.model == model) & (d.k == k) & (d.alpha == alpha) &
             (d.seed == seed) & (d.scenario == scenario) & (d.encoding == encoding) &
             (d.variant == variant))
        if not m.any() or metric not in d:
            return None
        v = d.loc[m, metric].iloc[0]
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def _scen(path, alpha):
    if path == "warm":
        return "-"
    return "default" if alpha == 0 else C.COLD_PRIMARY_SCENARIO


def _seeds(cells):
    return sorted(cells.df.seed.unique().tolist())


def _decide(per_seed: dict, n_seeds: int) -> dict:
    """per_seed: {seed: {"p_adj": .., "sign": +1/-1/0}} -> putusan mayoritas seed."""
    pos = sum(1 for v in per_seed.values() if v["p_adj"] is not None and
              v["p_adj"] < C.SIG_LEVEL and v["sign"] > 0)
    neg = sum(1 for v in per_seed.values() if v["p_adj"] is not None and
              v["p_adj"] < C.SIG_LEVEL and v["sign"] < 0)
    need = min(C.SEED_MAJORITY, n_seeds)
    avail = sum(1 for v in per_seed.values() if v["p_adj"] is not None)
    n_ne = sum(1 for v in per_seed.values() if v.get("ne"))
    if avail == 0:
        verdict = "tidak terestimasi" if n_ne else "data belum tersedia"
    elif pos >= need:
        verdict = "naik"
    elif neg >= need:
        verdict = "turun"
    elif avail < need:
        verdict = "data belum lengkap"
    else:
        verdict = "tidak berbeda"
    return {"n_sig_positive": pos, "n_sig_negative": neg, "seeds_required": need,
            "n_seeds_available": avail, "n_seeds_not_estimable": n_ne, "verdict": verdict}


MISSING = ("data belum tersedia", "data belum lengkap")


def _family(cells, build, labels=None, family=""):
    """Keluarga uji generik: build(seed) -> {nama: hasil _diff_test atau None}. Holm per fold,
    lalu putusan mayoritas fold (>= 4 dari 5)."""
    seeds = _seeds(cells)
    per_seed = {}
    for seed in seeds:
        tests = build(seed)
        adj = S.holm({k: (v["p"] if v else None) for k, v in tests.items()})
        for k, v in tests.items():
            if v:
                v["p_holm"] = adj[k]
        per_seed[seed] = tests
    dec = {}
    for name in per_seed[seeds[0]] if seeds else []:
        per = {s: {"p_adj": (per_seed[s][name] or {}).get("p_holm"),
                   "sign": (per_seed[s][name] or {}).get("sign", 0),
                   "ne": (per_seed[s][name] or {}).get("ne", False)} for s in seeds}
        dd = _decide(per, len(seeds))
        if labels:
            dd["label"] = labels.get(dd["verdict"], dd["verdict"])
        dd["mean_diff"] = S.across_seeds([per_seed[s][name]["mean"] for s in seeds
                                          if per_seed[s][name] and per_seed[s][name].get("mean") is not None])
        dec[name] = dd
    n_tests = len(per_seed[seeds[0]]) if seeds else 0
    return {"family": family, "n_tests_per_fold": n_tests,
            "min_attainable_p_holm": (n_tests * 2 / (C.N_BOOT + 1)) if n_tests else None,
            "per_seed": per_seed, "decisions": dec}


def _diff_test(a, b):
    if a is None or b is None:
        return None
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.isfinite(d).sum() < 10:
        return {"ne": True, "p": None, "mean": None, "ci95": [None, None], "sign": 0}
    fin = d[np.isfinite(d)]
    return {"mean": float(fin.mean()), "ci95": S.ci(d), "p": S.p_boot(d),
            "sign": int(np.sign(np.median(fin)))}


# ==========================================================================
# Agregasi
# ==========================================================================
AGG_METRICS = ["mae", "rmse", "gap_gender", "gap_gender_signed", "gap_age_ref", "gap_occ",
               "relx_occ", "excess_occ", "p_perm_occ", "dp_o",
               "dxi_gender", "xi_F", "xi_M", "moment_F", "moment_M", "dmoment_gender", "dp_g", "dp_u",
               "p10", "p10_F", "p10_M", "p10_nothr", "amp_gender", "relx_gender", "relx_age",
               "excess_gender", "excess_age", "p_perm_gender", "p_perm_age", "misc_gap_g",
               "misc_gap_u", "cos_F_M", "w_median_w", "xi_top10_F", "xi_top10_M",
               "xi_top10nothr_F", "xi_top10nothr_M", "train_seconds", "train_mae_sample",
               "train_mae_global_mean", "jaccard_FvsM_top10", "sub_mae", "sub_gap_gender",
               "sub_gap_age_ref", "sub_gap_occ", "sub_relx_gender", "sub_relx_age", "sub_relx_occ",
               "rgap_gender", "rgap_age_ref", "rgap_occ"]


def aggregate(cells: Cells) -> list:
    d = cells.df
    cols = [c for c in AGG_METRICS if c in d]
    g = d.groupby(["path", "scenario", "model", "k", "alpha", "encoding", "variant"])
    out = []
    for key, grp in g:
        row = dict(zip(["path", "scenario", "model", "k", "alpha", "encoding", "variant"], key))
        row["n_seeds"] = int(grp.seed.nunique())
        for c in cols:
            v = pd.to_numeric(grp[c], errors="coerce")
            row[c] = float(v.mean()) if v.notna().any() else None
            row[f"{c}_sd"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else None
        for gname in C.GENDER_ORDER:
            for c in [f"xi_estimable_{gname}"]:
                if c in grp:
                    row[c + "_rate"] = float(grp[c].astype(float).mean())
        out.append(row)
    return out


# ==========================================================================
# Pareto
# ==========================================================================
def pareto(cells: Cells) -> dict:
    d = cells.df[(cells.df.encoding == "full") & (cells.df.variant == "-")]
    seeds = _seeds(cells)
    need = min(C.SEED_MAJORITY, len(seeds))
    axes = {"MAE_vs_gap_gender": "gap_gender", "MAE_vs_gap_age_ref": "gap_age_ref",
            "MAE_vs_gap_occ": "gap_occ", "MAE_vs_dxi_gender": "dxi_gender"}
    res = {}
    for path in ["warm", "cold"]:
        sub = d[d.path == path]
        if path == "cold":
            sub = sub[((sub.alpha == 0) & (sub.scenario == "default")) |
                      ((sub.alpha > 0) & (sub.scenario == C.COLD_PRIMARY_SCENARIO))]
        if sub.empty:
            continue
        res[path] = {}
        for ax, met in axes.items():
            if met not in sub:
                continue
            piv = sub.pivot_table(index=["model", "k", "alpha"], columns="seed",
                                  values=["mae", met], aggfunc="first")
            pts = []
            for idx, r in piv.iterrows():
                mae_s = r["mae"].to_numpy(float)
                met_s = pd.to_numeric(r[met], errors="coerce").to_numpy(float)
                if np.isfinite(met_s).sum() < need:
                    continue
                pts.append({"model": idx[0], "k": int(idx[1]), "alpha": int(idx[2]),
                            "mae": float(np.nanmean(mae_s)), "disp": float(np.nanmean(met_s)),
                            "mae_sd": float(np.nanstd(mae_s, ddof=1)),
                            "disp_sd": float(np.nanstd(met_s, ddof=1)),
                            "_mae": mae_s, "_disp": met_s})
            if not pts:
                continue
            for p in pts:
                p["dominated_by"] = []
            for a in pts:
                for b in pts:
                    if a is b:
                        continue
                    mean_dom = (a["mae"] <= b["mae"] and a["disp"] <= b["disp"] and
                                (a["mae"] < b["mae"] or a["disp"] < b["disp"]))
                    if not mean_dom:
                        continue
                    ok = np.isfinite(a["_mae"]) & np.isfinite(b["_mae"]) & \
                        np.isfinite(a["_disp"]) & np.isfinite(b["_disp"])
                    seed_dom = ((a["_mae"] <= b["_mae"]) & (a["_disp"] <= b["_disp"]) &
                                ((a["_mae"] < b["_mae"]) | (a["_disp"] < b["_disp"])) & ok).sum()
                    if seed_dom >= need:
                        b["dominated_by"].append(f"{a['model']}|k{a['k']}|a{a['alpha']}")
            for p in pts:
                p["on_front"] = len(p["dominated_by"]) == 0
                p.pop("_mae"); p.pop("_disp")
            front = [p for p in pts if p["on_front"]]
            res[path][ax] = {
                "points": pts, "front": sorted(front, key=lambda p: p["mae"]),
                "n_front_SVD": sum(p["model"] == "SVD" for p in front),
                "n_front_NMF": sum(p["model"] == "NMF" for p in front),
                "NMF_points_dominated_by_SVD": int(sum(
                    1 for p in pts if p["model"] == "NMF" and
                    any(x.startswith("SVD") for x in p["dominated_by"]))),
                "SVD_points_dominated_by_NMF": int(sum(
                    1 for p in pts if p["model"] == "SVD" and
                    any(x.startswith("NMF") for x in p["dominated_by"]))),
                "n_points": {m: sum(p["model"] == m for p in pts) for m in C.MODELS}}
    return res


# ==========================================================================
# F1: Hipotesis Pertama
# ==========================================================================
def h1(cells: Cells, ksel: dict) -> dict:
    """F1 (v3.3): SVD vs NMF pada alpha = 0, k terpilih. Metrik gap berbasis MAE (jalur warm
    dan cold) menjadi keluarga F1; Delta-xi (EVT) diuji di keluarga terpisah F1-EVT dan tidak
    ikut menentukan putusan, karena presisi Pickands jauh lebih rendah (lihat analysis.evt)."""
    def pair(path, met, seed):
        sc = _scen(path, 0)
        a = cells.rep(path, "SVD", ksel["SVD"], 0, seed, sc, metric=met)
        b = cells.rep(path, "NMF", ksel["NMF"], 0, seed, sc, metric=met)
        pa = cells.point(path, "SVD", ksel["SVD"], 0, seed, met, sc)
        pb = cells.point(path, "NMF", ksel["NMF"], 0, seed, met, sc)
        t = _diff_test(a, b)
        if t is not None and (pa is None or pb is None):
            t = {"ne": True, "p": None, "mean": None, "ci95": [None, None], "sign": 0}
        return t
    labels = {"naik": "SVD lebih bias", "turun": "NMF lebih bias", "tidak berbeda": "tidak berbeda"}
    fam = _family(cells, lambda seed: {f"{path}|{met}": pair(path, met, seed)
                                       for path in ["warm", "cold"] for met in C.GAP_METRICS},
                  labels, "F1")
    # EVT: hanya jalur warm (pada cold alpha = 0 semua pengguna mendapat daftar Default yang
    # sama sehingga indeks ekor tidak terestimasi)
    fam_evt = _family(cells, lambda seed: {f"warm|{met}": pair("warm", met, seed)
                                           for met in C.EVT_METRICS}, labels, "F1-EVT")
    decisions = fam["decisions"]
    warm = {m: decisions.get(f"warm|{m}") for m in C.GAP_METRICS}
    n_svd = sum(1 for v in warm.values() if v and v["label"] == "SVD lebih bias")
    n_nmf = sum(1 for v in warm.values() if v and v["label"] == "NMF lebih bias")
    if any((v is None) or (v["verdict"] in MISSING) for v in warm.values()):
        verdict = "belum dapat diputuskan (data belum lengkap)"
    elif n_svd >= 2 and n_nmf == 0:
        verdict = "didukung"
    elif n_nmf >= 2:
        verdict = "ditolak"
    elif n_svd == 0 and n_nmf == 0:
        verdict = "tidak didukung (tidak ada perbedaan signifikan)"
    else:
        verdict = "didukung sebagian"
    secondary = {}
    for path in ["warm", "cold"]:
        for a in [x for x in C.ALPHAS if x > 0]:
            sc = _scen(path, a)
            for met in C.DISPARITY_METRICS:
                diffs = []
                for seed in _seeds(cells):
                    t = _diff_test(cells.rep(path, "SVD", ksel["SVD"], a, seed, sc, metric=met),
                                   cells.rep(path, "NMF", ksel["NMF"], a, seed, sc, metric=met))
                    if t and not t.get("ne"):
                        diffs.append(t)
                if diffs:
                    secondary[f"{path}|a{a}|{met}"] = {
                        "mean_diff": S.across_seeds([t["mean"] for t in diffs]),
                        "n_seeds_p_lt_05_svd_more": sum(t["p"] < .05 and t["sign"] > 0 for t in diffs),
                        "n_seeds_p_lt_05_nmf_more": sum(t["p"] < .05 and t["sign"] < 0 for t in diffs)}
    for f in (fam, fam_evt):
        for name, dd in f["decisions"].items():
            dd["mean_diff_SVD_minus_NMF"] = dd.pop("mean_diff")
            dd["n_seeds_estimable"] = sum(1 for sd, tt in f["per_seed"].items()
                                          if tt.get(name) and not tt[name].get("ne"))
    return {**fam, "primary_config": {"alpha": 0, "k": ksel}, "verdict": verdict,
            "evt": fam_evt,
            "rule": "didukung bila SVD lebih bias pada >= 2 dari 3 metrik gap jalur warm dan "
                    "tidak ada metrik di mana NMF lebih bias; ditolak bila NMF lebih bias pada "
                    ">= 2 metrik; tidak didukung bila tidak ada yang berbeda. Delta-xi dilaporkan "
                    "di keluarga F1-EVT dan tidak ikut menentukan putusan (v3.3).",
            "secondary_uncorrected": secondary}


# ==========================================================================
# F2 efek alpha, bobot aman, F4 amplifikasi
# ==========================================================================
def alpha_effect(cells: Cells, ksel: dict) -> dict:
    """F2: m(alpha) - m(0) per model x jalur x alpha. Metrik gap = keluarga F2; Delta-xi
    (jalur warm) = keluarga F2-EVT."""
    labels = {"naik": "disparitas memburuk", "turun": "disparitas membaik",
              "tidak berbeda": "tidak berubah signifikan"}
    def t(path, m, a, met, seed):
        return _diff_test(cells.rep(path, m, ksel[m], a, seed, _scen(path, a), metric=met),
                          cells.rep(path, m, ksel[m], 0, seed, _scen(path, 0), metric=met))
    pos = [x for x in C.ALPHAS if x > 0]
    fam = _family(cells, lambda seed: {f"{m}|{path}|a{a}|{met}": t(path, m, a, met, seed)
                                       for m in C.MODELS for path in ["warm", "cold"]
                                       for a in pos for met in C.GAP_METRICS}, labels, "F2")
    fam["evt"] = _family(cells, lambda seed: {f"{m}|warm|a{a}|{met}": t("warm", m, a, met, seed)
                                              for m in C.MODELS for a in pos
                                              for met in C.EVT_METRICS}, labels, "F2-EVT")
    return fam


def safe_alpha(cells: Cells, ksel: dict) -> dict:
    """Bobot aman (non-inferioritas) per model x jalur x encoding. v3.3: hanya metrik gap
    berbasis MAE (C.SAFE_METRICS); Delta-xi dilaporkan deskriptif karena margin 10% x xi(0)
    jauh di bawah presisi Pickands sehingga uji non-inferioritasnya tidak memiliki daya."""
    seeds = _seeds(cells)
    need = min(C.SEED_MAJORITY, len(seeds))
    out = {}
    for m in C.MODELS:
        for path in ["warm", "cold"]:
            for enc in [C.MAIN_ENCODING] + C.ABLATION_ENCODINGS:
                if enc != C.MAIN_ENCODING and path not in C.ABLATION_PATHS:
                    continue
                table = []
                for a in [x for x in C.ALPHAS if x > 0]:
                    passes, detail = 0, []
                    for seed in seeds:
                        ok, info = True, {"seed": seed}
                        for met in C.SAFE_METRICS + C.EVT_METRICS:
                            p0 = cells.point(path, m, ksel[m], 0, seed, met, _scen(path, 0))
                            t = _diff_test(
                                cells.rep(path, m, ksel[m], a, seed, _scen(path, a), metric=met,
                                          encoding=enc),
                                cells.rep(path, m, ksel[m], 0, seed, _scen(path, 0), metric=met))
                            if met in C.EVT_METRICS:
                                info[met + "_descriptive"] = None if not t or t.get("ne") else \
                                    {"mean": t["mean"], "ci95": t["ci95"]}
                                continue
                            if t is None or p0 is None or t.get("ne"):
                                ok, info[met] = False, "tidak ada data"
                                continue
                            delta = C.SAFE_MARGIN_REL * abs(p0)
                            ub = t["ci95"][1]
                            info[met] = {"ub95": ub, "delta": delta,
                                         "pass": ub is not None and ub <= delta}
                            ok &= bool(info[met]["pass"])
                        mae0 = cells.point(path, m, ksel[m], 0, seed, "mae", _scen(path, 0))
                        maea = cells.point(path, m, ksel[m], a, seed, "mae", _scen(path, a),
                                           encoding=enc)
                        mae_ok = (mae0 is not None and maea is not None and
                                  maea - mae0 <= C.SAFE_MAE_TOL)
                        info["mae_diff"] = None if mae0 is None or maea is None else maea - mae0
                        info["mae_pass"] = mae_ok
                        ok &= mae_ok
                        info["pass"] = bool(ok)
                        info["has_data"] = mae0 is not None and maea is not None
                        passes += int(ok)
                        detail.append(info)
                    w = cells.df[(cells.df.path == path) & (cells.df.model == m) &
                                 (cells.df.k == ksel[m]) & (cells.df.alpha == a) &
                                 (cells.df.encoding == enc) & (cells.df.variant == "-")]
                    n_data = sum(1 for x in detail if x["has_data"])
                    table.append({"alpha": a, "n_seeds_pass": passes, "n_seeds_with_data": n_data,
                                  "safe": (passes >= need) if n_data >= need else None,
                                  "median_relative_weight": float(w.w_median_w.mean())
                                  if "w_median_w" in w and w.w_median_w.notna().any() else None,
                                  "per_seed": detail})
                safe = [r["alpha"] for r in table if r["safe"]]
                missing = any(r["safe"] is None for r in table)
                if missing:
                    stmt = "data belum lengkap; bobot aman belum dapat ditentukan"
                elif safe:
                    stmt = (f"alpha aman: {safe} (aman = tidak merugikan disparitas dan MAE, "
                            "bukan berarti bermanfaat)")
                else:
                    stmt = "tidak ada bobot demografi yang aman pada konfigurasi ini"
                out[f"{m}|{path}|{enc}"] = {"k": ksel[m], "encoding": enc, "table": table,
                                            "alpha_safe": max(safe) if safe and not missing else None,
                                            "alpha_safe_list": safe, "statement": stmt}
    return {"rule": "UB95[m(alpha)-m(0)] <= 10% m(0) untuk setiap metrik gap (gender, usia, "
                    "pekerjaan) dan MAE(alpha)-MAE(0) <= 0,005, pada >= 4 dari 5 fold; per "
                    "encoding (v3.3). Delta-xi dilaporkan deskriptif.", "results": out}


def amplification(cells: Cells, ksel: dict) -> dict:
    labels = {"naik": "amplifikasi naik", "turun": "amplifikasi turun", "tidak berbeda": "tidak berbeda"}
    fam = _family(cells, lambda seed: {
        f"{m}|a{a}": _diff_test(cells.rep("warm", m, ksel[m], a, seed, metric="amp_gender"),
                                cells.rep("warm", m, ksel[m], 0, seed, metric="amp_gender"))
        for m in C.MODELS for a in C.ALPHAS if a > 0}, labels, "F4")
    for dd in fam["decisions"].values():
        dd["claim_valid"] = dd["verdict"] == "naik"
        dd["mean_A_alpha_minus_A0"] = dd.pop("mean_diff")
    fam["rule"] = ("klaim memperbesar perbedaan genre sah bila A(alpha)-A(0) > 0 signifikan "
                   "(Holm) pada >= 4 dari 5 fold, jalur warm")
    return fam


# ==========================================================================
# F3 klaim pengguna baru (berpasangan) dan F3b profil vs Default
# ==========================================================================
def new_users(cells: Cells, ksel: dict) -> dict:
    """F3 (v3.3): excess gap relatif pengguna fold cold-start saat menjadi pengguna baru
    (jalur cold, skenario utama) dikurangi excess gap PENGGUNA YANG SAMA saat menjadi pengguna
    lama (jalur warm, metrik sub_*). Kedua sisi memakai bobot bootstrap jalur cold yang sama,
    sehingga uji berpasangan (v3.2 membandingkan dua populasi dengan bobot independen)."""
    labels = {"naik": "pengguna baru menanggung excess gap lebih besar",
              "turun": "pengguna baru menanggung excess gap lebih kecil",
              "tidak berbeda": "tidak berbeda"}
    def t(m, a, attr, seed):
        c = cells.rep("cold", m, ksel[m], a, seed, C.COLD_PRIMARY_SCENARIO, metric=f"relx_{attr}")
        w = cells.rep("warm", m, ksel[m], a, seed, metric=f"sub_relx_{attr}")
        r = _diff_test(c, w)
        if r and not r.get("ne"):
            r["cold_point"] = cells.point("cold", m, ksel[m], a, seed, f"relx_{attr}",
                                          C.COLD_PRIMARY_SCENARIO)
            r["warm_same_users_point"] = cells.point("warm", m, ksel[m], a, seed, f"sub_relx_{attr}")
        return r
    fam = _family(cells, lambda seed: {f"{m}|a{a}|{attr}": t(m, a, attr, seed)
                                       for m in C.MODELS for a in C.ALPHAS if a > 0
                                       for attr in ["gender", "age", "occ"]}, labels, "F3")
    for name, dd in fam["decisions"].items():
        vals = [fam["per_seed"][s][name] for s in fam["per_seed"] if fam["per_seed"][s][name]]
        dd["claim_valid"] = dd["verdict"] == "naik"
        dd["mean_relx_cold"] = S.across_seeds([v.get("cold_point") for v in vals])
        dd["mean_relx_warm_same_users"] = S.across_seeds([v.get("warm_same_users_point") for v in vals])
    fam["rule"] = ("klaim sah bila excess gap relatif cold > warm pada pengguna yang sama "
                   "signifikan (Holm) pada >= 4 dari 5 fold")
    return fam


def new_user_profile(cells: Cells, ksel: dict) -> dict:
    """F3b (v3.3): apakah profil demografi membantu pengguna baru? Skenario utama (gender +
    usia + pekerjaan) dikurangi skenario Default pada model yang SAMA (alpha dan encoding
    sama), pengguna sama, bobot bootstrap sama. Nilai negatif = profil lebih baik (MAE/gap
    lebih kecil). Encoding utama = keluarga F3b; encoding hanya-anggota = F3b-ablation."""
    labels = {"naik": "profil demografi memperburuk", "turun": "profil demografi memperbaiki",
              "tidak berbeda": "tidak berbeda"}
    mets = ["mae"] + C.GAP_METRICS
    def t(m, a, met, seed, enc):
        return _diff_test(
            cells.rep("cold", m, ksel[m], a, seed, C.COLD_PRIMARY_SCENARIO, metric=met, encoding=enc),
            cells.rep("cold", m, ksel[m], a, seed, "default", metric=met, encoding=enc))
    pos = [x for x in C.ALPHAS if x > 0]
    fam = _family(cells, lambda seed: {f"{m}|a{a}|{met}": t(m, a, met, seed, C.MAIN_ENCODING)
                                       for m in C.MODELS for a in pos for met in mets},
                  labels, "F3b")
    if "cold" in C.ABLATION_PATHS:
        fam["ablation"] = _family(cells, lambda seed: {
            f"{enc}|{m}|a{a}|{met}": t(m, a, met, seed, enc)
            for enc in C.ABLATION_ENCODINGS for m in C.MODELS for a in pos for met in mets},
            labels, "F3b-ablation")
    fam["relative"] = _family(cells, lambda seed: {
        f"{m}|a{a}|r{met}": t(m, a, f"r{met}", seed, C.MAIN_ENCODING)
        for m in C.MODELS for a in pos for met in C.GAP_METRICS}, labels, "F3b-relatif")
    fam["rule"] = ("profil demografi (skenario utama) dibanding Default pada model yang sama; "
                   "putusan per metrik bila signifikan (Holm) pada >= 4 dari 5 fold; 'relative' = "
                   "gap / MAE (v3.4)")
    return fam


# ==========================================================================
# F5 mekanisme, F6 H2, kebocoran
# ==========================================================================
def mechanism(cells: Cells, ksel: dict) -> dict:
    d = cells.df
    seeds = _seeds(cells)
    res = {"cos_F_M": {}, "probe_gender_auc": {}, "frac_negative": {}}
    for m in C.MODELS:
        for enc in ["full"] + C.ABLATION_ENCODINGS:
            for a in [x for x in C.ALPHAS if x > 0]:
                sub = d[(d.path == "warm") & (d.model == m) & (d.k == ksel[m]) &
                        (d.alpha == a) & (d.encoding == enc) & (d.variant == "-")]
                if sub.empty:
                    continue
                res["cos_F_M"][f"{m}|{enc}|a{a}"] = S.across_seeds(sub.cos_F_M.tolist())
                res["frac_negative"][f"{m}|{enc}|a{a}"] = S.across_seeds(sub.frac_negative_pu.tolist())
                aucs = []
                for seed in seeds:
                    ex = cells.extra.get(cells.key("warm", m, ksel[m], a, seed, encoding=enc), {})
                    pr = ex.get("latent", {}).get("probe", {}).get("gender")
                    if pr:
                        aucs.append(pr["mean"])
                res["probe_gender_auc"][f"{m}|{enc}|a{a}"] = S.across_seeds(aucs)
    verdict = {}
    for enc in C.ABLATION_ENCODINGS + ["full"]:
        hold = []
        for a in [x for x in C.ALPHAS if x > 0]:
            v = res["cos_F_M"].get(f"SVD|{enc}|a{a}")
            if v and v["ci95_t"][1] is not None:
                hold.append(v["ci95_t"][1] <= C.MECH_COS_SVD_MAX)
        verdict[enc] = {"n_alpha_hold": int(sum(hold)), "n_alpha": len(hold),
                        "holds": bool(sum(hold) >= min(C.MECH_MIN_ALPHAS, max(len(hold), 1)))}
    ab = [verdict[e]["holds"] for e in C.ABLATION_ENCODINGS if verdict[e]["n_alpha"]]
    res["verdict_per_encoding"] = verdict
    if len(ab) < len(C.ABLATION_ENCODINGS):
        res["mechanism_supported"] = None
        res["mechanism_status"] = "data ablation belum lengkap"
    else:
        res["mechanism_supported"] = all(ab)
        res["mechanism_status"] = "didukung" if all(ab) else "tidak didukung"
        res["interpretation"] = (
            "kosinus F vs M SVD tetap negatif tanpa encoding full: polarisasi berasal dari SVD"
            if all(ab) else
            "kosinus F vs M SVD negatif hanya pada encoding full: polarisasi gender di ruang laten "
            "adalah artefak nilai bukan-anggota (1) pada encoding, bukan sifat SVD")
    res["note_nmf"] = ("Cosinus antar vektor NMF selalu >= 0 karena seluruh komponennya "
                       "non-negatif, sehingga kriteria NMF terpenuhi secara matematis; bukti "
                       "mekanisme ditentukan oleh perilaku SVD lintas encoding.")
    return res


def encoding_effect(cells: Cells, ksel: dict) -> dict:
    """F5b (v3.3): encoding hanya-anggota dikurangi encoding full pada alpha, model, fold, dan
    pengguna yang sama (bootstrap berpasangan). Nilai negatif = encoding hanya-anggota lebih
    baik (MAE, gap, atau amplifikasi lebih kecil). Satu keluarga per (jalur, encoding)."""
    labels = {"naik": "hanya-anggota lebih buruk", "turun": "hanya-anggota lebih baik",
              "tidak berbeda": "tidak berbeda"}
    out = {}
    pos = [x for x in C.ALPHAS if x > 0]
    for path in C.ABLATION_PATHS:
        mets = ["mae"] + C.GAP_METRICS + (["amp_gender"] if path == "warm" else [])
        for enc in C.ABLATION_ENCODINGS:
            def build(seed, path=path, enc=enc, mets=mets):
                sc = lambda a: _scen(path, a)
                return {f"{m}|a{a}|{met}": _diff_test(
                    cells.rep(path, m, ksel[m], a, seed, sc(a), metric=met, encoding=enc),
                    cells.rep(path, m, ksel[m], a, seed, sc(a), metric=met))
                    for m in C.MODELS for a in pos for met in mets}
            out[f"{path}|{enc}"] = _family(cells, build, labels, f"F5b-{path}-{enc}")
    return out


def _partial_spearman(x, y, z):
    """Korelasi Spearman parsial x-y dengan z dikontrol (residual peringkat)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = len(x)
    if n <= 3 or abs(r) >= 1:
        return r, 0.0 if abs(r) >= 1 else np.nan
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), n - 3))


def h2(cells: Cells) -> dict:
    """F6 (v3.2). Uji utama: korelasi Spearman PARSIAL antara akurasi (-MAE) dan disparitas
    lintas konfigurasi k x alpha di dalam setiap algoritma, dengan alpha dikontrol.
    Alasan: alpha menaikkan MAE dan disparitas sekaligus, sehingga korelasi biasa lintas
    k x alpha terkacaukan oleh alpha (v3.1). Analisis deskriptif: Spearman per alpha lintas
    k, dan korelasi gabungan tanpa kontrol (versi lama)."""
    d = cells.df[(cells.df.path == "warm") & (cells.df.encoding == "full") &
                 (cells.df.variant == "-")]
    tests, per_alpha, pooled = {}, {}, {}
    for m in C.MODELS:
        agg = d[d.model == m].groupby(["k", "alpha"])[["mae"] + C.DISPARITY_METRICS] \
            .mean(numeric_only=True).reset_index()
        n_k = agg.k.nunique()
        for met in C.DISPARITY_METRICS:
            x, y, z = -agg["mae"].to_numpy(), agg[met].to_numpy(float), agg["alpha"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if n_k < 3:
                tests[f"{m}|{met}"] = None
            elif ok.sum() < 6:
                tests[f"{m}|{met}"] = {"ne": True, "p": None, "partial_rho": None}
            else:
                r, p = _partial_spearman(x[ok], y[ok], z[ok])
                tests[f"{m}|{met}"] = {"partial_rho": r, "p": p, "n_points": int(ok.sum()),
                                       "n_k": int(n_k)}
                rp = stats.spearmanr(x[ok], y[ok])
                pooled[f"{m}|{met}"] = {"rho": float(rp.statistic), "p": float(rp.pvalue)}
            for a, ga in agg.groupby("alpha"):
                xx, yy = -ga["mae"].to_numpy(), ga[met].to_numpy(float)
                ok2 = np.isfinite(xx) & np.isfinite(yy)
                per_alpha[f"{m}|{met}|a{int(a)}"] = (
                    {"rho": float(stats.spearmanr(xx[ok2], yy[ok2]).statistic), "n_k": int(ok2.sum())}
                    if ok2.sum() >= 3 else None)
    adj = S.holm({k: (v["p"] if v else None) for k, v in tests.items()})
    verdicts = {}
    for m in C.MODELS:
        if any(tests.get(f"{m}|{met}") is None for met in C.DISPARITY_METRICS):
            verdicts[m] = "data belum lengkap"
            continue
        avail = [met for met in C.DISPARITY_METRICS if not tests[f"{m}|{met}"].get("ne")]
        pos = sum(1 for met in avail if adj[f"{m}|{met}"] is not None and
                  adj[f"{m}|{met}"] < C.SIG_LEVEL and tests[f"{m}|{met}"]["partial_rho"] > 0)
        neg = sum(1 for met in avail if adj[f"{m}|{met}"] is not None and
                  adj[f"{m}|{met}"] < C.SIG_LEVEL and tests[f"{m}|{met}"]["partial_rho"] < 0)
        verdicts[m] = ("arah trade-off" if pos >= 2 else
                       ("arah searah (akurasi dan keadilan sejalan)" if neg >= 2 else "campuran"))
    for k, v in tests.items():
        if v:
            v["p_holm"] = adj[k]
    return {"family": "F6 (deskriptif, v3.3)", "tests": tests, "per_alpha_descriptive": per_alpha,
            "pooled_uncontrolled_descriptive": pooled, "direction": verdicts,
            "verdict": {m: "deskriptif (tidak diuji konfirmatori)" for m in C.MODELS},
            "rule": "korelasi Spearman parsial (-MAE vs disparitas, alpha dikontrol) pada "
                    "konfigurasi k x alpha (rata-rata fold), dilaporkan deskriptif: titik-titik "
                    "konfigurasi tidak saling bebas (data dan pengguna sama) sehingga nilai p "
                    "terlalu optimistis; arah dibaca bersama front Pareto",
            "caveat": f"hanya {len(C.KS)} nilai k per alpha"}


def leakage(cells: Cells, ksel: dict) -> dict:
    """Kebocoran kelompok pekerjaan ke ruang laten. Hanya sah bila pekerjaan BUKAN input:
    alpha = 0 (konfigurasi utama) dan ablation input gender + usia (variant 'gu')."""
    seeds = _seeds(cells)
    out = {}
    combos = [("-", 0)] + [("gu", a) for a in C.ALPHAS if a > 0]
    for m in C.MODELS:
        for var, a in combos:
            diffs, lat, base = [], [], []
            for seed in seeds:
                ex = cells.extra.get(cells.key("warm", m, ksel[m], a, seed, variant=var),
                                     {}).get("latent", {})
                p = ex.get("probe", {}).get("occ_group")
                b = ex.get("baseline_probe", {}).get("occ_group")
                if p and b:
                    diffs += list(np.array(p["folds"]) - np.array(b["folds"]))
                    lat.append(p["mean"]); base.append(b["mean"])
            if not diffs:
                continue
            s_ = S.across_seeds(diffs)
            out[f"{m}|{'utama' if var == '-' else 'input_gender_usia'}|a{a}"] = {
                "latent_bal_acc": S.across_seeds(lat), "baseline_bal_acc": S.across_seeds(base),
                "diff_latent_minus_baseline": s_,
                "leak": bool(s_["ci95_t"][0] is not None and s_["ci95_t"][0] > 0)}
    return {"rule": "kelompok pekerjaan disebut bocor hanya bila balanced accuracy probe laten "
                    "melampaui baseline probe (gender + usia) dengan CI 95% di atas nol; diuji "
                    "hanya pada konfigurasi tanpa pekerjaan sebagai input",
            "results": out}


def occupation_effect(cells: Cells, ksel: dict) -> dict:
    """F7: kontribusi marginal pekerjaan sebagai input = konfigurasi utama (gender + usia +
    pekerjaan) dikurangi ablation (gender + usia), jalur warm, k terpilih, per alpha."""
    seeds = _seeds(cells)
    mets = ["mae"] + C.GAP_METRICS
    per_seed = {}
    for seed in seeds:
        tests = {}
        for m in C.MODELS:
            for a in [x for x in C.ALPHAS if x > 0]:
                for met in mets:
                    tests[f"{m}|a{a}|{met}"] = _diff_test(
                        cells.rep("warm", m, ksel[m], a, seed, metric=met),
                        cells.rep("warm", m, ksel[m], a, seed, metric=met, variant="gu"))
        adj = S.holm({k: (v["p"] if v else None) for k, v in tests.items()})
        for k, v in tests.items():
            if v:
                v["p_holm"] = adj[k]
        per_seed[seed] = tests
    dec = {}
    for name in per_seed[seeds[0]] if seeds else []:
        per = {s: {"p_adj": (per_seed[s][name] or {}).get("p_holm"),
                   "sign": (per_seed[s][name] or {}).get("sign", 0),
                   "ne": (per_seed[s][name] or {}).get("ne", False)} for s in seeds}
        dd = _decide(per, len(seeds))
        dd["label"] = {"naik": "menambah pekerjaan menaikkan nilai",
                       "turun": "menambah pekerjaan menurunkan nilai",
                       "tidak berbeda": "tidak berbeda"}.get(dd["verdict"], dd["verdict"])
        dd["mean_diff_with_minus_without"] = S.across_seeds(
            [per_seed[s][name]["mean"] for s in seeds if per_seed[s][name]])
        dec[name] = dd
    return {"family": "F7", "per_seed": per_seed, "decisions": dec,
            "note": "Untuk MAE dan metrik disparitas, nilai turun berarti lebih baik."}


def evt_diagnostics(cells: Cells, ksel: dict) -> dict:
    d = cells.df[(cells.df.encoding == "full") & (cells.df.variant == "-")]
    out = {"estimability": {}, "agreement": {}, "curves": {}}
    for (path, sc, m, k), g in d.groupby(["path", "scenario", "model", "k"]):
        if "xi_estimable_F" in g:
            out["estimability"][f"{path}|{sc}|{m}|k{k}"] = {
                "F": float(g.xi_estimable_F.astype(float).mean()),
                "M": float(g.xi_estimable_M.astype(float).mean())}
    w = d[d.path == "warm"]
    if {"dxi_gender", "dmoment_gender"} <= set(w.columns):
        sub = w[["dxi_gender", "dmoment_gender"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) >= 5:
            out["agreement"]["spearman_dxi_vs_dmoment"] = float(stats.spearmanr(sub.dxi_gender, sub.dmoment_gender).statistic)
    for m in C.MODELS:
        for a in [0, max(C.ALPHAS)]:
            ex = cells.extra.get(cells.key("warm", m, ksel[m], a, 0), {})
            if ex.get("curves"):
                out["curves"][f"{m}|a{a}"] = ex["curves"]
    return out


def popularity(agg: list) -> list:
    rows = []
    for r in agg:
        if r["encoding"] != "full" or r["variant"] != "-":
            continue
        rows.append({k: r.get(k) for k in ["path", "scenario", "model", "k", "alpha", "p10",
                                           "p10_nothr", "xi_top10_F", "xi_top10nothr_F",
                                           "xi_top10_M", "xi_top10nothr_M"]})
    return rows


def rankings(cells: Cells, ksel: dict, titles) -> dict:
    out = {}
    for m in C.MODELS:
        for a in C.ALPHAS:
            for sc in (["default"] if a == 0 else list(C.SCENARIOS)):
                ex = cells.extra.get(cells.key("cold", m, ksel[m], a, 0, sc), {})
                rk = ex.get("rankings")
                if rk:
                    out[f"{m}|a{a}|{sc}"] = {p: [{"rank": i + 1, "movieId": mid,
                                                   "title": titles.title.get(mid, ""),
                                                   "genres": titles.genres.get(mid, "")}
                                                  for i, mid in enumerate(lst)]
                                             for p, lst in rk.items()}
    return out


def evt_sensitivity(cells: Cells, ksel: dict) -> dict:
    """Uji sensitivitas rentang k dari jumlah eksposur yang disimpan (jalur warm, k terpilih):
    Pickands dan Moment stabil per gender untuk beberapa rentang k, rata-rata fold. Hill tidak
    dipakai karena hanya konsisten untuk xi > 0 (Hill, 1975; de Haan & Ferreira, 2006)."""
    from src import evt as E
    out = {}
    for m in C.MODELS:
        for a in C.ALPHAS:
            acc = {}
            for seed in _seeds(cells):
                ex = cells.extra.get(cells.key("warm", m, ksel[m], a, seed), {})
                cnt = ex.get("curves", {}).get("exposure_counts", {}).get("gender", {})
                for g, arr in cnt.items():
                    for rng, v in E.sensitivity(arr).items():
                        for est in ["pickands", "moment"]:
                            acc.setdefault((g, rng, est), []).append(v[est])
            if acc:
                out[f"{m}|a{a}"] = {f"{g}|{rng}|{est}": S.across_seeds(vals)["mean"]
                                    for (g, rng, est), vals in acc.items()}
    return {"ranges": [f"{lo:g}-{hi:g}" for lo, hi in E.SENSITIVITY_RANGES],
            "primary_range": f"{C.EVT_K_LO:g}-{C.EVT_K_HI:g}", "results": out}


def evt_agreement(cells: Cells) -> dict:
    """Kesesuaian Pickands dengan Moment (keduanya konsisten untuk semua tanda xi; Dekkers,
    Einmahl, & de Haan, 1989). Hill tidak dipakai sebagai pembanding (hanya sah untuk xi > 0)."""
    d = cells.df[(cells.df.path == "warm") & (cells.df.variant == "-")]
    out = {}
    if {"dxi_gender", "dmoment_gender"} <= set(d.columns):
        sub = d[["dxi_gender", "dmoment_gender"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) >= 5:
            out["spearman_dpickands_vs_dmoment"] = float(stats.spearmanr(sub.dxi_gender, sub.dmoment_gender).statistic)
            out["n_cells"] = int(len(sub))
    for g in C.GENDER_ORDER:
        c = f"moment_negative_{g}"
        if c in d:
            v = d[c].dropna()
            out[f"share_moment_negative_{g}"] = float(v.astype(float).mean()) if len(v) else None
        if f"xi_{g}" in d and f"moment_{g}" in d:
            sub = d[[f"xi_{g}", f"moment_{g}"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub):
                out[f"share_pickands_gt_1_{g}"] = float((sub[f"xi_{g}"] > 1).mean())
                out[f"share_same_sign_pickands_moment_{g}"] = float(
                    (np.sign(sub[f"xi_{g}"]) == np.sign(sub[f"moment_{g}"])).mean())
                out[f"spearman_xi_pickands_vs_moment_{g}"] = float(
                    stats.spearmanr(sub[f"xi_{g}"], sub[f"moment_{g}"]).statistic) if len(sub) >= 5 else None
    out["note"] = ("Jumlah eksposur dibatasi ukuran kelompok, sehingga ekor berujung hingga "
                   "(xi <= 0) secara teoretis mungkin. Bila Pickands dan Moment tidak searah, "
                   "kesimpulan berbasis xi dinyatakan tidak robust.")
    return out


def _pickands_asym_sd(xi: float, k: float) -> float:
    """SD asimtotik Pickands (Dekkers & de Haan, 1989):
    Var = xi^2 (2^(2xi+1) + 1) / [2 (2^xi - 1) ln 2]^2 / k; limit xi -> 0 = 3 / (4 (ln 2)^4) / k."""
    if xi is None or not np.isfinite(xi) or k <= 0:
        return np.nan
    ln2 = np.log(2.0)
    if abs(xi) < 1e-6:
        v = 3.0 / (4 * ln2 ** 4)
    else:
        v = xi ** 2 * (2 ** (2 * xi + 1) + 1) / (2 * (2 ** xi - 1) * ln2) ** 2
    return float(np.sqrt(v / k))


def evt_precision(cells: Cells, ksel: dict) -> dict:
    """Presisi indeks ekor (v3.3): SD bootstrap xi per gender, SD asimtotik Pickands pada k
    tengah rentang stabil, dan selisih terkecil yang dapat dideteksi (MDE, daya 80%,
    alpha 5% dua sisi): MDE = (1,96 + 0,84) x sqrt(SD_F^2 + SD_M^2)."""
    z = 1.959964 + 0.841621
    out = {}
    for m in C.MODELS:
        for a in C.ALPHAS:
            rows = []
            for seed in _seeds(cells):
                key = cells.key("warm", m, ksel[m], a, seed)
                reps = cells.reps.get(key, {})
                rF, rM = reps.get("xi_F"), reps.get("xi_M")
                xF = cells.point("warm", m, ksel[m], a, seed, "xi_F")
                xM = cells.point("warm", m, ksel[m], a, seed, "xi_M")
                nF = cells.point("warm", m, ksel[m], a, seed, "n_items_F")
                nM = cells.point("warm", m, ksel[m], a, seed, "n_items_M")
                if rF is None or rM is None or xF is None or xM is None:
                    continue
                sdF, sdM = float(np.nanstd(rF, ddof=1)), float(np.nanstd(rM, ddof=1))
                kF = 0.5 * (C.EVT_K_LO + C.EVT_K_HI) * (nF or 0)
                kM = 0.5 * (C.EVT_K_LO + C.EVT_K_HI) * (nM or 0)
                aF, aM = _pickands_asym_sd(xF, kF), _pickands_asym_sd(xM, kM)
                rows.append({"seed": seed, "xi_F": xF, "xi_M": xM, "boot_sd_F": sdF, "boot_sd_M": sdM,
                             "asym_sd_F": aF, "asym_sd_M": aM,
                             "mde_boot": z * np.sqrt(sdF ** 2 + sdM ** 2),
                             "mde_asym": z * np.sqrt(aF ** 2 + aM ** 2),
                             "observed_dxi": abs(xF - xM)})
            if rows:
                df = pd.DataFrame(rows)
                out[f"{m}|a{a}"] = {c: float(df[c].mean()) for c in df.columns if c != "seed"}
    return {"note": "Delta-xi yang teramati lebih kecil dari MDE berarti uji EVT tidak memiliki "
                    "daya untuk mendeteksi selisih sebesar itu; hasil 'tidak berbeda' tidak "
                    "boleh dibaca sebagai bukti kesetaraan ekor.", "results": out}


def confound(cells: Cells, ksel: dict, baselines: dict | None, strata: bool = True) -> dict:
    """F8 (v3.3): apakah gap model melampaui gap yang sudah dimiliki prediktor bias saja
    (BaselineOnly; Koren, 2010)? Selisih gap_model - gap_baseline pada fold, pengguna, dan
    bobot bootstrap yang sama, k terpilih, alpha = 0. Nilai positif = faktorisasi menambah
    disparitas di atas perbedaan yang sudah ada pada pola rating kelompok.
    Juga deskriptif: gap per kuartil jumlah rating latih (jalur warm), mengikuti Ekstrand
    dkk. (2018) yang menunjukkan ukuran profil merancukan perbedaan utilitas antar kelompok."""
    if not baselines:
        return {"available": False}
    labels = {"naik": "model menambah disparitas di atas baseline",
              "turun": "disparitas model lebih kecil dari baseline", "tidak berbeda": "tidak berbeda"}
    brel = {}
    for s_, b in baselines.items():
        brel[s_] = {}
        for path in ["warm", "cold"]:
            pt, rp = dict(b[path]["point"]), dict(b[path]["reps"])
            add_relative(pt, rp)
            brel[s_][path] = {"point": pt, "reps": rp}
    def build(seed, mets):
        b = brel.get(seed)
        tests = {}
        for m in C.MODELS:
            for path in ["warm", "cold"]:
                for met in mets:
                    a = cells.rep(path, m, ksel[m], 0, seed, _scen(path, 0), metric=met)
                    r = None if b is None else b[path]["reps"].get(met)
                    tests[f"{m}|{path}|{met}"] = _diff_test(a, r)
        return tests
    fam = _family(cells, lambda sd: build(sd, C.GAP_METRICS), labels, "F8")
    fam["relative"] = _family(cells, lambda sd: build(sd, [f"r{m}" for m in C.GAP_METRICS]),
                              labels, "F8-relatif")
    # titik gap baseline dan rasio
    pts = {}
    for path in ["warm", "cold"]:
        for met in C.GAP_METRICS + [f"r{m}" for m in C.GAP_METRICS] + ["mae"]:
            pts[f"baseline|{path}|{met}"] = S.across_seeds(
                [brel[s][path]["point"].get(met) for s in brel])
    fam["baseline_points"] = pts
    if not strata:
        fam["rule"] = "gap model dikurangi gap BaselineOnly (absolut dan relatif)"
        return fam
    # strata aktivitas (deskriptif)
    strata = {}
    for m in C.MODELS:
        acc = {}
        for seed, b in baselines.items():
            ex = cells.extra.get(cells.key("warm", m, ksel[m], 0, seed), {})
            pu = ex.get("per_user")
            bu = b["warm"]["extra"]["per_user"]
            if not pu:
                continue
            ntr = bu["n_train"].astype(float)
            q = pd.qcut(ntr, 4, labels=False, duplicates="drop")
            units = bu["units"]
            lab = _USERS_LAB.reindex(units) if _USERS_LAB is not None else None
            if lab is None:
                continue
            for src, arr in [("model", pu), ("baseline", bu)]:
                ae, n = np.asarray(arr["ae"], float), np.asarray(arr["n"], float)
                for qi in sorted(pd.unique(q)):
                    mq = q == qi
                    gF = mq & (lab.gender.to_numpy() == "F"); gM = mq & (lab.gender.to_numpy() == "M")
                    if gF.sum() and gM.sum():
                        gap = ae[gF].sum() / n[gF].sum() - ae[gM].sum() / n[gM].sum()
                        acc.setdefault(f"{src}|Q{int(qi) + 1}|gap_gender_signed", []).append(gap)
                    ag = lab.age_group.to_numpy()
                    ref = mq & (ag == C.AGE_REFERENCE)
                    if ref.sum():
                        mref = ae[ref].sum() / n[ref].sum()
                        vals = [ae[mq & (ag == g)].sum() / n[mq & (ag == g)].sum() - mref
                                for g in C.AGE_ORDER if g != C.AGE_REFERENCE and (mq & (ag == g)).sum() >= 10]
                        if vals:
                            acc.setdefault(f"{src}|Q{int(qi) + 1}|gap_age_ref", []).append(max(vals))
        strata[m] = {k: S.across_seeds(v)["mean"] for k, v in acc.items()}
    fam["activity_strata_descriptive"] = strata
    fam["rule"] = ("gap model melampaui gap BaselineOnly bila selisihnya > 0 signifikan (Holm) "
                   "pada >= 4 dari 5 fold; strata aktivitas = kuartil jumlah rating latih")
    return fam


_USERS_LAB = None   # diisi run_all dari tabel pengguna (gender, age_group, occ_group)


def completeness(cells: Cells, ksel: dict, baselines: dict | None = None) -> dict:
    present = {k for k, v in cells.reps.items()}
    seeds = list(C.SEEDS)
    exp = {"main": 0, "ablation": 0}
    miss = []
    def chk(key, bucket):
        exp[bucket] += 1
        if key not in present:
            miss.append("|".join(map(str, key)))
    pos = [a for a in C.ALPHAS if a > 0]
    for m in C.MODELS:
        for k in C.KS:
            for a in C.ALPHAS:
                for sd in seeds:
                    chk(cells.key("warm", m, k, a, sd), "main")
                    chk(cells.key("cold", m, k, a, sd, _scen("cold", a)), "main")
        for a in pos:
            for sd in seeds:
                for enc in C.ABLATION_ENCODINGS:
                    for path in C.ABLATION_PATHS:
                        chk(cells.key(path, m, ksel[m], a, sd, _scen(path, a), encoding=enc), "ablation")
                chk(cells.key("warm", m, ksel[m], a, sd, variant="gu"), "ablation")
    miss_base = [s for s in seeds if not baselines or s not in baselines]
    n_exp = exp["main"] + exp["ablation"]
    complete = len(miss) == 0 and not miss_base and not cells.diverged
    return {"expected_cell_rows": n_exp, "expected_main": exp["main"],
            "expected_ablation": exp["ablation"], "missing": len(miss),
            "missing_baseline_folds": miss_base, "diverged_cells": cells.diverged,
            "complete": complete, "missing_examples": miss[:30],
            "note": "putusan akhir hanya sah bila complete = true (tanpa sel hilang, baseline "
                    "lengkap, dan tanpa sel divergen)"}


def run_all(recs: list, tuning: dict, titles=None, baselines: dict | None = None,
            users: pd.DataFrame | None = None) -> dict:
    global _USERS_LAB
    if users is not None:
        _USERS_LAB = users.set_index("userId")[["gender", "age_group", "occ_group"]]
    cells = Cells(recs)
    ksel = {m: int(v) for m, v in tuning["k_selected"].items()}
    agg = aggregate(cells)
    res = {"n_cells": len(recs), "n_rows": len(cells.df), "k_selected": ksel,
           "seeds_available": _seeds(cells), "n_boot": C.N_BOOT, "n_boot_evt": C.N_BOOT_EVT,
           "table": cells.df.to_dict(orient="records"), "aggregate": agg,
           "pareto": pareto(cells), "h1": h1(cells, ksel), "alpha_effect": alpha_effect(cells, ksel),
           "safe_alpha": safe_alpha(cells, ksel), "new_users": new_users(cells, ksel),
           "new_user_profile": new_user_profile(cells, ksel),
           "amplification": amplification(cells, ksel), "mechanism": mechanism(cells, ksel),
           "encoding_effect": encoding_effect(cells, ksel),
           "h2": h2(cells), "leakage": leakage(cells, ksel),
           "occupation_effect": occupation_effect(cells, ksel),
           "confound": confound(cells, ksel, baselines),
           "evt": evt_diagnostics(cells, ksel), "evt_precision": evt_precision(cells, ksel),
           "evt_sensitivity": evt_sensitivity(cells, ksel), "evt_agreement": evt_agreement(cells),
           "completeness": completeness(cells, ksel, baselines),
           "popularity": popularity(agg), "evt_robustness": evt_robustness(cells, ksel),
           "tuning_robustness": tuning_robustness(cells, tuning)}
    res["pooled"] = pooled_analysis(cells, ksel, baselines)
    res["inference_summary"] = inference_summary(res, res["pooled"])
    res["analysis_version"] = C.ANALYSIS_VERSION
    if titles is not None:
        res["rankings"] = rankings(cells, ksel, titles)
    res["deviations"] = DEVIATIONS
    return res


# ==========================================================================
# v3.4: inferensi gabungan lintas fold (BBC-CV) dan ringkasan robust
# ==========================================================================
def _strip_missing(fam):
    """Buang keputusan yang tidak punya data (misalnya metrik non-MAE pada analisis gabungan)."""
    if isinstance(fam, dict) and "decisions" in fam:
        fam["decisions"] = {k: v for k, v in fam["decisions"].items()
                            if v.get("verdict") not in MISSING}
        fam.pop("per_seed", None)
        fam.pop("evt", None)
        for sub in ("ablation", "relative"):
            if sub in fam:
                _strip_missing(fam[sub])
    return fam


def pooled_analysis(cells: Cells, ksel: dict, baselines: dict | None) -> dict:
    from src import pooled as PO
    if _USERS_LAB is None:
        return {"available": False, "reason": "tabel pengguna tidak diberikan ke run_all"}
    pc, bp, info = PO.build(cells, baselines, _USERS_LAB, ksel)
    if pc is None:
        return {"available": False, "reason": info}
    out = {"available": True, "info": info,
           "h1": _strip_missing(h1(pc, ksel)), "alpha_effect": _strip_missing(alpha_effect(pc, ksel)),
           "safe_alpha": safe_alpha(pc, ksel), "new_users": _strip_missing(new_users(pc, ksel)),
           "new_user_profile": _strip_missing(new_user_profile(pc, ksel)),
           "encoding_effect": {k: _strip_missing(v) for k, v in encoding_effect(pc, ksel).items()},
           "occupation_effect": _strip_missing(occupation_effect(pc, ksel)),
           "confound": _strip_missing(confound(pc, ksel, bp, strata=False)),
           "aggregate": [{k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                         for r in pc.df.to_dict(orient="records")]}
    for r in out["safe_alpha"]["results"].values():
        for row in r["table"]:
            row.pop("per_seed", None)
    return out


def _sig_dir(dec):
    if not dec:
        return None
    v = dec.get("verdict")
    return {"naik": 1, "turun": -1}.get(v, 0) if v not in MISSING and v != "tidak terestimasi" else None


def inference_summary(fold_res: dict, pooled: dict) -> dict:
    """Menggabungkan putusan per fold (aturan protokol) dan putusan gabungan (BBC-CV).
    robust            : keduanya signifikan dengan arah sama
    gabungan saja     : hanya analisis gabungan yang signifikan
    per fold saja     : hanya aturan per fold yang signifikan
    tidak berbeda     : keduanya tidak signifikan
    bertentangan      : keduanya signifikan dengan arah berlawanan"""
    if not pooled.get("available"):
        return {"available": False}
    fams = {"F1": ("h1", None), "F2": ("alpha_effect", None), "F3": ("new_users", None),
            "F3b": ("new_user_profile", None), "F3b-ablation": ("new_user_profile", "ablation"),
            "F3b-relatif": ("new_user_profile", "relative"), "F7": ("occupation_effect", None),
            "F8": ("confound", None), "F8-relatif": ("confound", "relative")}
    for k in pooled.get("encoding_effect", {}):
        fams[f"F5b-{k}"] = ("encoding_effect", k)
    out, counts, tiers = {}, {}, {}
    for fam, (key, sub) in fams.items():
        if key == "encoding_effect":
            fobj = fold_res.get(key, {}).get(sub, {})
            pdec = pooled.get(key, {}).get(sub, {}).get("decisions", {})
        else:
            f0, p0 = fold_res.get(key, {}), pooled.get(key, {})
            if sub:
                f0, p0 = f0.get(sub, {}), p0.get(sub, {})
            fobj, pdec = f0, p0.get("decisions", {})
        fd = fobj.get("decisions", {})
        nb = _nadeau_bengio(fobj.get("per_seed", {}), list(pdec))
        rows = {}
        for name, pv in pdec.items():
            a, b = _sig_dir(fd.get(name)), _sig_dir(pv)
            if b is None:
                continue
            if a is None:
                lab = "gabungan " + ("naik" if b > 0 else "turun" if b < 0 else "tidak berbeda") + \
                      " (per fold tidak tersedia)"
            elif a and b:
                lab = "robust" if a == b else "bertentangan"
            elif b:
                lab = "gabungan saja"
            elif a:
                lab = "per fold saja"
            else:
                lab = "tidak berbeda"
            md = (pv.get("mean_diff") or pv.get("mean_diff_SVD_minus_NMF")
                  or pv.get("mean_diff_with_minus_without") or pv.get("mean_A_alpha_minus_A0") or {})
            n_ = nb.get(name, {})
            nb_dir = (int(np.sign(n_["mean"])) if n_.get("p_holm") is not None
                      and n_["p_holm"] < C.SIG_LEVEL else 0) if n_ else None
            if b and nb_dir == b:
                tier = "A (gabungan + Nadeau-Bengio)"
            elif b:
                tier = "B (gabungan saja)"
            elif a or nb_dir:
                tier = "C (tidak signifikan pada analisis gabungan)"
            else:
                tier = "-"
            rows[name] = {"status": lab, "tingkat_bukti": tier, "arah": b,
                          "selisih_gabungan": md.get("mean") if isinstance(md, dict) else None,
                          "putusan_per_fold": (fd.get(name) or {}).get("verdict"),
                          "putusan_gabungan": pv.get("verdict"),
                          "p_nadeau_bengio": n_.get("p"), "p_nadeau_bengio_holm": n_.get("p_holm")}
            counts.setdefault(fam, {}).setdefault(lab, 0)
            counts[fam][lab] += 1
            tiers.setdefault(fam, {}).setdefault(tier, 0)
            tiers[fam][tier] += 1
        out[fam] = rows
    return {"available": True, "primary": C.PRIMARY_INFERENCE, "counts": counts,
            "evidence_tiers": tiers, "tests": out,
            "rule": "status 'robust' bila analisis gabungan (BBC-CV) dan aturan per fold sama-sama "
                    "signifikan searah. Tingkat bukti A = analisis gabungan dan uji t terkoreksi "
                    "Nadeau-Bengio atas 5 estimasi fold (Holm per keluarga) sama-sama signifikan "
                    "searah; B = gabungan saja; C = hanya salah satu uji lain. Analisis gabungan "
                    "menangkap ketidakpastian sampling pengguna; Nadeau-Bengio menangkap variabilitas "
                    "akibat set latih (Nadeau & Bengio, 2003)."}


def _nadeau_bengio(per_seed: dict, names: list) -> dict:
    """Uji t terkoreksi Nadeau & Bengio (2003) atas estimasi per fold: varians antar fold
    dikalikan (1/J + n_uji/n_latih) = 1/5 + 1/4 untuk 5-fold, df = J - 1. Holm per keluarga."""
    res, ps = {}, {}
    for name in names:
        v = [t[name]["mean"] for t in per_seed.values()
             if t.get(name) and t[name].get("mean") is not None]
        v = np.asarray(v, float)
        if v.size < 3 or v.std(ddof=1) == 0:
            continue
        J = v.size
        se = np.sqrt((1 / J + 1 / (C.N_FOLDS - 1)) * v.var(ddof=1))
        t = v.mean() / se
        p = float(2 * stats.t.sf(abs(t), J - 1))
        res[name] = {"mean": float(v.mean()), "t": float(t), "p": p}
        ps[name] = p
    adj = S.holm(ps)
    for k in res:
        res[k]["p_holm"] = adj[k]
    return res


# ==========================================================================
# v3.4: diagnostik keandalan EVT dan ketahanan tuning
# ==========================================================================
def evt_robustness(cells: Cells, ksel: dict) -> dict:
    """Diagnostik dari jumlah eksposur tersimpan (jalur warm, k terpilih, semua alpha, fold).
    1. Kesesuaian tanda Pickands vs Moment pada rentang k yang sama (1%-5%) dan pada KEDALAMAN
       data yang sama: Pickands(k) memakai statistik terurut hingga X(4k), sehingga pembanding
       setara adalah Moment pada k' = 4k (rentang 4%-20%).
    2. Plateau: korelasi Spearman antara k dan Pickands(k) pada k = 0,5% sampai 20% n. Nilai
       mendekati +1 berarti estimasi terus berubah dengan kedalaman (tidak ada wilayah stabil),
       sehingga xi tidak teridentifikasi (Embrechts dkk., 1997; de Haan & Ferreira, 2006).
    3. Saturasi: porsi film dengan eksposur >= 90% ukuran kelompok, dan porsi eksposur yang
       diambil 1% film teratas. Ekor berujung hingga dan jenuh menjelaskan Moment negatif."""
    from src import evt as E
    rows = []
    gsize = _USERS_LAB.gender.value_counts().to_dict() if _USERS_LAB is not None else {}
    for m in C.MODELS:
        for a in C.ALPHAS:
            for seed in _seeds(cells):
                ex = cells.extra.get(cells.key("warm", m, ksel[m], a, seed), {})
                cnt = ex.get("curves", {}).get("exposure_counts", {}).get("gender", {})
                for g, arr in cnt.items():
                    x = np.sort(np.asarray(arr, float))[::-1]
                    n = x.size
                    if n < 50:
                        continue
                    st = E.stable(x)
                    md = E.stable(x, 4 * C.EVT_K_LO, 4 * C.EVT_K_HI)
                    ks = np.arange(max(1, int(np.ceil(0.005 * n))), min(int(0.2 * n), n // 4) + 1)
                    cv = E.curves(x, ks)
                    pk = np.array(cv["pickands"], float)
                    ok = np.isfinite(pk)
                    drift = float(stats.spearmanr(np.array(cv["k"])[ok], pk[ok]).statistic) \
                        if ok.sum() >= 10 else np.nan
                    G = gsize.get(g)
                    top = max(1, int(np.ceil(0.01 * n)))
                    rows.append({"model": m, "alpha": a, "seed": seed, "gender": g, "n_items": n,
                                 "pickands": st["pickands"], "moment_same_k": st["moment"],
                                 "moment_same_depth": md["moment"], "drift_spearman": drift,
                                 "share_saturated_items": float((x >= 0.9 * G).mean()) if G else None,
                                 "max_exposure_over_group": float(x[0] / G) if G else None,
                                 "share_exposure_top1pct": float(x[:top].sum() / x.sum())})
    if not rows:
        return {"available": False}
    df = pd.DataFrame(rows)
    def same(a, b):
        sub = df[[a, b]].dropna()
        return float((np.sign(sub[a]) == np.sign(sub[b])).mean()) if len(sub) else None
    summ = {"n": int(len(df)),
            "share_same_sign_same_k": same("pickands", "moment_same_k"),
            "share_same_sign_same_depth": same("pickands", "moment_same_depth"),
            "median_drift_spearman": float(df.drift_spearman.median()),
            "share_cells_drift_gt_0_5": float((df.drift_spearman > 0.5).mean()),
            "median_share_saturated_items": float(df.share_saturated_items.median())
            if df.share_saturated_items.notna().any() else None,
            "median_share_exposure_top1pct": float(df.share_exposure_top1pct.median())}
    robust = (summ["share_same_sign_same_depth"] or 0) >= 0.8 and summ["median_drift_spearman"] < 0.5
    summ["robust"] = bool(robust)
    reasons = []
    if (summ["share_same_sign_same_depth"] or 0) < 0.8:
        reasons.append("Pickands dan Moment tidak searah pada kedalaman data yang sama")
    if summ["median_drift_spearman"] >= 0.5:
        reasons.append("Pickands(k) berubah monoton dengan kedalaman k (tanpa wilayah stabil)")
    summ["verdict"] = ("indeks ekor stabil dan estimator sepakat" if robust else
                       "TIDAK ROBUST: " + "; ".join(reasons) + ". Delta-xi tidak dipakai untuk klaim "
                       "keadilan dan dilaporkan sebagai temuan diagnostik")
    per = df.groupby(["model", "alpha", "gender"])[[
        "pickands", "moment_same_k", "moment_same_depth", "drift_spearman",
        "share_saturated_items", "share_exposure_top1pct"]].mean().reset_index()
    return {"available": True, "summary": summ, "per_config": per.to_dict(orient="records")}


def tuning_robustness(cells: Cells, tuning: dict) -> dict:
    """Ketahanan kesimpulan terhadap tuning yang masih menyentuh batas grid:
    (1) selisih MAE validasi konfigurasi terbaik vs terbaik kedua pada k terpilih;
    (2) rentang gap (alpha = 0, jalur warm) lintas seluruh k grid utama. Rentang gap yang kecil
    berarti kapasitas/konvergensi tambahan hampir tidak mengubah kesimpulan disparitas."""
    out = {}
    rows = pd.DataFrame(tuning.get("rows", []))
    d = cells.df[(cells.df.path == "warm") & (cells.df.alpha == 0)]
    for m in C.MODELS:
        ks = int(tuning["k_selected"][m])
        r = rows[(rows.model == m) & (rows.k == ks)].sort_values("val_MAE") if len(rows) else rows
        g = d[d.model == m].groupby("k")[["mae"] + C.GAP_METRICS].mean()
        out[m] = {"k_selected": ks,
                  "val_MAE_best": float(r.val_MAE.iloc[0]) if len(r) else None,
                  "val_MAE_second": float(r.val_MAE.iloc[1]) if len(r) > 1 else None,
                  "gain_best_vs_second": float(r.val_MAE.iloc[1] - r.val_MAE.iloc[0]) if len(r) > 1 else None,
                  "diagnostic_k300": tuning.get("grid_edges", {}).get(f"{m}|diagnostik_k300"),
                  "gap_range_across_k": {met: float(g[met].max() - g[met].min()) for met in C.GAP_METRICS
                                         if met in g},
                  "gap_by_k": {int(k): {met: float(v[met]) for met in ["mae"] + C.GAP_METRICS}
                               for k, v in g.iterrows()}}
    return out


DEVIATIONS = [
    {"rule": "Inferensi gabungan lintas fold (v3.4, analisis saja)",
     "change": "Ditambahkan bootstrap pengguna atas prediksi out-of-fold gabungan (BBC-CV) untuk "
               "seluruh keluarga uji berbasis MAE, dengan satu matriks bobot untuk semua konfigurasi "
               "dan kedua jalur. Inferensi utama ditentukan config.PRIMARY_INFERENCE; putusan per fold "
               "dilaporkan sebagai uji stabilitas; klaim 'robust' bila keduanya signifikan searah.",
     "reason": "Pada jalur warm kelima fold berisi pengguna yang sama sehingga syarat 4 dari 5 fold "
               "tidak menambah informasi independen; pada jalur cold satu fold hanya sekitar 1.208 "
               "pengguna. Efek yang searah di 5 dari 5 fold gagal lolos (contoh: F8 gap gender, p per "
               "fold 0,0004 sampai 0,05). BBC-CV: Tsamardinos dkk. (2018). Ditambahkan setelah melihat "
               "hasil v3.3, sehingga kedua putusan dilaporkan berdampingan."},
    {"rule": "Uji t terkoreksi Nadeau-Bengio (v3.4b, analisis saja)",
     "change": "Setiap uji pada ringkasan inferensi diberi uji t terkoreksi atas 5 estimasi fold "
               "(faktor varians 1/5 + 1/4, df = 4, Holm per keluarga) dan tingkat bukti A/B/C.",
     "reason": "Bootstrap gabungan hanya menangkap ketidakpastian sampling pengguna; variabilitas akibat "
               "set latih ditangkap uji Nadeau & Bengio (2003). Klaim utama paper memakai tingkat A."},
    {"rule": "Gap relatif (v3.4, analisis saja)",
     "change": "Gap relatif = gap / MAE dihitung untuk titik dan replikat; keluarga F3b-relatif dan "
               "F8-relatif ditambahkan.",
     "reason": "Gap absolut ikut mengecil bila error keseluruhan turun; perbandingan model vs "
               "BaselineOnly (MAE 0,670 vs 0,719) dan profil vs Default memerlukan skala relatif."},
    {"rule": "Diagnostik keandalan EVT (v3.4, analisis saja)",
     "change": "Kesesuaian tanda Pickands vs Moment pada kedalaman data yang sama (Moment k' = 4k), "
               "uji plateau (Spearman k vs Pickands(k)), dan saturasi eksposur, dari jumlah eksposur "
               "tersimpan. Bila tidak robust, Delta-xi tidak dipakai untuk klaim keadilan.",
     "reason": "Pada v3.3 Pickands dan Moment hanya searah pada 15% (F) dan 30% (M) sel, dan Pickands "
               "naik monoton dengan kedalaman k (0,04 sampai 0,98), sehingga tidak ada wilayah stabil "
               "(Embrechts dkk., 1997; de Haan & Ferreira, 2006)."},
    {"rule": "Ketahanan tuning (v3.4, analisis saja)",
     "change": "Dilaporkan selisih MAE validasi terbaik vs terbaik kedua dan rentang gap lintas k.",
     "reason": "Optimum epoch NMF dan lr SVD masih di batas grid; perlu ditunjukkan bahwa kesimpulan "
               "disparitas tidak sensitif terhadap tuning lanjutan, tanpa melatih ulang."},
    {"rule": "Fold-in pengguna baru (v3.2)",
     "change": "Regularisasi fold-in dikalikan jumlah pseudo-observasi (lambda x n_obs), sesuai "
               "fungsi objektif pelatihan SGD Surprise/Koren dkk. (2009) yang mengenakan lambda per rating.",
     "reason": "v3.1 memakai lambda sekali saja sehingga vektor pengguna baru kurang diregularisasi; "
               "MAE kelompok Under 18 meledak (1,56) dan klaim pengguna baru menjadi artefak."},
    {"rule": "Rentang k EVT (v3.2)",
     "change": "Rentang k stabil diubah dari 5%-20% menjadi 1%-5% dari n; jumlah eksposur disimpan "
               "untuk uji sensitivitas rentang k; Moment menjadi uji ketahanan utama.",
     "reason": "Pada rentang lama Pickands memakai X(4k) hingga 80% data (bukan ekor); teori EVT "
               "mensyaratkan k/n -> 0 (de Haan & Ferreira, 2006). Pickands, Hill, Moment tidak sepakat "
               "pada v3.1 (Spearman -0,11 dan -0,13)."},
    {"rule": "Hipotesis Kedua (v3.2)",
     "change": "Uji utama memakai korelasi Spearman parsial dengan alpha dikontrol; korelasi per alpha "
               "dan korelasi gabungan lama menjadi deskriptif.",
     "reason": "alpha menaikkan MAE dan disparitas sekaligus sehingga korelasi lintas k x alpha "
               "terkacaukan oleh alpha."},
    {"rule": "Grid tuning (v3.2)",
     "change": "SVD lr {0,005; 0,01; 0,02} x reg {0,02; 0,05; 0,1} x epoch {40; 80}; NMF reg "
               "{0,02; 0,06; 0,15} x epoch {100; 200}; k = 200 ditambahkan sebagai diagnostik. k terpilih "
               "dibatasi pada grid utama {10, 20, 50, 100}; parameter di batas grid ditandai.",
     "reason": "Optimum v3.1 berada di batas grid (NMF k = 100 dan epoch 100; SVD epoch 40); k = 200 "
               "pada grid utama tidak layak secara komputasi."},
    {"rule": "Pelaporan data tidak lengkap (v3.2)",
     "change": "Putusan bernilai 'data belum tersedia/lengkap' bila sel yang diperlukan belum ada; "
               "analysis.completeness mencatat sel yang hilang.",
     "reason": "Pada v3.1 sel NMF k = 100 belum dijalankan tetapi putusan tertulis 'tidak berbeda'."},
    {"rule": "Resolusi bootstrap (v3.3)",
     "change": "B = 9999 replikat untuk metrik berbasis MAE dan amplifikasi; B = 999 (baris pertama "
               "matriks bobot yang sama) untuk indeks ekor. p bootstrap empiris tetap menjadi p utama.",
     "reason": "Pada v3.2 (B = 500) p terkecil = 2/501 = 0,004, sehingga keluarga dengan 24 sampai 64 "
               "uji per fold mustahil signifikan setelah Holm (p Holm terkecil 0,096 sampai 0,26) "
               "berapa pun besar efeknya. B harus memenuhi 2/(B+1) < 0,05/m (Davidson & MacKinnon, 2000)."},
    {"rule": "Keluarga uji EVT terpisah (v3.3)",
     "change": "Delta-xi dikeluarkan dari keluarga F1/F2 berbasis MAE dan diuji di keluarga F1-EVT dan "
               "F2-EVT. Putusan H1 ditentukan oleh 3 metrik gap jalur warm (>= 2 dari 3).",
     "reason": "Keluarga disusun per pertanyaan penelitian (Bender & Lange, 2001); presisi Pickands jauh "
               "lebih rendah daripada gap MAE sehingga mencampurnya melemahkan kedua jenis uji."},
    {"rule": "Bobot aman tanpa Delta-xi (v3.3)",
     "change": "Aturan non-inferioritas hanya memakai gap gender, usia, pekerjaan, dan MAE; Delta-xi "
               "dilaporkan deskriptif. Bobot aman dihitung per encoding (full, member5, membermean) "
               "dan per jalur.",
     "reason": "Margin 10% x xi(0) (sekitar 0,01) jauh di bawah SD Pickands (sekitar 0,25 pada k sekitar "
               "50; ragam asimtotik Dekkers & de Haan, 1989), sehingga uji non-inferioritas Delta-xi "
               "tidak memiliki daya dan selalu gagal."},
    {"rule": "Hill dikeluarkan sebagai pembanding (v3.3)",
     "change": "Kesesuaian estimator hanya Pickands vs Moment; Hill tetap tercatat sebagai titik "
               "deskriptif.",
     "reason": "Hill hanya konsisten untuk xi > 0 (Hill, 1975), sedangkan data menunjukkan ekor "
               "berujung hingga (Moment negatif pada 99% sel v3.2). Pickands dan Moment konsisten "
               "untuk semua tanda xi."},
    {"rule": "Presisi EVT (v3.3)",
     "change": "Ditambahkan analysis.evt_precision: SD bootstrap, SD asimtotik Pickands, dan selisih "
               "terkecil yang dapat dideteksi (MDE, daya 80%).",
     "reason": "Hasil 'tidak berbeda' pada Delta-xi hanya bermakna bila dibandingkan dengan MDE."},
    {"rule": "Uji pengguna baru berpasangan (v3.3)",
     "change": "F3 membandingkan pengguna fold cold-start sebagai pengguna baru (jalur cold) dengan "
               "pengguna yang sama sebagai pengguna lama (jalur warm, metrik sub_*), dengan bobot "
               "bootstrap yang sama. F3b baru: profil demografi vs Default pada model yang sama.",
     "reason": "v3.2 membandingkan dua populasi berbeda dengan bobot bootstrap independen. Pembanding "
               "alami untuk pengguna baru adalah rekomendasi Default (tanpa demografi)."},
    {"rule": "Ablation encoding pada pengguna baru (v3.3)",
     "change": "Encoding member5 dan membermean juga dijalankan pada jalur cold. Nilai pseudo-item "
               "membermean untuk pengguna baru = rata-rata dari rata-rata rating pengguna latih. "
               "Ditambahkan F5b: hanya-anggota vs full pada alpha yang sama.",
     "reason": "v3.2 menunjukkan encoding hanya-anggota hampir netral pada pengguna lama; klaim "
               "bobot aman untuk pengguna baru memerlukan sel cold."},
    {"rule": "Ablation NMF dengan bias dihapus (v3.3)",
     "change": "Varian NMF dengan bias tidak dijalankan.",
     "reason": "Pada v3.2 implementasi Surprise divergen (MAE 2,008 dan 1,089 pada 2 dari 25 sel) dan "
               "P@10 sekitar 0,003; dokumentasi Surprise menyebut varian ini sangat rawan overfit. "
               "Bukti mekanisme diperoleh dari ablation encoding."},
    {"rule": "Pembanding BaselineOnly (v3.3)",
     "change": "Prediktor bias saja (Koren, 2010) per fold; keluarga F8 menguji gap model dikurangi "
               "gap baseline; gap per kuartil jumlah rating latih dilaporkan deskriptif.",
     "reason": "Sebagian gap kelompok berasal dari pola rating (variasi rating, ukuran profil), bukan "
               "dari faktorisasi (Ekstrand dkk., 2018)."},
    {"rule": "Hipotesis Kedua deskriptif (v3.3)",
     "change": "Korelasi Spearman parsial dilaporkan sebagai arah (trade-off, searah, campuran), "
               "bukan putusan konfirmatori.",
     "reason": "Titik konfigurasi k x alpha memakai data dan pengguna yang sama sehingga tidak saling "
               "bebas; nilai p terlalu optimistis."},
    {"rule": "Grid k dan tuning (v3.3)",
     "change": "k = 200 masuk grid utama; grid tuning SVD lr {0,0025; 0,005; 0,01} x epoch {40; 80; "
               "160} x reg {0,02; 0,05; 0,1}; NMF epoch {200; 300; 400} x reg {0,02; 0,06; 0,15}; "
               "k = 300 dilatih sekali sebagai diagnostik batas grid k.",
     "reason": "Pada v3.2 MAE validasi NMF terus membaik hingga k = 200 dan seluruh optimum epoch NMF "
               "berada di batas grid."},
    {"rule": "Integritas cache (v3.3)",
     "change": "Nama berkas sel memuat sidik hyperparameter; analisis hanya membaca sel dengan "
               "hyperparameter aktif; penulisan berkas atomik; sel alpha = 0 dipakai bersama oleh "
               "semua encoding/variant.",
     "reason": "Pada v3.2 perubahan hasil tuning tidak terdeteksi oleh cache sehingga sel lama dapat "
               "terpakai ulang secara diam-diam."},
    {"rule": "Penanda divergensi (v3.3)",
     "change": "Sel dengan MAE latih > MAE prediktor rata-rata global pada sampel yang sama ditandai "
               "divergen, replikatnya tidak dipakai uji, dan completeness = false.",
     "reason": "Model divergen menghasilkan metrik yang tidak bermakna."},
    {"rule": "Skema validasi",
     "change": "Opsi A (5 kali pengulangan acak 80:20) diganti 5-fold: rating per pengguna dibagi "
               "ke 5 fold; pengguna dipartisi ke 5 fold cold-start (stratified gender x usia). "
               "Kolom 'seed' pada JSON berarti indeks fold (0 sampai 4). Kontrol baseline tetap "
               "memakai split lama 80:20 seed 42.",
     "reason": "Arahan pembimbing; sesuai proposal; data uji antar fold saling lepas."},
    {"rule": "Atribut input",
     "change": "Kelompok pekerjaan (5 kelompok) ditambahkan sebagai atribut input ketiga; "
               "kategori input pengguna baru menjadi 8; metrik disparitas utama menjadi 4 "
               "(ditambah gap kelompok pekerjaan, tanpa kelompok tak terklasifikasi); keluarga "
               "uji F1 sampai F6 membesar; ablation input gender + usia ditambahkan (F7).",
     "reason": "Arahan pembimbing. Kelompok tak terklasifikasi tetap menjadi nilai input tetapi "
               "dikeluarkan dari metrik disparitas karena isinya heterogen."},
    {"rule": "Uji kebocoran pekerjaan",
     "change": "Hanya diuji pada alpha = 0 dan ablation input gender + usia.",
     "reason": "Bila pekerjaan menjadi input, probe pekerjaan pasti tinggi sehingga uji tidak bermakna."},
    {"rule": "Putusan H1",
     "change": "Kategori tambahan: bila tidak ada metrik yang berbeda signifikan, putusan "
               "adalah 'tidak didukung (tidak ada perbedaan signifikan)', bukan 'didukung sebagian'.",
     "reason": "Kategori 'kondisi lain' pada protokol tidak membedakan kasus tanpa perbedaan "
               "dari kasus bukti campuran."},
    {"rule": "Klaim pengguna baru (F3)",
     "change": "Uji utama memakai selisih excess gap relatif (cold - warm) > 0. Keluarga F3 mencakup "
               "semua alpha > 0, bukan satu alpha.",
     "reason": "Protokol tidak menetapkan satu alpha untuk F3."},
    {"rule": "Bootstrap excess gap",
     "change": "Rata-rata gap permutasi dihitung sekali pada data asli lalu dikurangkan dari "
               "setiap replikat bootstrap.",
     "reason": "Permutasi di dalam setiap replikat bootstrap (9.999 x 1.000) terlalu mahal."},
    {"rule": "Kriteria mekanisme lintas alpha",
     "change": "Mekanisme dinyatakan terbukti bila kriteria cosinus SVD terpenuhi pada >= 3 "
               "dari 4 nilai alpha > 0 di setiap varian ablation.",
     "reason": "Protokol tidak menetapkan alpha untuk F5."},
    {"rule": "Bootstrap berpasangan",
     "change": "Resampling pengguna dilakukan pada seluruh pengguna jalur (bukan per kelompok), "
               "dengan indeks identik untuk semua konfigurasi dalam satu seed.",
     "reason": "Agar satu set replikat melayani semua atribut dan semua perbandingan "
               "antar konfigurasi secara berpasangan."},
]
