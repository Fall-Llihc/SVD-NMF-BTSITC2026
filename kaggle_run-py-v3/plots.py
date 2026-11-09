"""
plots.py
========
PERAN FILE
    Seluruh grafik Opsi C v3. Setiap fungsi menerima hasil analysis.run_all (dict)
    atau tabel tuning, menyimpan PNG ke outputs/figures/, dan mengembalikan path.
    Fungsi aman terhadap data yang belum lengkap (misalnya tahap k = 100 belum jalan).

DAFTAR GRAFIK
    01 tuning         : MAE validasi terbaik per k
    02 akurasi        : MAE vs alpha per k (warm), rata-rata +- sd antar seed
    03 disparitas     : gap gender dan gap usia vs alpha pada k terpilih, warm dan cold
    04 pareto         : front Pareto warm/cold x sumbu gender/usia
    05 pickands plot  : xi(k) Pickands dan Moment untuk F dan M (fold 0)
    06 delta xi       : |xi F - xi M| vs alpha
    07 mekanisme      : cos(F, M) per encoding x alpha
    08 pengguna baru  : excess gap relatif cold vs warm
    09 amplifikasi    : A(alpha) - A(0)
    10 kebocoran      : probe kelompok pekerjaan laten vs baseline
    11 efek pekerjaan : selisih dengan vs tanpa pekerjaan sebagai input
    12 encoding       : gap dan amplifikasi per encoding (full vs hanya-anggota), warm dan cold
    13 profil cold    : profil demografi dikurangi Default pada pengguna baru
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402

COL = {"SVD": "#1f6fb2", "NMF": "#c0661a"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, name):
    C.ensure_dirs()
    p = C.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _agg(res):
    return pd.DataFrame(res["aggregate"])


def fig_tuning(tuning):
    df = pd.DataFrame(tuning["rows"])
    best = df.groupby(["model", "k"]).val_MAE.min().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m, g in best.groupby("model"):
        ax.plot(g.k, g.val_MAE, "o-", color=COL[m], label=m)
        ks = tuning["k_selected"][m]
        ax.scatter([ks], g[g.k == ks].val_MAE, s=120, facecolor="none", edgecolor=COL[m], lw=2)
    ax.set_xscale("log"); ax.set_xticks(C.KS); ax.set_xticklabels(C.KS)
    ax.set_xlabel("k (faktor laten)"); ax.set_ylabel("MAE validasi terbaik")
    ax.set_title("Tuning per k (lingkaran = k terpilih)"); ax.legend()
    return _save(fig, "01_tuning.png")


def fig_accuracy(res):
    a = _agg(res)
    a = a[(a.path == "warm") & (a.encoding == "full") & (a.variant == "-")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, m in zip(axes, C.MODELS):
        for k, g in a[a.model == m].groupby("k"):
            g = g.sort_values("alpha")
            ax.errorbar(range(len(g)), g.mae, yerr=g.mae_sd, marker="o", capsize=3, label=f"k={k}")
            ax.set_xticks(range(len(g))); ax.set_xticklabels(g.alpha)
        ax.set_title(f"{m}: MAE warm vs alpha"); ax.set_xlabel("alpha"); ax.set_ylabel("MAE")
        ax.legend(fontsize=7)
    return _save(fig, "02_akurasi_alpha.png")


def fig_disparity(res):
    a = _agg(res)
    ks = res["k_selected"]
    mets = ["gap_gender", "gap_age_ref", "gap_occ"]
    fig, axes = plt.subplots(len(mets), 2, figsize=(9, 3 * len(mets)))
    for j, path in enumerate(["warm", "cold"]):
        sub = a[(a.path == path) & (a.encoding == "full") & (a.variant == "-")]
        if path == "cold":
            sub = sub[((sub.alpha == 0) & (sub.scenario == "default")) |
                      ((sub.alpha > 0) & (sub.scenario == C.COLD_PRIMARY_SCENARIO))]
        for i, met in enumerate(mets):
            ax = axes[i, j]
            for m in C.MODELS:
                g = sub[(sub.model == m) & (sub.k == ks[m])].sort_values("alpha")
                if g.empty:
                    continue
                ax.errorbar(range(len(g)), g[met], yerr=g[f"{met}_sd"], marker="o", capsize=3,
                            color=COL[m], label=f"{m} (k={ks[m]})")
                ax.set_xticks(range(len(g))); ax.set_xticklabels(g.alpha)
            ax.set_title(f"{path}: {C.DISPARITY_LABELS[met]}"); ax.set_xlabel("alpha")
            ax.legend(fontsize=7)
    return _save(fig, "03_disparitas_alpha.png")


def fig_pareto(res):
    P = res["pareto"]
    axn = ["MAE_vs_gap_gender", "MAE_vs_gap_age_ref", "MAE_vs_gap_occ"]
    fig, axes = plt.subplots(len(axn), 2, figsize=(10, 3.7 * len(axn)))
    for j, path in enumerate(["warm", "cold"]):
        for i, ax_name in enumerate(axn):
            ax = axes[i, j]
            v = P.get(path, {}).get(ax_name)
            if not v:
                ax.set_visible(False); continue
            for p in v["points"]:
                ax.errorbar(p["mae"], p["disp"], xerr=p["mae_sd"], yerr=p["disp_sd"],
                            fmt="o" if p["on_front"] else ".", color=COL[p["model"]],
                            alpha=1 if p["on_front"] else .35, ms=7 if p["on_front"] else 5,
                            capsize=2, lw=.8)
            fr = v["front"]
            ax.plot([p["mae"] for p in fr], [p["disp"] for p in fr], "k--", lw=1)
            for p in fr:
                ax.annotate(f"{p['model'][0]} k{p['k']} a{p['alpha']}", (p["mae"], p["disp"]),
                            fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("MAE (rata-rata 5 fold)")
            ax.set_ylabel(ax_name.replace("MAE_vs_", ""))
            ax.set_title(f"Pareto {path}: biru SVD, oranye NMF; tebal = front")
    return _save(fig, "04_pareto.png")


def fig_pickands(res):
    cur = res["evt"]["curves"]
    if not cur:
        return None
    keys = sorted(cur)
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.6 * nrow), squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    for ax, key in zip(axes.ravel(), keys):
        for grp, ls in [("F", "-"), ("M", "--")]:
            c = cur[key].get(f"curve_{grp}")
            if not c or not c["k"]:
                continue
            k = np.array(c["k"]); n = c["n"]
            ax.plot(k, c["pickands"], ls, color="k", lw=1, label=f"Pickands {grp}")
            ax.plot(k, c["moment"], ls, color="tab:purple", lw=.8, label=f"Moment {grp}")
            ax.axvspan(C.EVT_K_LO * n, C.EVT_K_HI * n, color="grey", alpha=.08)
        ax.set_title(f"{key} (fold 0); area abu = rentang k stabil")
        ax.set_xlabel("k"); ax.set_ylabel("indeks ekor"); ax.set_ylim(-3, 4)
        ax.legend(fontsize=6, ncol=2)
    return _save(fig, "05_pickands_plot.png")


def fig_dxi(res):
    a = _agg(res)
    ks = res["k_selected"]
    sub = a[(a.path == "warm") & (a.encoding == "full") & (a.variant == "-")]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in C.MODELS:
        g = sub[(sub.model == m) & (sub.k == ks[m])].sort_values("alpha")
        if g.empty or "dxi_gender" not in g:
            continue
        ax.errorbar(range(len(g)), g.dxi_gender, yerr=g.dxi_gender_sd, marker="o", capsize=3,
                    color=COL[m], label=m)
        ax.set_xticks(range(len(g))); ax.set_xticklabels(g.alpha)
    ax.set_title("|xi F - xi M| Pickands Top-100 (warm)"); ax.set_xlabel("alpha"); ax.legend()
    return _save(fig, "06_delta_xi.png")


def fig_mechanism(res):
    cos = res["mechanism"]["cos_F_M"]
    if not cos:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    encs = ["full"] + C.ABLATION_ENCODINGS
    for ax, m in zip(axes, C.MODELS):
        for e in encs:
            xs, ys, lo, hi = [], [], [], []
            for a in [x for x in C.ALPHAS if x > 0]:
                v = cos.get(f"{m}|{e}|a{a}")
                if v and v["mean"] is not None:
                    xs.append(a); ys.append(v["mean"])
                    c = v["ci95_t"]
                    lo.append(v["mean"] - (c[0] if c[0] is not None else v["mean"]))
                    hi.append((c[1] if c[1] is not None else v["mean"]) - v["mean"])
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=e)
        ax.axhline(C.MECH_COS_SVD_MAX, color="r", ls=":", lw=1)
        ax.set_title(f"{m}: cos(F, M) pseudo-item"); ax.set_xlabel("alpha"); ax.legend(fontsize=7)
    return _save(fig, "07_mekanisme_cos.png")


def fig_new_users(res):
    dec = res["new_users"]["decisions"]
    if not dec:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, attr in zip(axes, ["gender", "age", "occ"]):
        for m in C.MODELS:
            xs, c, w = [], [], []
            for a in [x for x in C.ALPHAS if x > 0]:
                v = dec.get(f"{m}|a{a}|{attr}")
                if v and v["mean_relx_cold"]["mean"] is not None:
                    xs.append(a); c.append(v["mean_relx_cold"]["mean"])
                    w.append(v["mean_relx_warm_same_users"]["mean"])
            if xs:
                ax.plot(xs, c, "o-", color=COL[m], label=f"{m} cold")
                ax.plot(xs, w, "s--", color=COL[m], alpha=.6, label=f"{m} warm")
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"Excess gap relatif ({attr})"); ax.set_xlabel("alpha"); ax.legend(fontsize=7)
    return _save(fig, "08_pengguna_baru.png")


def fig_amplification(res):
    dec = res["amplification"]["decisions"]
    if not dec:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m in C.MODELS:
        xs, ys, e = [], [], []
        for a in [x for x in C.ALPHAS if x > 0]:
            v = dec.get(f"{m}|a{a}")
            if v and v["mean_A_alpha_minus_A0"]["mean"] is not None:
                xs.append(a); ys.append(v["mean_A_alpha_minus_A0"]["mean"])
                e.append(v["mean_A_alpha_minus_A0"]["sd"] or 0)
        if xs:
            ax.errorbar(xs, ys, yerr=e, marker="o", capsize=3, color=COL[m], label=m)
    ax.axhline(0, color="k", lw=.6)
    ax.set_title("A(alpha) - A(0), gender, warm"); ax.set_xlabel("alpha"); ax.legend()
    return _save(fig, "09_amplifikasi.png")


def fig_leakage(res):
    lk = res["leakage"]["results"]
    if not lk:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for m in C.MODELS:
        xs, lat, base = [], [], []
        for a in C.ALPHAS:
            key = f"{m}|utama|a0" if a == 0 else f"{m}|input_gender_usia|a{a}"
            v = lk.get(key)
            if v:
                xs.append(a); lat.append(v["latent_bal_acc"]["mean"]); base.append(v["baseline_bal_acc"]["mean"])
        if xs:
            ax.plot(xs, lat, "o-", color=COL[m], label=f"{m} laten")
            ax.plot(xs, base, "s:", color=COL[m], alpha=.6, label=f"{m} baseline (gender+usia)")
    ax.set_title("Probe kelompok pekerjaan (alpha > 0: input gender + usia saja)")
    ax.set_xlabel("alpha"); ax.set_ylabel("balanced accuracy"); ax.legend(fontsize=7)
    return _save(fig, "10_kebocoran_pekerjaan.png")


def fig_occupation_effect(res):
    dec = res.get("occupation_effect", {}).get("decisions", {})
    if not dec:
        return None
    mets = ["mae"] + C.GAP_METRICS
    fig, axes = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 3.2))
    for ax, met in zip(axes, mets):
        for m in C.MODELS:
            xs, ys, e = [], [], []
            for a in [x for x in C.ALPHAS if x > 0]:
                v = dec.get(f"{m}|a{a}|{met}")
                if v and v["mean_diff_with_minus_without"]["mean"] is not None:
                    xs.append(a); ys.append(v["mean_diff_with_minus_without"]["mean"])
                    e.append(v["mean_diff_with_minus_without"]["sd"] or 0)
            if xs:
                ax.errorbar(xs, ys, yerr=e, marker="o", capsize=3, color=COL[m], label=m)
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{met}: dengan - tanpa pekerjaan", fontsize=8)
        ax.set_xlabel("alpha")
    axes[0].legend(fontsize=7)
    return _save(fig, "11_efek_pekerjaan.png")


def fig_encoding(res):
    a = _agg(res)
    ks = res["k_selected"]
    mets = ["mae", "gap_gender", "gap_age_ref", "gap_occ", "amp_gender"]
    encs = ["full"] + C.ABLATION_ENCODINGS
    sty = {"full": "-", "member5": "--", "membermean": ":"}
    fig, axes = plt.subplots(2, len(mets), figsize=(3.0 * len(mets), 5.6))
    for i, path in enumerate(["warm", "cold"]):
        for j, met in enumerate(mets):
            ax = axes[i, j]
            if met not in a or (path == "cold" and met == "amp_gender"):
                ax.axis("off"); continue
            for m in C.MODELS:
                for enc in encs:
                    sub = a[(a.path == path) & (a.model == m) & (a.k == ks[m]) & (a.variant == "-")]
                    if path == "cold":
                        sub = sub[((sub.alpha == 0) & (sub.scenario == "default")) |
                                  ((sub.alpha > 0) & (sub.scenario == C.COLD_PRIMARY_SCENARIO))]
                    sub = sub[(sub.alpha == 0) | (sub.encoding == enc)].sort_values("alpha")
                    if sub.alpha.nunique() < 2:
                        continue
                    ax.plot(sub.alpha, sub[met], sty[enc], marker="o", ms=3, color=COL[m],
                            label=f"{m} {enc}")
            ax.set_title(f"{path}: {met}", fontsize=8); ax.set_xlabel("alpha")
    axes[0, 0].legend(fontsize=6)
    return _save(fig, "12_encoding.png")


def fig_cold_profile(res):
    fam = res.get("new_user_profile", {})
    dec = fam.get("decisions", {})
    if not dec:
        return None
    mets = ["mae"] + C.GAP_METRICS
    fig, axes = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 3.2))
    for ax, met in zip(axes, mets):
        for m in C.MODELS:
            xs, ys, e = [], [], []
            for a in [x for x in C.ALPHAS if x > 0]:
                v = dec.get(f"{m}|a{a}|{met}")
                if v and v["mean_diff"]["mean"] is not None:
                    xs.append(a); ys.append(v["mean_diff"]["mean"]); e.append(v["mean_diff"]["sd"] or 0)
            if xs:
                ax.errorbar(xs, ys, yerr=e, marker="o", capsize=3, color=COL[m], label=m)
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{met}: profil - Default (cold)", fontsize=8)
        ax.set_xlabel("alpha")
    axes[0].legend(fontsize=7)
    return _save(fig, "13_profil_vs_default_cold.png")


def all_figures(res, tuning) -> list:
    fns = [lambda: fig_tuning(tuning), lambda: fig_accuracy(res), lambda: fig_disparity(res),
           lambda: fig_pareto(res), lambda: fig_pickands(res), lambda: fig_dxi(res),
           lambda: fig_mechanism(res), lambda: fig_new_users(res),
           lambda: fig_amplification(res), lambda: fig_leakage(res),
           lambda: fig_occupation_effect(res), lambda: fig_encoding(res),
           lambda: fig_cold_profile(res)]
    out = []
    for f in fns:
        try:
            p = f()
            if p:
                out.append(str(p))
        except Exception as e:  # grafik tidak boleh menghentikan analisis
            out.append(f"GAGAL: {e!r}")
    return out
