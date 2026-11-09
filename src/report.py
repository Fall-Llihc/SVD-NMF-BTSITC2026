"""
report.py
=========
PERAN FILE
    Menyusun ringkasan otomatis (insight) dari hasil analysis.run_all dalam bentuk
    kalimat singkat berbahasa Indonesia. Setiap kalimat langsung menyebut putusan dan
    angka kuncinya, sehingga notebook dan JSON memuat bacaan yang sama.
"""
from __future__ import annotations

from src import config as C


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}".replace(".", ",")


def insights(res: dict, tuning: dict, control: dict) -> list:
    out = []
    comp = res.get("completeness", {})
    if comp and not comp.get("complete"):
        out.append(f"PERINGATAN: {comp['missing']} dari {comp['expected_cell_rows']} sel belum ada di cache, "
                   f"baseline belum ada untuk fold {comp.get('missing_baseline_folds')}, sel divergen "
                   f"{len(comp.get('diverged_cells', []))}; putusan terkait belum sah.")
    out.append(f"Bootstrap: B = {res.get('n_boot')} (metrik MAE), {res.get('n_boot_evt')} (indeks ekor).")
    ks = res["k_selected"]
    po = res.get("pooled", {})
    summ = res.get("inference_summary", {})
    out.append(f"Versi analisis {res.get('analysis_version')}; inferensi utama: "
               + ("gabungan lintas fold (BBC-CV)" if C.PRIMARY_INFERENCE == "pooled" else "aturan per fold")
               + "; status 'robust' = gabungan dan per fold sama-sama signifikan searah.")
    if po.get("available"):
        out.append(f"H1 (gabungan lintas fold): {po['h1']['verdict']}. Rincian: " + "; ".join(
            f"{k}: {v.get('label', v['verdict'])} (selisih SVD-NMF {_f((v.get('mean_diff_SVD_minus_NMF') or {}).get('mean'))})"
            for k, v in po["h1"]["decisions"].items()) + ".")
        for k, v in po["safe_alpha"]["results"].items():
            out.append(f"Bobot aman (gabungan) {k}: {v['statement']}.")
    if summ.get("available"):
        for fam, c in summ["counts"].items():
            out.append(f"Ringkasan {fam}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + ".")
        for fam, c in summ.get("evidence_tiers", {}).items():
            out.append(f"Tingkat bukti {fam}: " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) + ".")
        rob = [f"{fam} {n}" for fam, t in summ["tests"].items() for n, v in t.items()
               if v["status"] in ("robust", "gabungan saja")]
        if rob:
            out.append("Uji signifikan pada analisis gabungan: " + "; ".join(rob[:40]) + ".")
    er = res.get("evt_robustness", {}).get("summary")
    if er:
        out.append(f"Keandalan EVT: {er['verdict']} (tanda Pickands = Moment pada kedalaman sama "
                   f"{_f(er['share_same_sign_same_depth'], 2)}, pada k sama {_f(er['share_same_sign_same_k'], 2)}; "
                   f"median Spearman k vs Pickands(k) {_f(er['median_drift_spearman'], 2)}; median porsi "
                   f"eksposur 1% film teratas {_f(er['median_share_exposure_top1pct'], 3)}).")
    tr = res.get("tuning_robustness", {})
    for m, v in tr.items():
        rg = ", ".join(f"{k} {_f(x)}" for k, x in v.get("gap_range_across_k", {}).items())
        out.append(f"Ketahanan tuning {m}: selisih MAE validasi terbaik vs terbaik kedua "
                   f"{_f(v.get('gain_best_vs_second'), 5)}; rentang gap lintas k (alpha = 0): {rg}.")
    out.append(f"k terpilih dari tuning (MAE validasi terbaik pada alpha = 0): SVD k = {ks['SVD']}, "
               f"NMF k = {ks['NMF']}.")
    if control.get("available"):
        s = "identik" if control.get("identical") else "TIDAK identik"
        out.append(f"Kontrol reprodusibilitas dengan hyperparameter v2 {s} dengan baseline "
                   f"(selisih MAE SVD {_f(control['models']['SVD'].get('abs_diff_MAE'), 6)}, "
                   f"NMF {_f(control['models']['NMF'].get('abs_diff_MAE'), 6)}).")
    h1 = res["h1"]
    det = "; ".join(f"{k}: {v['label']} ({v['n_sig_positive']}/{v['n_sig_negative']} fold)"
                    for k, v in h1["decisions"].items())
    out.append(f"Hipotesis Pertama: {h1['verdict']}. Rincian (fold signifikan SVD>NMF / NMF>SVD): {det}.")
    he = h1.get("evt", {}).get("decisions", {})
    if he:
        out.append("F1-EVT (Delta-xi, terpisah): " + "; ".join(f"{k}: {v['label']}" for k, v in he.items()) + ".")
    out.append("Hipotesis Kedua (deskriptif): " + "; ".join(
        f"{m} {v}" for m, v in res["h2"].get("direction", {}).items()) + ".")
    ae = res.get("alpha_effect", {}).get("decisions", {})
    if ae:
        ch = [f"{k}: {v['label']}" for k, v in ae.items() if v["verdict"] in ("naik", "turun")]
        out.append(f"Efek alpha (F2): {len(ch)} dari {len(ae)} uji berubah signifikan"
                   + (f" ({'; '.join(ch[:16])})." if ch else "."))
    for k, v in res["safe_alpha"]["results"].items():
        out.append(f"Bobot aman {k}: {v['statement']}.")
    for m in ["SVD", "NMF"]:
        d = tuning.get("grid_edges", {}).get(f"{m}|diagnostik_k300")
        if isinstance(d, dict):
            out.append(f"Diagnostik k = 300 {m}: MAE validasi {_f(d['val_MAE'])}, selisih vs k = 200 "
                       f"{_f(d['selisih_vs_k_maks_grid'], 5)}" +
                       (" (k = 200 masih di batas grid)." if d["k_maks_grid_di_batas"] else "."))
    np_ = res.get("new_user_profile", {}).get("decisions", {})
    if np_:
        ch = [f"{k}: {v['label']}" for k, v in np_.items() if v["verdict"] in ("naik", "turun")]
        out.append(f"Profil demografi vs Default pada pengguna baru (F3b): {len(ch)} dari {len(np_)} uji "
                   "signifikan" + (f" ({'; '.join(ch[:16])})." if ch else "."))
    for k, fam in res.get("encoding_effect", {}).items():
        dec = fam.get("decisions", {})
        ch = [f"{n}: {v['label']}" for n, v in dec.items() if v["verdict"] in ("naik", "turun")]
        out.append(f"Encoding {k} vs full (F5b): {len(ch)} dari {len(dec)} uji signifikan"
                   + (f" ({'; '.join(ch[:12])})." if ch else "."))
    cf = res.get("confound", {})
    if cf.get("decisions"):
        ch = [f"{k}: {v['label']}" for k, v in cf["decisions"].items() if v["verdict"] in ("naik", "turun")]
        out.append(f"Gap model vs BaselineOnly (F8): {len(ch)} dari {len(cf['decisions'])} uji signifikan"
                   + (f" ({'; '.join(ch)})." if ch else "."))
    nu = res["new_users"]["decisions"]
    valid = [k for k, v in nu.items() if v["claim_valid"]]
    miss = sum(1 for v in nu.values() if v["verdict"] in ("data belum tersedia", "data belum lengkap"))
    out.append(f"Klaim pengguna baru menanggung bias lebih besar sah pada {len(valid)} dari "
               f"{len(nu)} uji" + (f" ({', '.join(valid)})" if valid else "") +
               (f"; {miss} uji belum punya data." if miss else "."))
    am = res["amplification"]["decisions"]
    valid = [k for k, v in am.items() if v["claim_valid"]]
    miss = sum(1 for v in am.values() if v["verdict"] in ("data belum tersedia", "data belum lengkap"))
    out.append(f"Amplifikasi perbedaan genre (A(alpha) - A(0) > 0) sah pada {len(valid)} dari "
               f"{len(am)} uji" + (f" ({', '.join(valid)})" if valid else "") +
               (f"; {miss} uji belum punya data." if miss else "."))
    me = res["mechanism"]
    out.append("Mekanisme nilai negatif: " + me.get("mechanism_status", str(me["mechanism_supported"]))
               + " lintas encoding ablation (" + "; ".join(
                   f"{e}: {v['n_alpha_hold']}/{v['n_alpha']} alpha" for e, v in me["verdict_per_encoding"].items())
               + ").")
    oe = res.get("occupation_effect", {}).get("decisions", {})
    if oe:
        ch = [f"{k}: {v['label']}" for k, v in oe.items() if v["verdict"] in ("naik", "turun")]
        out.append(f"Efek menambah pekerjaan sebagai input (F7): {len(ch)} dari {len(oe)} uji berubah "
                   "signifikan" + (f" ({'; '.join(ch[:12])})." if ch else "."))
    lk = res["leakage"]["results"]
    leaks = [k for k, v in lk.items() if v["leak"]]
    out.append(f"Kebocoran kelompok pekerjaan (tanpa pekerjaan sebagai input) melampaui baseline probe pada {len(leaks)} dari {len(lk)} konfigurasi"
               + (f" ({', '.join(leaks)})." if leaks else "."))
    for path, axes in res["pareto"].items():
        for ax, v in axes.items():
            out.append(f"Pareto {path} {ax}: front berisi {v['n_front_SVD']} titik SVD dan "
                       f"{v['n_front_NMF']} titik NMF; titik NMF yang didominasi SVD: "
                       f"{v['NMF_points_dominated_by_SVD']} dari {v['n_points'].get('NMF', 0)}.")
    ag = res.get("evt_agreement", {})
    if ag.get("spearman_dpickands_vs_dmoment") is not None:
        out.append(f"Kesesuaian estimator EVT (rentang k {C.EVT_K_LO:g} sampai {C.EVT_K_HI:g} dari n): "
                   f"Spearman |dxi| Pickands vs Moment {_f(ag.get('spearman_dpickands_vs_dmoment'), 2)}; "
                   f"porsi Moment negatif F {_f(ag.get('share_moment_negative_F'), 2)}, "
                   f"M {_f(ag.get('share_moment_negative_M'), 2)}.")
    pr = res.get("evt_precision", {}).get("results", {})
    if pr:
        out.append("Presisi EVT (MDE bootstrap Delta-xi, daya 80%): " + "; ".join(
            f"{k} {_f(v.get('mde_boot'), 2)} (teramati {_f(v.get('observed_dxi'), 2)})"
            for k, v in pr.items() if k.endswith("|a0")) + ".")
    te = tuning.get("grid_edges", {})
    edge = [f"{k}: {', '.join(v)}" for k, v in te.items() if isinstance(v, list) and v]
    if edge:
        out.append("Parameter tuning terbaik di batas grid: " + "; ".join(edge) + ".")
    for m in ["SVD", "NMF"]:
        if f"{m}|k_terbaik_seluruh_grid" in te:
            out.append(f"Tuning {m}: k terbaik seluruh grid = {te[f'{m}|k_terbaik_seluruh_grid']}, "
                       f"selisih MAE validasi k terpilih vs terbaik = "
                       f"{_f(te.get(f'{m}|selisih_MAE_k_terpilih_vs_terbaik'), 5)}.")
    est = res["evt"]["estimability"]
    warm_rates = [min(v["F"], v["M"]) for k, v in est.items() if k.startswith("warm")]
    if warm_rates:
        out.append(f"Pickands terestimasi pada jalur warm minimal {_f(min(warm_rates), 2)} dari sel "
                   "(rasio per kombinasi model x k).")
    return out
