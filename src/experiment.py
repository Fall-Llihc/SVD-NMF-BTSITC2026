"""
experiment.py
=============
PERAN FILE
    Menjalankan grid eksperimen 5-fold secara bertahap dengan cache per sel, sehingga
    pekerjaan panjang dapat dipecah ke beberapa sesi dan dilanjutkan bila terhenti.

TAHAP (variabel STAGE di notebook)
    tuning   : hyperparameter tuning (tuning.py)
    control  : kontrol reprodusibilitas baseline (hyperparameter v2, k = 50, seed 42,
               alpha = 0, jalur warm) dibandingkan dengan berkas referensi baseline
    baseline : prediktor BaselineOnly per fold (pembanding gap, v3.3)
    k50      : grid utama k = 50 (warm + cold), seluruh alpha dan fold
    k10_20   : grid utama k = 10 dan 20
    k100     : grid utama k = 100
    k200     : grid utama k = 200 (v3.3)
    ablation : pada k terpilih, alpha > 0: encoding member5 dan membermean (warm + cold),
               serta input gender + usia saja tanpa pekerjaan (warm)
    all      : semua tahap di atas berurutan
    Daftar tahap juga diterima, misalnya ["tuning", "control", "k50"].
    Tahap analisis tidak melatih model; ia membaca seluruh cache yang tersedia.

CACHE (v3.3)
    outputs/cache_v3.3/<path>__<model>__k<k>__a<alpha>__s<fold>__<encoding>[__<variant>]__h<hash>.pkl
    <hash> = sidik hyperparameter. Bila hasil tuning berubah, sel lama tidak terpakai dan
    tidak ikut dianalisis (load_cells menyaring berdasarkan hyperparameter aktif).
    Sel alpha = 0 tidak bergantung pada encoding/variant (tanpa pseudo-item), sehingga
    ablation hanya dijalankan pada alpha > 0 dan memakai sel utama alpha = 0 sebagai acuan.
"""
from __future__ import annotations

import json
import pickle

from src import config as C
from src import evaluation as EV
from src import tuning as T
from src.utils import Timer, to_py

STAGES = ["tuning", "control", "baseline", "k50", "k10_20", "k100", "k200", "ablation"]
STAGE_KS = {"k50": [50], "k10_20": [10, 20], "k100": [100], "k200": [200]}


def cell_file(path, model, k, alpha, seed, encoding="full", variant=None, hp=None):
    name = f"{path}__{model}__k{k}__a{alpha}__s{seed}__{encoding}"
    if variant:
        name += f"__{variant}"
    if hp is not None:
        name += f"__h{T.hp_hash(hp)}"
    return C.CACHE_DIR / f"{name}.pkl"


def baseline_file(seed):
    return C.CACHE_DIR / f"baseline__s{seed}.pkl"


def _save(obj, f):
    tmp = f.with_suffix(f.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    tmp.replace(f)          # tulis atomik: berkas setengah jadi tidak pernah terbaca


def run_cells(data, tuning, ks, seeds=None, alphas=None, models=None, paths=("warm", "cold"),
              encoding="full", variant=None, verbose=True) -> dict:
    ratings, users, movies = data
    seeds = C.SEEDS if seeds is None else seeds
    alphas = C.ALPHAS if alphas is None else alphas
    models = C.MODELS if models is None else models
    C.ensure_dirs()
    done, skipped, failed = 0, 0, 0
    for seed in seeds:
        todo = [(p, m, k, a) for p in paths for k in ks for m in models for a in alphas
                if not cell_file(p, m, k, a, seed, encoding, variant, T.hp_for(tuning, m, k)).exists()]
        skipped += len(paths) * len(ks) * len(models) * len(alphas) - len(todo)
        if not todo:
            continue
        with Timer() as tc:
            ctx = EV.SeedContext(ratings, users, movies, seed)
        if verbose:
            print(f"[fold {seed}] konteks siap ({tc.seconds:.0f} dtk), {len(todo)} sel")
        for p, m, k, a in todo:
            hp = T.hp_for(tuning, m, k)
            probe = (k == tuning["k_selected"][m]) and p == "warm"
            with Timer() as t:
                try:
                    if p == "warm":
                        attrs = C.INPUT_ABLATION_ATTRS if variant == "gu" else None
                        res = EV.warm_cell(ctx, m, k, a, hp, encoding, None, probe, attrs)
                    else:
                        if variant:
                            raise ValueError("variant tidak didukung pada jalur cold")
                        res = EV.cold_cell(ctx, m, k, a, hp, encoding)
                except Exception as e:  # sel gagal dicatat, running tidak berhenti
                    res = {"failed": True, "error": f"{type(e).__name__}: {e}"}
            res["key"] = dict(path=p, model=m, k=k, alpha=a, seed=seed, encoding=encoding,
                              variant=variant, seconds=t.seconds, hp=hp, hp_hash=T.hp_hash(hp))
            cf = cell_file(p, m, k, a, seed, encoding, variant, hp)
            marker = cf.with_suffix(".failed")
            if res.get("failed"):
                # penanda terpisah (.failed): sel ini dicoba ulang pada running berikutnya
                _save(res, marker)
                failed += 1
                print(f"  GAGAL {p} {m} k={k} a={a} s={seed} {encoding}/{variant}: {res['error']}")
                continue
            _save(res, cf)
            if marker.exists():
                marker.unlink()
            done += 1
            if verbose:
                pt = res["point"] if p == "warm" else \
                    res["scenarios"].get(C.COLD_PRIMARY_SCENARIO, res["scenarios"]["default"])["point"]
                div = " | DIVERGEN" if pt.get("diverged") else ""
                print(f"  {p:4s} {m} k={k:3d} a={a} s={seed} {encoding}{'/' + variant if variant else ''}"
                      f" | MAE {pt['mae']:.4f} gapG {pt['gap_gender']:.4f} gapU {pt['gap_age_ref']:.4f}"
                      f" xiF {pt.get('xi_F', float('nan')):.3f} xiM {pt.get('xi_M', float('nan')):.3f}"
                      f" | {t.seconds:.0f} dtk{div}")
    return {"computed": done, "from_cache": skipped, "failed": failed}


def run_baselines(data, seeds=None, verbose=True) -> dict:
    ratings, users, movies = data
    seeds = C.SEEDS if seeds is None else seeds
    C.ensure_dirs()
    done = 0
    for seed in seeds:
        f = baseline_file(seed)
        if f.exists():
            continue
        ctx = EV.SeedContext(ratings, users, movies, seed)
        res = EV.baseline_cell(ctx)
        _save(res, f)
        done += 1
        if verbose:
            print(f"  baseline fold {seed} | MAE warm {res['warm']['point']['mae']:.4f} "
                  f"gapG {res['warm']['point']['gap_gender']:.4f} | MAE cold "
                  f"{res['cold']['point']['mae']:.4f}")
    return {"computed": done, "from_cache": len(seeds) - done}


def load_failures() -> list:
    """Daftar sel yang gagal (disimpan sebagai penanda di cache)."""
    out = []
    for f in sorted(C.CACHE_DIR.glob("*.failed")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        out.append({"file": f.name, "error": r.get("error"), "key": r.get("key")})
    return out


def run_control(data, verbose=True) -> dict:
    """Hyperparameter v2, k = 50, seed 42, alpha = 0, warm: harus identik baseline."""
    import numpy as np
    f = C.CACHE_DIR / "control.json"
    if f.exists():
        return json.load(open(f))
    ratings, users, movies = data
    ctx = EV.SeedContext(ratings, users, movies, 999, baseline=True)   # 999 = penanda kontrol
    out = {"k": C.CONTROL_K, "seed": C.BASE_SEED, "split": "80:20 seed 42 (baseline)",
           "models": {}}
    ref = None
    if C.REFERENCE_BASELINE.exists():
        ref = json.load(open(C.REFERENCE_BASELINE)).get("per_config", {}).get(f"k{C.CONTROL_K}")
    for m in C.MODELS:
        hp = {k: v for k, v in C.V2_PARAMS[m].items() if k != "biased"}
        res = EV.warm_cell(ctx, m, C.CONTROL_K, 0, hp)
        pt = res["point"]
        row = {"MAE": pt["mae"], "gap_gender": pt["gap_gender"], "dp_g": pt["dp_g"]}
        if ref:
            b = ref["models"][m]
            row.update(MAE_baseline=b["accuracy"]["MAE"],
                       gap_gender_baseline=b["fairness"]["gender"]["per_group_mae"]["disparity"],
                       dp_g_baseline=b["fairness"]["gender"]["demographic_parity"]["dp_gap"])
            row["abs_diff_MAE"] = abs(row["MAE"] - row["MAE_baseline"])
        out["models"][m] = row
        if verbose:
            print(f"  kontrol {m}: MAE {row['MAE']:.4f}"
                  + (f" | baseline {row['MAE_baseline']:.4f}" if ref else ""))
    out["available"] = ref is not None
    out["identical"] = bool(ref) and all(v["abs_diff_MAE"] < 1e-4 for v in out["models"].values())
    json.dump(to_py(out), open(f, "w"), indent=1)
    return out


def run_stage(stage, data, verbose=True) -> dict:
    ratings, users, _ = data
    log = {}
    if isinstance(stage, (list, tuple)):
        stages = list(stage)
    else:
        stages = STAGES if stage == "all" else [stage]
    tuning = None
    for st in stages:
        with Timer() as t:
            if st == "tuning":
                tuning = T.run_tuning(ratings, users, verbose)
                log[st] = {"k_selected": tuning["k_selected"]}
            else:
                tuning = tuning or T.run_tuning(ratings, users, verbose)
                if st == "control":
                    log[st] = run_control(data, verbose)
                elif st == "baseline":
                    log[st] = run_baselines(data, verbose=verbose)
                elif st in STAGE_KS:
                    ks = [k for k in STAGE_KS[st] if k in C.KS]
                    log[st] = run_cells(data, tuning, ks, verbose=verbose)
                elif st == "ablation":
                    r = {}
                    alphas = [a for a in C.ALPHAS if a > 0]
                    for m in C.MODELS:
                        ksel = tuning["k_selected"][m]
                        for enc in C.ABLATION_ENCODINGS:
                            r[f"{m}_{enc}"] = run_cells(data, tuning, [ksel], alphas=alphas,
                                                        models=[m], paths=C.ABLATION_PATHS,
                                                        encoding=enc, verbose=verbose)
                        r[f"{m}_input_gender_usia"] = run_cells(
                            data, tuning, [ksel], alphas=alphas, models=[m], paths=("warm",),
                            variant="gu", verbose=verbose)
                    log[st] = r
                else:
                    raise ValueError(f"tahap tidak dikenal: {st}")
        log[f"{st}_seconds"] = round(t.seconds, 1)
        if verbose:
            print(f"== tahap {st} selesai dalam {t.seconds / 60:.1f} menit")
    return log


def load_cells(tuning: dict | None = None) -> list:
    """Membaca seluruh sel di cache. Bila tuning diberikan, hanya sel yang dilatih dengan
    hyperparameter hasil tuning aktif yang dibaca."""
    recs = []
    for f in sorted(C.CACHE_DIR.glob("*__h*.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        if r.get("failed"):
            continue
        k = r["key"]
        if tuning is not None and k.get("hp_hash") != T.hp_hash(T.hp_for(tuning, k["model"], k["k"])):
            continue
        recs.append(r)
    return recs


def load_baselines() -> dict:
    out = {}
    for f in sorted(C.CACHE_DIR.glob("baseline__s*.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        out[int(r["key"]["seed"])] = r
    return out
