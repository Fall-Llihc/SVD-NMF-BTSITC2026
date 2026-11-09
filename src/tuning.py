"""
tuning.py
=========
PERAN FILE
    Hyperparameter tuning per algoritma per nilai k (empat aturan protokol):
      1. tuning hanya pada alpha = 0, sehingga efek alpha tidak tercampur efek tuning;
      2. hanya pada data validasi (10% rating train fold 0 per pengguna), test tidak
         pernah disentuh;
      3. kriteria hanya MAE validasi; fairness tidak dipakai;
      4. sekali pada fold pertama, lalu konfigurasinya dikunci untuk semua fold.
    Juga menentukan k terpilih per algoritma (MAE validasi terbaik di antara k grid utama)
    untuk ablation dan uji H1. Setiap parameter terbaik yang berada di batas grid ditandai
    pada "grid_edges". v3.3: grid diperluas lagi (SVD lr 0,0025 sampai 0,01, epoch 40
    sampai 160; NMF epoch 200 sampai 400), k = 200 masuk grid utama, dan k = 300 dilatih
    sekali dengan hyperparameter terbaik k = 200 sebagai diagnostik batas grid k.

KELUARAN
    outputs/cache_v3.3/tuning.json : seluruh hasil grid, parameter terbaik per (model, k),
                                k terpilih per model. Bila berkas ini sudah ada,
                                tuning tidak diulang.
"""
from __future__ import annotations

import json

import numpy as np

from src import config as C
from src import features as F
from src import models as MD
from src.utils import to_py

TUNING_FILE = "tuning.json"


def tuning_path():
    return C.CACHE_DIR / TUNING_FILE


def run_tuning(ratings, users, verbose: bool = True) -> dict:
    C.ensure_dirs()
    path = tuning_path()
    if path.exists():
        res = json.load(open(path))
        if set(map(int, res["best"]["SVD"])) >= set(C.TUNING_KS) and \
                res.get("grid") == _grid_signature():
            if verbose:
                print("Tuning sudah ada di cache, dipakai ulang.")
            return res
    from src.data import FoldSplit
    sp = FoldSplit(ratings, users, 0, with_validation=True)
    ts = F.build_trainset(sp.fit_df, users, [], 0)
    val = sp.val_df
    rows = []
    for name in C.MODELS:
        for k in C.TUNING_KS:
            for hp in C.TUNING_GRID[name]:
                params = MD.build_params(name, k, hp, C.BASE_SEED)
                algo = MD.train_model(name, ts, params)
                est = MD.predict_ratings(algo, ts, val.userId.tolist(), val.movieId.tolist())
                mae = float(np.abs(val.rating.to_numpy() - est).mean())
                rows.append({"model": name, "k": k, **hp, "val_MAE": mae,
                             "seconds": float(algo._train_seconds)})
                if verbose:
                    print(f"  tuning {name} k={k} {hp} -> MAE val {mae:.4f} "
                          f"({algo._train_seconds:.0f} dtk)")
    best, k_sel, edges = {}, {}, {}
    for name in C.MODELS:
        best[name] = {}
        grid = C.TUNING_GRID[name]
        for k in C.TUNING_KS:
            cand = [r for r in rows if r["model"] == name and r["k"] == k]
            b = min(cand, key=lambda r: r["val_MAE"])
            hp = {kk: b[kk] for kk in grid[0]}
            best[name][str(k)] = {"hp": hp, "val_MAE": b["val_MAE"]}
            edge = []
            for kk, v in hp.items():
                vals = sorted({g[kk] for g in grid})
                if len(vals) > 1 and v in (vals[0], vals[-1]):
                    edge.append(f"{kk}={v} ({'minimum' if v == vals[0] else 'maksimum'} grid)")
            edges[f"{name}|k{k}"] = edge
        # k terpilih dibatasi pada grid utama (C.KS) agar biaya komputasi terkendali;
        # k di luar grid utama hanya diagnostik batas grid
        k_sel[name] = int(min((str(k) for k in C.KS), key=lambda kk: best[name][kk]["val_MAE"]))
        k_all = int(min(best[name], key=lambda kk: best[name][kk]["val_MAE"]))
        edges[f"{name}|k_terbaik_seluruh_grid"] = k_all
        edges[f"{name}|selisih_MAE_k_terpilih_vs_terbaik"] = \
            best[name][str(k_sel[name])]["val_MAE"] - best[name][str(k_all)]["val_MAE"]
        # diagnostik batas grid k: k di atas grid utama dengan hyperparameter k terbesar
        kmax = max(C.TUNING_KS)
        for kd in C.TUNING_DIAG_K:
            hp = best[name][str(kmax)]["hp"]
            algo = MD.train_model(name, ts, MD.build_params(name, kd, hp, C.BASE_SEED))
            est = MD.predict_ratings(algo, ts, val.userId.tolist(), val.movieId.tolist())
            mae = float(np.abs(val.rating.to_numpy() - est).mean())
            diff = mae - best[name][str(kmax)]["val_MAE"]
            edges[f"{name}|diagnostik_k{kd}"] = {"hp": hp, "val_MAE": mae,
                                                  "selisih_vs_k_maks_grid": diff,
                                                  "k_maks_grid_di_batas": bool(diff < 0)}
            if verbose:
                print(f"  diagnostik {name} k={kd} -> MAE val {mae:.4f} (selisih {diff:+.4f})")
    res = {"rows": rows, "best": best, "k_selected": k_sel, "grid_edges": edges,
           "criterion": C.TUNING_CRITERION, "grid": _grid_signature(),
           "n_fit": int(len(sp.fit_df)), "n_val": int(len(val))}
    json.dump(to_py(res), open(path, "w"), indent=1)
    return res


def _grid_signature() -> str:
    return json.dumps({"grid": C.TUNING_GRID, "ks": C.TUNING_KS, "diag": C.TUNING_DIAG_K},
                      sort_keys=True)


def hp_for(tuning: dict, model: str, k: int) -> dict:
    return dict(tuning["best"][model][str(k)]["hp"])


def hp_hash(hp: dict) -> str:
    """Sidik hyperparameter (8 karakter) untuk nama berkas cache, sehingga sel yang dilatih
    dengan hyperparameter lain tidak pernah terpakai ulang secara diam-diam."""
    import hashlib
    return hashlib.sha1(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:8]
