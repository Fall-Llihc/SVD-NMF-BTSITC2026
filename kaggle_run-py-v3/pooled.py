"""
pooled.py
=========
PERAN FILE (v3.4, analisis saja, tanpa pelatihan ulang)
    Inferensi gabungan lintas fold untuk metrik berbasis MAE, mengikuti prinsip BBC-CV
    (Tsamardinos, Greasidou, & Borboudakis, 2018): prediksi di luar fold (out-of-fold)
    digabung lalu di-bootstrap SEKALI pada unit pengguna dengan bobot yang sama untuk semua
    konfigurasi dan kedua jalur.

    - Jalur warm : galat absolut per pengguna dijumlahkan atas 5 fold. Setiap rating diuji
                   tepat sekali, sehingga setiap pengguna diwakili seluruh ratingnya.
    - Jalur cold : setiap pengguna menjadi pengguna baru tepat sekali (partisi pengguna),
                   sehingga gabungan 5 fold = seluruh 6.040 pengguna, masing-masing sekali.
    - Bobot bootstrap W (B x U) dibangkitkan sekali untuk seluruh pengguna dan dipakai oleh
      semua konfigurasi pada kedua jalur, sehingga setiap selisih (antar konfigurasi, antar
      jalur, model vs baseline) adalah bootstrap berpasangan pada pengguna yang sama.

ALASAN
    Pada jalur warm kelima fold berisi pengguna yang sama, sehingga aturan "signifikan pada
    >= 4 dari 5 fold" menumpuk uji pada sampel pengguna yang sama tanpa menambah informasi
    independen; pada jalur cold satu fold hanya memuat sekitar 1.208 pengguna. Estimator
    gabungan memakai seluruh data sekali dan merupakan estimator cross-validation baku.

KETERBATASAN
    Bootstrap pengguna tidak memasukkan variabilitas akibat set latih (Nadeau & Bengio, 2003);
    karena itu putusan per fold tetap dilaporkan sebagai uji stabilitas. Metrik yang tidak
    tersedia per pengguna (indeks ekor, amplifikasi genre, P@10) tidak ikut dianalisis di sini.

KELUARAN
    PooledCells: objek dengan antarmuka yang sama dengan analysis.Cells (satu "fold" bernilai
    -1), sehingga seluruh fungsi keluarga uji di analysis.py dapat dipakai ulang.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import evaluation as EV
from src import stats_tools as S

POOLED_SEED = -1
POOLED_BOOT_SEED = 777


def _pool(path, per_fold: dict, units: dict, idx: dict, U: int):
    """per_fold: {fold: {"ae", "n"}}; units: {fold: array userId}. Kembali (AE, N)."""
    AE, N = np.zeros(U), np.zeros(U)
    seen = np.zeros(U, dtype=int)
    for f, pu in per_fold.items():
        pos = np.array([idx[u] for u in units[f]])
        ae, n = np.asarray(pu["ae"], float), np.asarray(pu["n"], float)
        if len(pos) != len(ae):
            raise ValueError(f"panjang per_user fold {f} tidak cocok dengan daftar pengguna")
        AE[pos] += ae
        N[pos] += n
        seen[pos] += 1
    if path == "cold" and (seen > 1).any():
        raise ValueError("pengguna cold muncul di lebih dari satu fold")
    return AE, N


def _metrics(AE, N, lab, W):
    point, reps = EV.accuracy_block(AE, np.zeros_like(AE), N, lab, W, POOLED_SEED % 1000, "warm")
    point.pop("rmse", None)
    point = {k: v for k, v in point.items() if not isinstance(v, dict)}
    reps = {k: np.asarray(v, np.float32) for k, v in reps.items()}
    return point, reps


def build(cells, baselines: dict, users_lab: pd.DataFrame, ksel: dict):
    """Membangun PooledCells dan baseline gabungan. Kembali (PooledCells, baselines_pooled,
    info) atau (None, None, alasan) bila data tidak memadai."""
    from src.analysis import Cells, add_relative
    folds = sorted(baselines or {})
    if len(folds) < len(C.SEEDS):
        return None, None, "baseline belum lengkap untuk semua fold"
    units = {p: {f: np.asarray(baselines[f][p]["extra"]["per_user"]["units"]) for f in folds}
             for p in ["warm", "cold"]}
    all_users = np.sort(np.unique(np.concatenate(list(units["warm"].values()) +
                                                 list(units["cold"].values()))))
    U = len(all_users)
    idx = {u: i for i, u in enumerate(all_users)}
    lab = users_lab.reindex(all_users)
    W = S.boot_weights(U, POOLED_BOOT_SEED, "warm")

    # kelompokkan sel per konfigurasi
    groups = {}
    for key, ex in cells.extra.items():
        path, sc, m, k, a, seed, enc, var = key
        if path == "cold" and sc not in ("default", C.COLD_PRIMARY_SCENARIO):
            continue
        pu = (ex or {}).get("per_user")
        if pu is None or key not in cells.reps or not cells.reps[key]:
            continue   # sel divergen atau tanpa error per pengguna
        groups.setdefault((path, sc, m, k, a, enc, var), {})[seed] = pu

    pc = Cells([])
    n_cfg = 0
    for (path, sc, m, k, a, enc, var), per_fold in groups.items():
        if set(per_fold) != set(folds):
            continue
        AE, N = _pool(path, per_fold, units[path], idx, U)
        point, reps = _metrics(AE, N, lab, W)
        if path == "warm":   # seluruh pengguna = pengguna yang sama dengan jalur cold gabungan
            point.update({f"sub_{x}": v for x, v in list(point.items())
                          if x in ("mae", "relx_gender", "relx_age", "relx_occ") or x in C.GAP_METRICS})
            reps.update({f"sub_{x}": v for x, v in list(reps.items())})
        add_relative(point, reps)
        kk = pc.key(path, m, k, a, POOLED_SEED, sc, enc, var)
        row = dict(path=path, scenario=sc, model=m, k=k, alpha=a, seed=POOLED_SEED,
                   encoding=kk[6], variant=kk[7])
        row.update({x: v for x, v in point.items() if not isinstance(v, (dict, list))})
        pc.rows.append(row)
        pc.reps[kk] = reps
        pc.extra[kk] = {}
        n_cfg += 1
    pc.df = pd.DataFrame(pc.rows)

    # baseline gabungan
    bp = {}
    for path in ["warm", "cold"]:
        per_fold = {f: baselines[f][path]["extra"]["per_user"] for f in folds}
        AE, N = _pool(path, per_fold, units[path], idx, U)
        point, reps = _metrics(AE, N, lab, W)
        bp[path] = {"point": point, "reps": reps, "extra": {}}
    info = {"n_users": int(U), "n_configs": n_cfg, "n_boot": int(W.shape[0]),
            "method": "BBC-CV: bootstrap pengguna atas prediksi out-of-fold gabungan "
                      "(Tsamardinos dkk., 2018)"}
    return pc, {POOLED_SEED: bp}, info
