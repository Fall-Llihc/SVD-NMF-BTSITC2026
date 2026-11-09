"""
stats_tools.py
==============
PERAN FILE
    Alat statistik bersama: bootstrap berpasangan tingkat pengguna, p-value
    bootstrap, uji permutasi untuk excess gap, koreksi Holm, dan ringkasan antar seed.

BOOTSTRAP BERPASANGAN
    Matriks bobot W (B x n_pengguna) dibangkitkan dari seed yang hanya bergantung pada
    (seed split, jalur). Urutan pengguna juga tetap (userId terurut). Akibatnya, untuk
    satu seed dan satu jalur, SEMUA konfigurasi (model, k, alpha, encoding) memakai
    sampel ulang pengguna yang identik. Selisih replikat dua konfigurasi adalah
    bootstrap berpasangan yang sah tanpa perlu menyimpan prediksi mentah.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon  # noqa: F401  (dipakai di modul lain)

from src import config as C

PATH_CODE = {"warm": 1, "cold": 2}


def boot_weights(n_units: int, seed: int, path: str, n_boot: int | None = None) -> np.ndarray:
    """Bobot bootstrap pengguna (B x U): W[b, u] = berapa kali pengguna u terambil pada
    replikat b. Bobot yang sama dipakai untuk SEMUA konfigurasi pada (fold, jalur) yang sama,
    sehingga selisih antar konfigurasi adalah bootstrap berpasangan. Dibangkitkan per baris
    agar memori sementara kecil (B = 9999); baris pertama dipakai juga untuk EVT."""
    B = C.N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(100_000 + 10 * int(seed) + PATH_CODE[path])
    W = np.empty((B, n_units), dtype=np.float32)
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_units, size=n_units), minlength=n_units)
    return W


def ci(reps, level: float = 0.95):
    r = np.asarray(reps, float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return [None, None]
    a = (1 - level) / 2
    return [float(np.quantile(r, a)), float(np.quantile(r, 1 - a))]


def p_boot(reps, null: float = 0.0):
    """p dua sisi = 2 * min(P(d <= null), P(d >= null)), koreksi +1."""
    r = np.asarray(reps, float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return None
    lo = (np.sum(r <= null) + 1) / (r.size + 1)
    hi = (np.sum(r >= null) + 1) / (r.size + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def summarize(reps, point=None) -> dict:
    r = np.asarray(reps, float)
    fin = r[np.isfinite(r)]
    return {"point": None if point is None or not np.isfinite(point) else float(point),
            "boot_mean": float(fin.mean()) if fin.size else None,
            "ci95": ci(r), "n_finite": int(fin.size)}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni. pvals: {nama: p atau None}. Kembali {nama: p_adj}."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    out = {k: None for k in pvals}
    running = 0.0
    for rank, (k, p) in enumerate(sorted(items, key=lambda t: t[1])):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[k] = running
    return out


def group_ratio(W, num, den, mask):
    """Rata-rata tertimbang per replikat: sum(w*num)/sum(w*den) pada mask."""
    m = np.asarray(mask, bool)
    # vektor float32 agar W (float32, hingga 9999 x 6040) tidak disalin ke float64
    a = (W @ np.where(m, num, 0.0).astype(np.float32)).astype(np.float64)
    b = (W @ np.where(m, den, 0.0).astype(np.float32)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


def js2_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon kuadrat (basis 2) per baris untuk distribusi ternormalisasi."""
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(P > 0, P * np.log2(P / M), 0.0).sum(1)
        kq = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum(1)
    return 0.5 * (kp + kq)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, X / s, 0.0)


def across_seeds(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": None, "sd": None, "n": 0, "ci95_t": [None, None]}
    from scipy import stats
    m, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size > 1:
        h = stats.t.ppf(0.975, v.size - 1) * sd / np.sqrt(v.size)
        c = [m - h, m + h]
    else:
        c = [None, None]
    return {"mean": m, "sd": sd, "n": int(v.size), "ci95_t": c}
