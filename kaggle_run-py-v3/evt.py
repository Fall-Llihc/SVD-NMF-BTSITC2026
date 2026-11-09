"""
evt.py
======
PERAN FILE
    Estimator indeks ekor Extreme Value Theory untuk distribusi eksposur
    rekomendasi: Pickands (metrik utama), Hill, dan Moment Dekkers-Einmahl-de Haan
    (pembanding ketahanan). Seluruh aturan mengikuti Lubang 6 protokol.

KONVENSI (memperbaiki Persamaan 2.16 proposal)
    Data diurutkan MENURUN: X(1) >= X(2) >= ... >= X(n). Dengan indeks 1:
        Pickands(k) = (1/ln 2) * ln[(X(k) - X(2k)) / (X(2k) - X(4k))],  1 <= k <= n/4
        Hill(k)     = (1/k) * sum_{i=1..k} ln X(i)  -  ln X(k+1)
        Moment(k)   = M1 + 1 - 0.5 / (1 - M1^2 / M2),
                      Mj = (1/k) * sum_{i=1..k} (ln X(i) - ln X(k+1))^j
    Masukan adalah jumlah eksposur per film (> 0) dalam Top-N seluruh anggota satu
    kelompok; n = jumlah film berbeda yang direkomendasikan ke kelompok itu.

ATURAN STABIL DAN KETERESTIMASIAN (v3.2)
    - Rentang k: bilangan bulat dari ceil(0,01 n) sampai floor(0,05 n). Pickands memakai
      X(4k), sehingga paling jauh 20% data teratas yang dipakai; teori EVT mensyaratkan
      k -> tak hingga dengan k/n -> 0 (de Haan & Ferreira, 2006). Rentang v3.1
      (5% sampai 20%, sampai 80% data) terlalu dalam masuk ke badan distribusi.
    - Jumlah eksposur dibatasi jumlah pengguna kelompok, sehingga ekor berujung hingga
      (xi <= 0) secara teoretis mungkin; Moment estimator (Dekkers, Einmahl, & de Haan,
      1989) konsisten untuk semua tanda xi dan menjadi uji ketahanan utama.
    - Ties: Pickands(k) dibuang bila pembilang atau penyebut <= 0 (tidak dipotong
      ke 0). Hill/Moment dibuang bila tidak terdefinisi.
    - Nilai stabil = median nilai yang tersisa.
    - Tidak terestimasi bila rentang berisi < 10 nilai k atau > 50% k terbuang.
"""
from __future__ import annotations

import numpy as np

from src import config as C

LN2 = np.log(2.0)


def k_range(n: int, lo_frac=None, hi_frac=None) -> np.ndarray:
    lo_frac = C.EVT_K_LO if lo_frac is None else lo_frac
    hi_frac = C.EVT_K_HI if hi_frac is None else hi_frac
    lo = max(1, int(np.ceil(lo_frac * n)))
    hi = min(int(np.floor(hi_frac * n)), n // 4)
    return np.arange(lo, hi + 1) if hi >= lo else np.array([], dtype=int)


def curves(x, ks=None) -> dict:
    """Nilai ketiga estimator untuk setiap k (untuk Pickands plot)."""
    x = np.sort(np.asarray(x, float)[np.asarray(x) > 0])[::-1]
    n = x.size
    if ks is None:
        ks = np.arange(1, n // 4 + 1)
    ks = np.asarray(ks, int)
    ks = ks[(ks >= 1) & (4 * ks <= n)]
    out = {"n": int(n), "k": ks.tolist()}
    if ks.size == 0:
        out.update(pickands=[], hill=[], moment=[])
        return out
    num = x[ks - 1] - x[2 * ks - 1]
    den = x[2 * ks - 1] - x[4 * ks - 1]
    ok = (num > 0) & (den > 0)
    pk = np.full(ks.size, np.nan)
    pk[ok] = np.log(num[ok] / den[ok]) / LN2
    lx = np.log(x)
    cs1 = np.cumsum(lx)
    cs2 = np.cumsum(lx ** 2)
    ref = lx[ks]                                   # ln X(k+1)
    s1 = cs1[ks - 1]
    s2 = cs2[ks - 1]
    M1 = s1 / ks - ref
    M2 = (s2 - 2 * ref * s1) / ks + ref ** 2
    hill = M1
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = M1 + 1 - 0.5 / (1 - M1 ** 2 / M2)
    mom[~np.isfinite(mom) | (M2 <= 1e-15)] = np.nan
    out.update(pickands=pk.tolist(), hill=hill.tolist(), moment=mom.tolist())
    return out


def stable(x, lo_frac=None, hi_frac=None) -> dict:
    """Estimasi stabil ketiga estimator beserta diagnostik keterestimasian."""
    x = np.asarray(x, float)
    x = x[x > 0]
    n = x.size
    ks = k_range(n, lo_frac, hi_frac)
    res = {"n": int(n), "n_k": int(ks.size), "k_lo": int(ks[0]) if ks.size else None,
           "k_hi": int(ks[-1]) if ks.size else None}
    if ks.size < C.EVT_MIN_K_VALUES:
        res.update(pickands=np.nan, hill=np.nan, moment=np.nan, n_dropped=None,
                   frac_dropped=None, estimable=False, reason="rentang k < 10 nilai")
        return res
    cv = curves(x, ks)
    pk = np.array(cv["pickands"], float)
    drop = int(np.isnan(pk).sum())
    frac = drop / ks.size
    est = frac <= C.EVT_MAX_DROP
    hill = np.array(cv["hill"], float)
    mom = np.array(cv["moment"], float)
    res.update(pickands=float(np.nanmedian(pk)) if est else np.nan,
               hill=float(np.nanmedian(hill)) if np.isfinite(hill).any() else np.nan,
               moment=float(np.nanmedian(mom)) if np.isfinite(mom).any() else np.nan,
               n_dropped=drop, frac_dropped=float(frac), estimable=bool(est),
               reason=None if est else "lebih dari 50% k terbuang (ties)")
    return res


SENSITIVITY_RANGES = [(0.005, 0.025), (0.01, 0.05), (0.025, 0.10), (0.05, 0.20)]


def sensitivity(x) -> dict:
    """Nilai stabil Pickands dan Moment untuk beberapa rentang k (uji sensitivitas)."""
    out = {}
    for lo, hi in SENSITIVITY_RANGES:
        r = stable(x, lo, hi)
        out[f"{lo:g}-{hi:g}"] = {"pickands": r["pickands"], "moment": r["moment"],
                                 "hill": r["hill"], "estimable": r["estimable"], "n_k": r["n_k"]}
    return out


def stable_rows(E: np.ndarray) -> np.ndarray:
    """E: matriks (B x n_item) jumlah eksposur hasil bootstrap.
    Kembali matriks (B x 3): Pickands, Hill, Moment stabil (NaN bila tidak terestimasi)."""
    out = np.full((E.shape[0], 3), np.nan)
    for b in range(E.shape[0]):
        r = stable(E[b])
        out[b] = [r["pickands"], r["hill"], r["moment"]]
    return out
