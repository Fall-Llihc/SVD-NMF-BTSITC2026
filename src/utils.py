"""
utils.py
========
PERAN FILE
    Fungsi bantu umum yang dipakai lintas modul dan tidak terkait langsung dengan
    sistem rekomendasi.

ISI UTAMA
    - to_py         : mengubah tipe numpy/pandas menjadi tipe Python native agar
                      aman ditulis ke JSON (NaN dan inf menjadi null).
    - Timer         : pengukur durasi sederhana untuk mencatat waktu setiap tahap.
    - environment   : versi pustaka dan informasi mesin untuk reprodusibilitas.
    - md_table      : mengubah DataFrame menjadi tabel Markdown untuk teks insight.
"""
from __future__ import annotations

import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def to_py(obj):
    """Konversi rekursif ke tipe yang bisa diserialisasi JSON."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_py(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_py(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Timer:
    """Pemakaian: with Timer() as t: ...; t.seconds"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


def environment() -> dict:
    import scipy
    import sklearn
    import surprise
    import matplotlib
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "scikit_surprise": surprise.__version__,
        "matplotlib": matplotlib.__version__,
    }


def md_table(df: pd.DataFrame, nd: int = 4) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else f"{v:.{nd}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def fmt(x, nd: int = 4, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
