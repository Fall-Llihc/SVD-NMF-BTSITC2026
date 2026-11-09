"""
recorder.py
===========
PERAN FILE
    Pencatat seluruh proses dan hasil running ke SATU berkas JSON
    (outputs/hasil_running_opsiC_v3.json) yang akan dianalisis lebih lanjut.

KUNCI TINGKAT ATAS
    meta, config (termasuk seluruh aturan protokol), environment, timing, dataset,
    eda, preprocessing, tuning, control, stage_logs, analysis, insights
"""
from __future__ import annotations

import json
from contextlib import contextmanager

from src import config as C
from src.utils import Timer, environment, now_iso, to_py


class RunRecorder:
    SECTIONS = ["meta", "config", "environment", "timing", "dataset", "eda", "preprocessing",
                "tuning", "control", "stage_logs", "analysis", "insights"]

    def __init__(self):
        self.data = {s: {} for s in self.SECTIONS}
        self.data["meta"] = {
            "title": "Opsi C v3.4 (analisis; cache v3.3) | Bobot demografi (gender, usia, kelompok pekerjaan) pada SVD "
                     "dan NMF, grid k x alpha x 5-fold",
            "version": "v3.4",
            "note_seed": "kolom/kunci 'seed' berarti indeks fold (0 sampai 4)", "started_at": now_iso(),
            "author": "Arkhan Falih Fahrie Puspita (103052330051)",
            "research": "Analisis Bias Demografis pada SVD dan NMF dalam Sistem Rekomendasi "
                        "Menggunakan Pendekatan Extreme Value Theory",
            "protocol": "Protokol Analisis Opsi C v3 (dikunci 29 Oktober 2026)",
            "json_guide": {
                "config": "seluruh konstanta dan aturan keputusan protokol",
                "tuning": "grid tuning per model x k, hyperparameter terpilih, k terpilih",
                "control": "kontrol reprodusibilitas: hyperparameter v2 vs baseline",
                "analysis.table": "satu baris per sel (jalur, kategori, model, k, alpha, fold, encoding, variant)",
                "analysis.aggregate": "rata-rata dan sd antar fold per konfigurasi",
                "analysis.h1": "F1 dan putusan Hipotesis Pertama",
                "analysis.h2": "F6 dan putusan Hipotesis Kedua",
                "analysis.alpha_effect": "F2: efek alpha terhadap disparitas",
                "analysis.safe_alpha": "bobot demografi aman (non-inferioritas)",
                "analysis.new_users": "F3: klaim pengguna baru menanggung bias lebih besar",
                "analysis.amplification": "F4: amplifikasi perbedaan genre antar gender",
                "analysis.mechanism": "F5: cosinus F vs M per encoding, ablation NMF bias",
                "analysis.leakage": "probe kelompok pekerjaan laten vs baseline probe (tanpa pekerjaan sebagai input)",
                "analysis.occupation_effect": "F7: efek menambah pekerjaan sebagai input",
                "analysis.pareto": "front Pareto per jalur dan sumbu",
                "analysis.evt": "keterestimasian dan kurva Pickands/Hill/Moment",
                "analysis.rankings": "Top-10 per profil gender x usia x pekerjaan (fold 0, k terpilih)",
                "analysis.deviations": "penyesuaian protokol saat implementasi",
            }}
        self.data["environment"] = environment()

    @contextmanager
    def stage(self, name: str):
        with Timer() as t:
            yield
        self.data["timing"][name] = round(t.seconds, 2)

    def set(self, section: str, value, key: str | None = None):
        if key is None:
            self.data[section] = value
        else:
            self.data.setdefault(section, {})[key] = value

    def save(self, path=None):
        C.ensure_dirs()
        self.data["meta"]["finished_at"] = now_iso()
        self.data["config"] = C.as_dict()
        path = path or C.RESULT_JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_py(self.data), f, ensure_ascii=False)
        return path
