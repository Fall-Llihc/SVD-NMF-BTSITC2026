"""
config.py
=========
PERAN FILE
    Pusat seluruh konstanta Opsi C v3. Isi file ini adalah terjemahan langsung dari
    dokumen "Protokol Analisis Opsi C v3": setiap aturan keputusan yang dikunci
    sebelum running ditulis di sini dan ikut tersimpan di JSON hasil, sehingga
    tidak ada aturan yang bisa diubah diam-diam setelah hasil terlihat.

    Modul lain membaca nilai lewat `from src import config as C` lalu `C.NAMA`
    pada saat fungsi dipanggil, sehingga notebook dapat mengubah nilai (misalnya
    mode cepat) sebelum eksperimen dijalankan.

ISI UTAMA
    1. Path proyek dan cache.
    2. Desain eksperimen: grid k x alpha x fold (5-fold), encoding, model, kelompok
       pekerjaan beserta landasannya.
    3. Hyperparameter: grid tuning dan konfigurasi kontrol baseline (v2).
    4. Evaluasi: Top-N, ambang popularitas, relevansi, Demographic Parity.
    5. Aturan EVT (Pickands utama, Moment pembanding; Hill hanya deskriptif) dan statistik.
    6. Aturan keputusan: bobot aman, H1, H2, mekanisme, klaim pengguna baru, Pareto.
    7. Label demografis MovieLens 1M.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Path
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("OPSI_DATA_DIR", ROOT / "data" / "ml-1m"))
REFERENCE_BASELINE = ROOT / "data" / "reference" / "baseline_v3_combined_analysis.json"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
CACHE_VERSION = "v3.3"   # cache versi lama (v3 sampai v3.2) TIDAK kompatibel: folder berbeda
CACHE_DIR = Path(os.environ.get("OPSI_CACHE_DIR", OUTPUT_DIR / f"cache_{CACHE_VERSION}"))
RESULT_JSON = OUTPUT_DIR / "hasil_running_opsiC_v3.json"
ANALYSIS_VERSION = "v3.4"  # v3.4 = revisi analisis saja; cache tetap v3.3 (tidak perlu latih ulang)

# --------------------------------------------------------------------------
# 2. Desain eksperimen
# --------------------------------------------------------------------------
BASE_SEED = 42                       # seed acak dasar (pengacakan fold, inisialisasi model)
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]              # INDEKS FOLD (nama "SEEDS"/"seed" dipertahankan agar
                                     # format kode dan JSON sama dengan v3 awal)
TEST_FRAC = 0.20                     # hanya untuk kontrol baseline (split lama seed 42)
VAL_FRAC = 0.10                      # porsi train per pengguna untuk validasi tuning
COLD_FRAC = 0.20                     # setara 1 dari 5 fold pengguna (cold-start)
RATING_SCALE = (1, 5)

KS = [10, 20, 50, 100, 200]          # v3.3: k = 200 masuk grid utama
ALPHAS = [0, 1, 2, 4, 8]
MODELS = ["SVD", "NMF"]
ENCODINGS = {
    "full": "Utama: anggota 5, bukan anggota 1 (semua pseudo-item dirating)",
    "member5": "Ablation (b): hanya pseudo-item kelompok sendiri, nilai 5",
    "membermean": "Ablation (c): hanya pseudo-item kelompok sendiri, nilai = rata-rata rating pengguna",
}
MAIN_ENCODING = "full"
ABLATION_ENCODINGS = ["member5", "membermean"]
ABLATION_PATHS = ("warm", "cold")      # v3.3: encoding hanya-anggota juga diuji pada pengguna baru
DEMO_HIGH, DEMO_LOW = 5.0, 1.0
PROFILE_ATTRS = ["gender", "age_group", "occ_group"]      # atribut input
SENSITIVE_ATTRS = ["gender", "age_group", "occ_group"]    # atribut evaluasi
INPUT_ABLATION_ATTRS = ["gender", "age_group"]            # ablation tanpa pekerjaan
_ATTR_KEY = {"gender": "gender", "age_group": "age", "occ_group": "occ"}
SCENARIOS = {"default": []}
for _r in range(1, 4):
    from itertools import combinations as _comb
    for _c in _comb(PROFILE_ATTRS, _r):
        SCENARIOS["_".join(_ATTR_KEY[a] for a in _c)] = list(_c)
_LBL = {"gender": "Gender", "age": "Usia", "occ": "Pekerjaan"}
SCENARIO_LABELS = {k: ("Default (tanpa demografi)" if k == "default" else
                       " + ".join(_LBL[x] for x in k.split("_"))) for k in SCENARIOS}
COLD_PRIMARY_SCENARIO = "gender_age_occ"   # kategori cold-start utama untuk uji

# Kelompok pekerjaan (landasan: ISCO-08 skill level; resolusi statistik ketenagakerjaan
# ILO/ICLS untuk status aktivitas; ICSE-18 untuk status dalam pekerjaan)
OCC_GROUP = {
    "executive/managerial": "Keahlian tinggi", "academic/educator": "Keahlian tinggi",
    "technician/engineer": "Keahlian tinggi", "programmer": "Keahlian tinggi",
    "writer": "Keahlian tinggi", "artist": "Keahlian tinggi",
    "doctor/health care": "Keahlian tinggi", "scientist": "Keahlian tinggi",
    "lawyer": "Keahlian tinggi",
    "sales/marketing": "Keahlian menengah", "clerical/admin": "Keahlian menengah",
    "customer service": "Keahlian menengah", "tradesman/craftsman": "Keahlian menengah",
    "farmer": "Keahlian menengah",
    "K-12 student": "Pelajar/mahasiswa", "college/grad student": "Pelajar/mahasiswa",
    "retired": "Tidak bekerja", "homemaker": "Tidak bekerja", "unemployed": "Tidak bekerja",
    "other": "Tidak terklasifikasi", "self-employed": "Tidak terklasifikasi",
}
OCC_GROUP_ORDER = ["Keahlian tinggi", "Keahlian menengah", "Pelajar/mahasiswa",
                   "Tidak bekerja", "Tidak terklasifikasi"]
OCC_UNCLASSIFIED = "Tidak terklasifikasi"   # dipakai sebagai input, tidak dalam metrik disparitas
OCC_GROUP_BASIS = {
    "Keahlian tinggi": "ISCO-08 kelompok utama 1-3 (skill level 3-4)",
    "Keahlian menengah": "ISCO-08 kelompok utama 4-8 (skill level 2); sales/marketing ambigu",
    "Pelajar/mahasiswa": "ILO ICLS: status aktivitas utama 'studies'",
    "Tidak bekerja": "ILO ICLS: not in employment (menganggur + di luar angkatan kerja)",
    "Tidak terklasifikasi": "other: tanpa informasi; self-employed: status dalam pekerjaan (ICSE-18)",
}

# Mode cepat untuk uji alur (bukan untuk hasil penelitian)
QUICK_MODE = os.environ.get("OPSI_QUICK", "0") == "1"
QUICK = dict(SEEDS=[0, 1], KS=[10, 20], ALPHAS=[0, 1, 4], N_BOOT=199, N_BOOT_EVT=59, N_PERM=60,
             TUNING_DIAG_K=[])

# --------------------------------------------------------------------------
# 3. Hyperparameter
# --------------------------------------------------------------------------
TUNING_GRID = {   # v3.3: diperluas lagi karena optimum v3.2 berada di batas grid
    "SVD": [dict(reg_all=r, lr_all=lr, n_epochs=e)
            for r in (0.02, 0.05, 0.1) for lr in (0.0025, 0.005, 0.01) for e in (40, 80, 160)],
    "NMF": [dict(reg_pu=r, reg_qi=r, n_epochs=e)
            for r in (0.02, 0.06, 0.15) for e in (200, 300, 400)],
}
TUNING_KS = list(KS)                  # k terpilih dipilih dari grid utama
TUNING_DIAG_K = [300]                 # diagnostik batas grid k (hyperparameter terbaik k = 200)
TUNING_CRITERION = "MAE validasi (10% train fold 0), alpha = 0; fairness tidak dipakai"
FIXED_PARAMS = {"SVD": dict(biased=True), "NMF": dict(biased=False)}
V2_PARAMS = {"SVD": dict(n_epochs=20, lr_all=0.005, reg_all=0.02, biased=True),
             "NMF": dict(n_epochs=50, reg_pu=0.06, reg_qi=0.06, biased=False)}
CONTROL_K = 50                        # kontrol reprodusibilitas baseline (v2)

# --------------------------------------------------------------------------
# 4. Evaluasi
# --------------------------------------------------------------------------
TOP_N_MAIN = 100                      # dasar estimasi indeks ekor
TOP_N_SMALL = 10                      # P@10, amplifikasi, sambungan ke baseline
POP_MIN_RATINGS = 20                  # ambang popularitas, sama untuk SVD dan NMF
LIKE_THRESHOLD = 4.0                  # relevan untuk P@10
DP_THRESHOLD = 4.0
AGE_REFERENCE = "25-34"
N_PRED_SAMPLES = 15

# --------------------------------------------------------------------------
# 5. EVT dan statistik
# --------------------------------------------------------------------------
EVT_K_LO, EVT_K_HI = 0.01, 0.05       # v3.2: rentang k = 1% sampai 5% dari n (4k <= 20%)
EVT_MIN_K_VALUES = 10
EVT_MAX_DROP = 0.50
# v3.3: B dipilih agar p terkecil 2/(B+1) = 0,0002 jauh di bawah ambang Holm keluarga
# terbesar (0,05/64 = 0,00078) (Davidson & MacKinnon, 2000). Indeks ekor memakai B lebih
# kecil (subset baris pertama W) karena setiap replikat memerlukan estimasi EVT penuh.
N_BOOT = int(os.environ.get("OPSI_NBOOT", 9999))
N_BOOT_EVT = int(os.environ.get("OPSI_NBOOT_EVT", 999))
N_PERM = int(os.environ.get("OPSI_NPERM", 1000))
SIG_LEVEL = 0.05
SEED_MAJORITY = 4                     # minimal 4 dari 5 seed
PROBE_FOLDS = 5
STECK_ALPHA = 0.01

# --------------------------------------------------------------------------
# 6. Aturan keputusan
# --------------------------------------------------------------------------
SAFE_MARGIN_REL = 0.10                # delta_m = 10% dari m(0)
SAFE_MAE_TOL = 0.005
MECH_COS_SVD_MAX = -0.5
MECH_COS_NMF_MIN = 0.0
MECH_MIN_ALPHAS = 3                   # dari 4 nilai alpha > 0
# v3.3: metrik disparitas berbasis MAE dan metrik ekor (EVT) dipisah ke keluarga uji sendiri
# (Bender & Lange, 2001); bobot aman hanya memakai metrik MAE karena margin 10% x xi(0)
# jauh di bawah presisi Pickands (ragam asimtotik Dekkers & de Haan, 1989).
GAP_METRICS = ["gap_gender", "gap_age_ref", "gap_occ"]
EVT_METRICS = ["dxi_gender"]
DISPARITY_METRICS = GAP_METRICS + EVT_METRICS
SAFE_METRICS = list(GAP_METRICS)
# v3.4: inferensi utama. "pooled" = bootstrap pengguna atas prediksi out-of-fold gabungan
# (BBC-CV; Tsamardinos dkk., 2018), putusan per fold menjadi uji stabilitas. "fold" = aturan
# protokol v3.3 (Holm per fold, signifikan pada >= 4 dari 5 fold) sebagai utama.
PRIMARY_INFERENCE = os.environ.get("OPSI_PRIMARY_INFERENCE", "pooled")
DIVERGENCE_RULE = "MAE latih > MAE prediktor rata-rata global pada sampel yang sama"
DISPARITY_LABELS = {"gap_gender": "|MAE F - MAE M|",
                    "gap_age_ref": "maks(MAE usia - MAE 25-34)",
                    "gap_occ": "maks - min MAE kelompok pekerjaan (tanpa tak terklasifikasi)",
                    "dxi_gender": "|xi F - xi M| (Pickands, Top-100)"}

# --------------------------------------------------------------------------
# 7. Label MovieLens 1M
# --------------------------------------------------------------------------
AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
              45: "45-49", 50: "50-55", 56: "56+"}
AGE_ORDER = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
GENDER_ORDER = ["F", "M"]
OCCUPATION_LABELS = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}
EXAMPLE_PROFILE = ("M", 18, "college/grad student")


def age_to_group(age) -> str:
    if isinstance(age, str):
        if age in AGE_ORDER:
            return age
        age = int(age)
    age = int(age)
    if age < 18:
        return "Under 18"
    for lo, hi, lab in [(18, 24, "18-24"), (25, 34, "25-34"), (35, 44, "35-44"),
                        (45, 49, "45-49"), (50, 55, "50-55")]:
        if lo <= age <= hi:
            return lab
    return "56+"


def apply_quick_mode(on: bool = True) -> None:
    """Mode cepat: grid kecil untuk menguji alur. Hasilnya tidak untuk dilaporkan."""
    global QUICK_MODE, SEEDS, KS, ALPHAS, N_BOOT, N_BOOT_EVT, N_PERM, CACHE_DIR, FIG_DIR, \
        RESULT_JSON, TUNING_KS, TUNING_DIAG_K
    QUICK_MODE = on
    if on:
        SEEDS, KS, ALPHAS = QUICK["SEEDS"], QUICK["KS"], QUICK["ALPHAS"]
        TUNING_KS = list(QUICK["KS"])
        N_BOOT, N_BOOT_EVT, N_PERM = QUICK["N_BOOT"], QUICK["N_BOOT_EVT"], QUICK["N_PERM"]
        TUNING_DIAG_K = list(QUICK["TUNING_DIAG_K"])
        # folder terpisah agar hasil uji cepat tidak pernah terpakai ulang oleh running penuh
        CACHE_DIR = OUTPUT_DIR / f"cache_quick_{CACHE_VERSION}"
        FIG_DIR = OUTPUT_DIR / "figures_quick"
        RESULT_JSON = OUTPUT_DIR / "hasil_running_QUICK_tidak_untuk_laporan.json"


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, FIG_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict:
    """Snapshot seluruh konfigurasi dan aturan untuk dicatat ke JSON."""
    keys = ["BASE_SEED", "N_FOLDS", "SEEDS", "OCC_GROUP", "OCC_GROUP_ORDER", "OCC_UNCLASSIFIED",
            "OCC_GROUP_BASIS", "INPUT_ABLATION_ATTRS", "TEST_FRAC", "VAL_FRAC", "COLD_FRAC", "KS", "ALPHAS",
            "MODELS", "ENCODINGS", "MAIN_ENCODING", "ABLATION_ENCODINGS", "DEMO_HIGH",
            "DEMO_LOW", "PROFILE_ATTRS", "SENSITIVE_ATTRS", "SCENARIOS",
            "COLD_PRIMARY_SCENARIO", "QUICK_MODE", "TUNING_GRID", "TUNING_CRITERION",
            "TUNING_KS", "FIXED_PARAMS", "V2_PARAMS", "CONTROL_K", "TOP_N_MAIN", "TOP_N_SMALL",
            "POP_MIN_RATINGS", "LIKE_THRESHOLD", "DP_THRESHOLD", "AGE_REFERENCE",
            "EVT_K_LO", "EVT_K_HI", "EVT_MIN_K_VALUES", "EVT_MAX_DROP", "N_BOOT", "N_BOOT_EVT",
            "TUNING_DIAG_K", "ABLATION_PATHS", "GAP_METRICS", "EVT_METRICS", "SAFE_METRICS",
            "DIVERGENCE_RULE", "PRIMARY_INFERENCE", "ANALYSIS_VERSION",
            "N_PERM", "SIG_LEVEL", "SEED_MAJORITY", "PROBE_FOLDS", "STECK_ALPHA",
            "SAFE_MARGIN_REL", "SAFE_MAE_TOL", "MECH_COS_SVD_MAX", "MECH_COS_NMF_MIN",
            "MECH_MIN_ALPHAS", "DISPARITY_METRICS", "DISPARITY_LABELS"]
    g = globals()
    out = {k.lower(): g[k] for k in keys}
    out["data_dir"] = str(DATA_DIR)
    return out
