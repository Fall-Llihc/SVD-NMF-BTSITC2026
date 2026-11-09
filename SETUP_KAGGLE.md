# Menjalankan Opsi C v3.3 di Kaggle

Versi 3.3 memakai k = 200 di grid utama, grid tuning yang lebih luas (NMF hingga 400 epoch),
ablation pengguna baru, dan bootstrap 9.999 replikat, sehingga total komputasi sekitar 12 sampai
18 jam. Batas satu sesi Kaggle adalah 12 jam dan output sebuah versi hanya tersimpan bila sesi
selesai, jadi **skema tiga sesi (Langkah 3B) wajib**. Running lokal (`SETUP_LOKAL.md`) lebih
disarankan. Cache sesi sebelumnya dipasang sebagai input sesi berikutnya, sehingga tidak ada
pekerjaan yang diulang. Batas output Kaggle (20 GB) cukup untuk cache v3.3 (sekitar 1 GB).

## Langkah 1: Dataset input

| Dataset | Isi |
|---|---|
| `movielens-1m-dat` (sudah ada) | `ratings.dat`, `users.dat`, `movies.dat` |
| `run-py-v3` (baru) | isi folder `kaggle_run-py-v3/` di ZIP ini (seluruh `.py` dari `src/` ditambah `baseline_v3_combined_analysis.json`), siap diunggah apa adanya |

Buat dataset `run-py-v3` yang baru; jangan menimpa `run-py` v2 agar kode lama tidak tercampur.

## Langkah 2: Notebook

1. **File > Import Notebook**, unggah `OpsiC_v3_Demographic_SVD_NMF.ipynb`.
2. **Add Input**: `movielens-1m-dat` dan `run-py-v3`.
3. Settings: Accelerator **None**, Internet **On**, Persistence **Files**.
4. Sisipkan sel berikut di **paling atas** notebook:

```python
!pip install -q scikit-surprise==1.1.5

import os, shutil, glob
from pathlib import Path

INPUT = "/kaggle/input"
WORK = Path("/kaggle/working/OpsiC-v3")

def cari(nama):
    hasil = sorted(glob.glob(f"{INPUT}/**/{nama}", recursive=True))
    assert hasil, f"{nama} tidak ditemukan di {INPUT}"
    return [Path(h) for h in hasil]

# 1. kode v3: folder yang memuat config.py, evt.py, dan tuning.py
code_dir = next(p.parent for p in cari("config.py")
                if (p.parent / "evt.py").exists() and (p.parent / "tuning.py").exists())
for sub in ["src", "data/ml-1m", "data/reference", "outputs/cache_v3.3"]:
    (WORK / sub).mkdir(parents=True, exist_ok=True)
for f in code_dir.glob("*.py"):
    shutil.copy(f, WORK / "src" / f.name)

# 2. dataset MovieLens 1M
for fn in ["ratings.dat", "users.dat", "movies.dat"]:
    shutil.copy(cari(fn)[0], WORK / "data" / "ml-1m" / fn)

# 3. referensi baseline untuk kontrol reprodusibilitas
ref = glob.glob(f"{INPUT}/**/baseline_v3_combined_analysis.json", recursive=True)
if ref:
    shutil.copy(ref[0], WORK / "data" / "reference" / "baseline_v3_combined_analysis.json")

# 4. cache dari sesi sebelumnya (bila output sesi lama dipasang sebagai input)
n_cache = 0
for f in glob.glob(f"{INPUT}/**/outputs/cache_v3.3/*", recursive=True):
    dst = WORK / "outputs" / "cache_v3.3" / Path(f).name
    if not dst.exists():
        shutil.copy(f, dst); n_cache += 1

os.chdir(WORK)
print("kode dari :", code_dir)
print("src/      :", sorted(os.listdir("src")))
print("referensi :", "ada" if ref else "TIDAK ADA (kontrol baseline dilewati)")
print("cache     :", n_cache, "berkas disalin dari sesi sebelumnya")
```

## Langkah 3A: Satu sesi (tidak bisa untuk v3.3)

Total komputasi melebihi 12 jam; gunakan Langkah 3B.

## Langkah 3B: Tiga sesi (wajib)

| Sesi | STAGE | Estimasi |
|---|---|---|
| 1 | `["tuning", "control", "baseline", "k10_20", "k50"]` | 6 sampai 9 jam |
| 2 | `["k100", "k200"]` | 5 sampai 8 jam |
| 3 | `["ablation"]` | 3 sampai 5 jam, lalu analisis akhir |

Setiap sesi: atur `STAGE` pada sel "0. Persiapan", lalu **Save & Run All (Commit)**.

## Langkah 4: Sesi 2 dan 3

1. Buka notebook yang sama, **Add Input > Your Work > Notebooks**, pilih versi sesi sebelumnya
   (output-nya berisi `OpsiC-v3/outputs/cache_v3.3/`). Sel setup di atas otomatis menyalin cache
   itu. Pada sesi 3, pasang output sesi 2 (yang sudah memuat cache sesi 1 dan 2).
2. Atur `STAGE` sesuai tabel, lalu **Save & Run All (Commit)**. Di akhir sesi 3 analisis berjalan
   pada seluruh cache dan menulis JSON final.

Bila sebuah sesi gagal di tengah jalan, pasang output versi terakhir yang berhasil sebagai input
lalu jalankan ulang dengan `STAGE` yang sama; sel yang sudah ada di cache dilewati. Bila hanya
ingin mengulang analisis atau grafik, gunakan `STAGE = "analysis"`.

## Langkah 5: Unduh hasil

Dari tab Output versi terakhir:

- `OpsiC-v3/outputs/hasil_running_opsiC_v3.json` (wajib, untuk dianalisis)
- `OpsiC-v3/outputs/figures/`
- notebook yang sudah tereksekusi

Cek di log awal bahwa kontrol reprodusibilitas (split lama 80:20 seed 42, hyperparameter v2)
mencetak MAE SVD 0,6863 dan NMF 0,7814. Bila tidak, hentikan dan laporkan, karena berarti data
berbeda dari baseline. Grid utama memakai 5-fold, sehingga angkanya memang tidak sama dengan
baseline.
