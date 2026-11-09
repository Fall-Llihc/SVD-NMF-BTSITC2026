# Menjalankan Opsi C v3.4 secara Lokal (Anaconda)

> **Pembaruan v3.4 dari running v3.3 (tanpa latih ulang):** ekstrak ZIP ke folder proyek dan
> timpa berkas lama (`src/`, notebook, dokumen). ZIP tidak berisi folder `outputs/`, sehingga
> `outputs/cache_v3.3/` tidak tersentuh. Buka notebook, atur `STAGE = "analysis"` dan
> `QUICK_MODE = False`, lalu Restart & Run All (sekitar 5 sampai 10 menit; bootstrap gabungan
> memakai 9.999 replikat pada 6.040 pengguna). Kirim `outputs/hasil_running_opsiC_v3.json`.

Notebook tidak perlu diubah untuk running lokal. Notebook mencari folder `src/` dari folder
tempat notebook dibuka, dan membaca data dari `data/ml-1m/`. Sel setup Kaggle **tidak dipakai**.

## 1. Pasang Anaconda

1. Unduh dan pasang Anaconda Distribution (atau Miniconda yang lebih ringan) dari situs resmi
   Anaconda. Pada Windows, centang opsi bawaan saja.
2. Buka **Anaconda Prompt** (Windows) atau Terminal (macOS/Linux). Semua perintah di bawah
   dijalankan di sana.

## 2. Siapkan folder proyek

1. Ekstrak ZIP, misalnya ke `D:\TA\OpsiC-v3` (hindari folder OneDrive yang sedang sinkron,
   karena cache berisi ratusan berkas).
2. Salin `ratings.dat`, `users.dat`, `movies.dat` MovieLens 1M ke `data\ml-1m\`.
3. Masuk ke folder proyek:

Windows (Anaconda Prompt):

```bat
cd /d D:\TA\OpsiC-v3
```

macOS/Linux:

```bash
cd ~/TA/OpsiC-v3
```

## 3. Buat environment

```bash
conda env create -f environment.yml
conda activate biasrs-v3
python -m ipykernel install --user --name biasrs-v3 --display-name "Python (biasrs-v3)"
```

`scikit-surprise` dipasang dari conda-forge (versi 1.1.5, sudah terkompilasi untuk Windows,
macOS, dan Linux), sehingga tidak perlu Microsoft C++ Build Tools.

Cek instalasi:

```bash
python -c "import surprise, numpy, sklearn; print(surprise.__version__, numpy.__version__, sklearn.__version__)"
```

Bila environment sudah pernah dibuat dan berkas `environment.yml` berubah:
`conda env update -f environment.yml --prune`.

## 4. Uji alur (wajib, sekitar 30 sampai 45 menit)

1. Jalankan `jupyter notebook`, buka `OpsiC_v3_Demographic_SVD_NMF.ipynb`.
2. Pilih kernel **Python (biasrs-v3)** (menu Kernel > Change kernel).
3. Pada sel "0. Persiapan" atur:

```python
STAGE = "all"
QUICK_MODE = True
```

4. **Kernel > Restart & Run All**. Hasil uji cepat ditulis ke `outputs/cache_quick_v3.3/`,
   `outputs/figures_quick/`, dan `outputs/hasil_running_QUICK_tidak_untuk_laporan.json`,
   sehingga tidak pernah tercampur dengan running penuh.
5. Pastikan sel kontrol mencetak MAE SVD 0,6863 dan NMF 0,7814. Bila tidak, berhenti dan
   laporkan.

## 5. Running penuh

Folder cache v3.3 berbeda dari versi sebelumnya (`outputs/cache_v3.3/`), sehingga sel lama
tidak akan terpakai ulang. Folder `outputs/cache/`, `outputs/cache_v3.3/`, dan
`outputs/cache_quick_v3.3/` boleh dihapus (atau dipindah sebagai arsip). Sediakan ruang disk
kosong minimal 2 GB: cache v3.3 sekitar 1 GB karena bootstrap memakai 9.999 replikat.

Atur kembali:

```python
STAGE = "all"
QUICK_MODE = False
```

Running lokal tidak punya batas waktu 12 jam, jadi semua tahap bisa sekaligus. Estimasi 12 sampai
18 jam untuk v3.3 (k = 200 di grid utama, NMF hingga 400 epoch, ablation pengguna baru,
bootstrap 9.999 replikat), bergantung pada kecepatan CPU (Surprise memakai satu core). Running
dapat dipecah: jalankan `STAGE = ["tuning", "control", "baseline", "k10_20", "k50"]` lalu
`STAGE = ["k100", "k200", "ablation"]`; sel yang sudah tersimpan tidak diulang. Disarankan menjalankannya dari
terminal agar tidak bergantung pada tab browser (atau dari Jupyter di Chrome dengan Kernel > Restart & Run All):

```bash
conda activate biasrs-v3
jupyter nbconvert --to notebook --execute OpsiC_v3_Demographic_SVD_NMF.ipynb --output hasil_eksekusi_v3.ipynb --ExecutePreprocessor.timeout=-1
```

Selama running:

- Matikan sleep/hibernate laptop dan sambungkan ke charger.
- Bila running terhenti (listrik, restart, error), jalankan ulang perintah yang sama. Sel yang
  sudah tersimpan di `outputs/cache_v3.3/` dilewati, sehingga pekerjaan berlanjut dari titik terakhir.
- Untuk memantau progres tanpa membuka notebook, hitung berkas di `outputs/cache_v3.3/`: running
  penuh menghasilkan 705 berkas `.pkl` (500 grid utama, 200 ablation, 5 baseline), ditambah
  `tuning.json` dan `control.json`.
- Bila sebuah sel gagal, notebook mencetak `GAGAL ...`, menyimpan penanda `*.failed` di folder
  cache, lalu melanjutkan sel berikutnya. Menjalankan ulang notebook akan mencoba ulang sel itu.
  Daftar sel gagal tercetak di bagian analisis dan tersimpan di JSON (`failed_cells`).

## 6. Hasil

- `outputs/hasil_running_opsiC_v3.json` (kirim berkas ini untuk dianalisis)
- `outputs/figures/` (grafik 01 sampai 13)
- `hasil_eksekusi_v3.ipynb` (notebook beserta seluruh output)

Untuk mengulang analisis atau grafik saja tanpa melatih ulang: `STAGE = "analysis"`.

Sebelum mengirim JSON, pastikan sel analisis mencetak **Kelengkapan: LENGKAP** (tanpa sel hilang,
baseline lengkap, tanpa sel divergen) dan kontrol mencetak MAE SVD 0,6863 dan NMF 0,7814.
