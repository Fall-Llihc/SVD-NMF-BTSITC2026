# Opsi C v3.4: Bobot Demografi pada SVD dan NMF (MovieLens 1M)

Kode eksperimen Tugas Akhir **"Analisis Bias Demografis pada SVD dan NMF dalam Sistem
Rekomendasi Menggunakan Pendekatan Extreme Value Theory"**
Arkhan Falih Fahrie Puspita (103052330051), S1 Sains Data, Telkom University.

Versi 3.1 sampai 3.3 mengimplementasikan dokumen **Protokol Analisis Opsi C v3** beserta arahan
pembimbing (30 Oktober 2026): validasi 5-fold dan kelompok pekerjaan sebagai atribut input
ketiga. Semua aturan keputusan disimpan di `src/config.py` dan di JSON hasil.

## Perubahan v3.4 (4 November 2026, analisis saja)

Tidak ada pelatihan ulang: cache tetap `outputs/cache_v3.3/`. Cukup ganti kode, lalu jalankan
notebook dengan `STAGE = "analysis"` (sekitar 5 sampai 10 menit).

| Aspek | v3.3 | v3.4 | Landasan |
|---|---|---|---|
| Inferensi | Holm per fold, signifikan pada >= 4 dari 5 fold | ditambah bootstrap pengguna atas prediksi out-of-fold gabungan (satu matriks bobot untuk semua konfigurasi dan kedua jalur); status klaim robust / gabungan saja / per fold saja | Tsamardinos dkk. (2018), BBC-CV |
| Skala gap | gap absolut | ditambah gap relatif (gap / MAE): F3b-relatif, F8-relatif | perbandingan antar prediktor dengan MAE berbeda |
| EVT | kesesuaian Pickands vs Moment pada k yang sama | ditambah kedalaman data yang sama (Moment k' = 4k), uji plateau, saturasi eksposur, putusan robust/tidak | Embrechts dkk. (1997); de Haan & Ferreira (2006) |
| Tuning | penanda batas grid | ditambah selisih MAE validasi terbaik vs terbaik kedua dan rentang gap lintas k | |

Pilihan inferensi utama ada di `src/config.py`: `PRIMARY_INFERENCE = "pooled"` (bawaan) atau
`"fold"`. Keduanya selalu dihitung dan disimpan di JSON (`analysis.pooled`,
`analysis.inference_summary`); pilihan ini hanya menentukan putusan yang ditonjolkan di insight.

## Perubahan v3.3 (3 November 2026)

| Aspek | v3.2 | v3.3 | Landasan |
|---|---|---|---|
| Bootstrap | B = 500, p terkecil 0,004; keluarga 24 sampai 64 uji mustahil signifikan setelah Holm | B = 9999 (metrik MAE), B = 999 (EVT, baris pertama bobot yang sama) | Davidson & MacKinnon (2000) |
| Keluarga uji | Delta-xi bercampur dengan gap MAE | keluarga EVT terpisah (F1-EVT, F2-EVT); putusan H1 dari 3 gap MAE | Bender & Lange (2001) |
| Pembanding EVT | Hill dan Moment | Moment saja (Hill hanya sah untuk xi > 0); presisi dan MDE dilaporkan | Hill (1975); Dekkers, Einmahl, & de Haan (1989) |
| Bobot aman | termasuk Delta-xi, encoding full saja | gap MAE saja; per encoding (full, member5, membermean) dan per jalur | ragam asimtotik Pickands (Dekkers & de Haan, 1989) |
| Pengguna baru | F3 membandingkan dua populasi dengan bobot independen | F3 berpasangan (pengguna sama); F3b profil demografi vs Default | bootstrap berpasangan |
| Ablation | encoding hanya-anggota di warm; NMF dengan bias | encoding hanya-anggota di warm dan cold; F5b; NMF dengan bias dihapus | dokumentasi Surprise |
| Perancu | tidak ada | BaselineOnly per fold (F8) dan strata jumlah rating | Koren (2010); Ekstrand dkk. (2018) |
| H2 | putusan konfirmatori | deskriptif (titik k x alpha tidak saling bebas) | |
| Grid | k maks 100; optimum tuning di batas grid | k = 200 masuk grid utama; grid tuning diperluas; diagnostik k = 300 | |
| Cache | tidak mencatat hyperparameter | nama berkas memuat sidik hyperparameter; tulis atomik; `outputs/cache_v3.3/` | |
| Divergensi | tidak diperiksa | MAE latih > MAE rata-rata global = divergen, dikeluarkan dari uji | |

## Perubahan v3.2 (2 November 2026)

| Aspek | v3.1 | v3.2 |
|---|---|---|
| Fold-in pengguna baru | lambda dipakai sekali | lambda x jumlah pseudo-observasi, sesuai fungsi objektif pelatihan |
| Rentang k EVT | 5% sampai 20% dari n | 1% sampai 5% dari n; jumlah eksposur disimpan untuk uji sensitivitas; Moment sebagai uji ketahanan utama |
| Hipotesis Kedua | Spearman lintas k x alpha | Spearman parsial dengan alpha dikontrol |
| Grid tuning | SVD lr <= 0,01, epoch <= 40; NMF epoch <= 100 | SVD lr <= 0,02, epoch <= 80; NMF epoch <= 200; k = 200 diagnostik; parameter di batas grid ditandai |
| Data tidak lengkap | putusan tertulis "tidak berbeda" | putusan "data belum tersedia/lengkap", ditambah analysis.completeness |
| Cache | outputs/cache/ | outputs/cache_v3.2/ (cache lama tidak terpakai) |
| Sel gagal | running berhenti | sel dicatat sebagai `*.failed`, running berlanjut, dicoba ulang pada running berikutnya |
| NMF dengan bias (ablation) | ZeroDivisionError menghentikan running | epoch dipangkas setengah (minimal 25) bila gagal; epoch terpakai dicatat (`fallback_epochs`) |

## 0. Kelompok pekerjaan

| Kelompok | Kategori MovieLens 1M | Landasan |
|---|---|---|
| Keahlian tinggi | executive/managerial, academic/educator, technician/engineer, programmer, writer, artist, doctor/health care, scientist, lawyer | ISCO-08 kelompok utama 1-3 (skill level 3-4), ILO |
| Keahlian menengah | sales/marketing, clerical/admin, customer service, tradesman/craftsman, farmer | ISCO-08 kelompok utama 4-8 (skill level 2), ILO; sales/marketing ambigu |
| Pelajar/mahasiswa | K-12 student, college/grad student | Resolusi statistik ketenagakerjaan ILO (ICLS): status aktivitas "studies" |
| Tidak bekerja | retired, homemaker, unemployed | Resolusi ICLS: not in employment |
| Tidak terklasifikasi | other, self-employed | tanpa informasi; self-employed adalah status dalam pekerjaan (ICSE-18) |

Kelompok tak terklasifikasi dipakai sebagai nilai input, tetapi tidak dihitung dalam metrik
disparitas pekerjaan.

---

## 1. Perubahan utama dari v2

| Aspek | v2 | v3 |
|---|---|---|
| Validasi | 1 split (seed 42) | 5-fold: rating per pengguna dan partisi pengguna cold-start |
| Atribut input | gender, usia | gender, usia, kelompok pekerjaan (8 kategori input pengguna baru) |
| Hyperparameter | tetap (k = 50, nilai baseline) | tuning per model x k pada data validasi, alpha = 0, kriteria MAE |
| Grid | alpha saja | k {10, 20, 50, 100, 200} x alpha {0, 1, 2, 4, 8} x 5 fold (v3.3) |
| Encoding pseudo-item | anggota 5, bukan anggota 1 | + ablation hanya-anggota (nilai 5) dan hanya-anggota (nilai rata-rata pengguna) |
| Rumus Pickands | urutan menaik, k = n/4 | urutan menurun (memperbaiki Persamaan 2.16), rentang k 5% sampai 20%, aturan ties, Hill dan Moment sebagai pembanding |
| Top-N eksposur | Top-10 | Top-100 (utama) dan Top-10, ambang popularitas 20 rating untuk kedua model |
| Klaim pengguna baru | gap mentah | excess gap relatif (dikoreksi uji permutasi), CI bootstrap, uji cold vs warm |
| Probe laten | tanpa pembanding | baseline probe kelompok pekerjaan dari gender dan usia (hanya pada konfigurasi tanpa pekerjaan sebagai input) |
| Ablation input | tidak ada | gender + usia saja (efek marginal pekerjaan, F7) |
| Uji statistik | CI per perbandingan | bootstrap berpasangan lintas konfigurasi, Holm per keluarga, mayoritas 4 dari 5 fold |
| Trade-off | titik kompromi | front Pareto dengan dominasi kokoh antar fold |
| Kontrol baseline | otomatis pada alpha = 0 | pelatihan kontrol terpisah dengan hyperparameter v2 |

---

## 2. Struktur proyek

```
OpsiC-v3/
├── OpsiC_v3_Demographic_SVD_NMF.ipynb   satu-satunya notebook (variabel STAGE)
├── README.md / SETUP_KAGGLE.md / SETUP_LOKAL.md
├── requirements.txt / environment.yml
├── src/
│   ├── config.py       konstanta dan seluruh aturan protokol
│   ├── data.py         memuat data, kelompok pekerjaan, 5-fold, kontrol baseline
│   ├── eda.py          analisis eksploratori (Subbab 3.3)
│   ├── features.py     pseudo-item (gender, usia, pekerjaan), tiga varian encoding, bobot relatif
│   ├── models.py       pelatihan, prediksi tervektorisasi, Top-N, fold-in
│   ├── tuning.py       tuning per model x k
│   ├── evt.py          Pickands, Hill, Moment
│   ├── stats_tools.py  bootstrap berpasangan, permutasi, Holm
│   ├── latent.py       ruang laten, probe, baseline probe, cosinus pseudo-item
│   ├── evaluation.py   evaluasi satu sel (warm atau cold-start)
│   ├── experiment.py   grid bertahap dengan cache, kontrol baseline
│   ├── analysis.py     seluruh aturan keputusan (F1 sampai F6, Pareto, bobot aman)
│   ├── report.py       insight otomatis
│   ├── plots.py        grafik 01 sampai 10
│   └── recorder.py     JSON hasil
├── data/ml-1m/          ratings.dat, users.dat, movies.dat
├── data/reference/      hasil baseline untuk kontrol reprodusibilitas
└── outputs/             cache_v3.3/, figures/, hasil_running_opsiC_v3.json
```

---

## 3. Menjalankan

1. Letakkan `ratings.dat`, `users.dat`, `movies.dat` di `data/ml-1m/`.
2. `pip install -r requirements.txt`
3. Buka notebook, atur `STAGE`, lalu **Run All**.

| STAGE | Isi |
|---|---|
| `"tuning"` | tuning hyperparameter (menghasilkan `outputs/cache_v3.3/tuning.json` dan k terpilih) |
| `"control"` | kontrol reprodusibilitas baseline |
| `"baseline"` | prediktor BaselineOnly per fold (v3.3) |
| `"k50"`, `"k10_20"`, `"k100"`, `"k200"` | grid utama per kelompok k |
| `"ablation"` | encoding hanya-anggota (warm + cold) dan input tanpa pekerjaan, pada k terpilih |
| `"all"` | semua tahap berurutan |
| `"analysis"` | analisis saja dari cache yang ada |
| daftar | misalnya `["tuning", "control", "k50", "k10_20"]` |

Setiap sel disimpan di `outputs/cache_v3.3/`; menjalankan ulang melewati sel yang sudah ada.
Bagian analisis selalu berjalan di akhir notebook dan menulis
`outputs/hasil_running_opsiC_v3.json`. Putusan akhir hanya sah setelah semua tahap selesai.

`QUICK_MODE = True` menjalankan grid kecil (2 fold, k 10 dan 20, alpha 0, 1, 4) untuk menguji
alur. Hasil mode cepat tidak untuk dilaporkan.

---

## 4. Isi JSON hasil

| Kunci | Isi |
|---|---|
| `config` | seluruh konstanta dan aturan keputusan |
| `tuning`, `control` | hasil tuning, k terpilih, diagnostik batas grid, kontrol baseline |
| `dataset`, `eda`, `preprocessing` | info data, EDA, ringkasan split per fold |
| `analysis.table` | satu baris per sel: jalur, kategori input, model, k, alpha, seed (= indeks fold), encoding, variant, seluruh metrik titik |
| `analysis.aggregate` | rata-rata dan sd antar fold |
| `analysis.h1`, `analysis.h2` | uji dan putusan hipotesis |
| `analysis.alpha_effect`, `safe_alpha`, `new_users`, `new_user_profile`, `amplification`, `mechanism`, `encoding_effect`, `leakage`, `occupation_effect`, `confound` | keluarga uji F2 sampai F8 (keluarga EVT di kunci `evt` masing-masing) |
| `analysis.evt_precision` | SD bootstrap dan asimtotik indeks ekor, MDE Delta-xi |
| `analysis.completeness` | sel hilang, baseline hilang, sel divergen; putusan sah hanya bila `complete = true` |
| `analysis.pareto`, `analysis.evt`, `analysis.popularity` | trade-off, diagnostik EVT, sensitivitas ambang |
| `analysis.rankings` | Top-10 per profil gender x usia x kelompok pekerjaan (fold 0, k terpilih) |
| `analysis.deviations` | penyesuaian protokol saat implementasi |
| `insights` | ringkasan otomatis |
