"""
src
===
PERAN PAKET
    Kode eksperimen Opsi C v3 (MovieLens 1M): bobot demografi alpha pada SVD dan NMF
    dengan grid k x alpha x seed, tuning, ablation encoding, estimator EVT yang
    diperbaiki, dan aturan keputusan yang dikunci sebelum running.

PETA MODUL (urutan alur kerja)
    config       konstanta dan seluruh aturan protokol
    utils        konversi JSON, pengukur waktu, info lingkungan
    data         memuat data dan membentuk split per seed
    eda          analisis eksploratori (Subbab 3.3)
    features     pseudo-item demografis, tiga varian encoding, bobot relatif
    models       pelatihan, prediksi tervektorisasi, Top-N, fold-in
    tuning       hyperparameter tuning per model x k
    evt          Pickands, Hill, Moment (urutan menurun, aturan ties)
    stats_tools  bootstrap berpasangan, permutasi, Holm
    latent       inspeksi ruang laten, probe, baseline probe, cosinus pseudo-item
    evaluation   evaluasi satu sel (warm atau cold-start)
    experiment   grid bertahap dengan cache, kontrol baseline
    analysis     agregasi antar seed, Pareto, H1, H2, bobot aman, klaim pengguna baru,
                 amplifikasi, mekanisme, kebocoran
    plots        grafik
    recorder     pencatat JSON hasil
"""
