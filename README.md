# Perkiraan Risiko Dropout Siswa

Prototype Streamlit untuk Jaya Jaya Institut (institusi fiktif dari proyek latihan). Aplikasi memperkirakan peluang
seorang siswa dropout dari 13 isian akademik, keuangan, dan profil, sehingga siswa yang berisiko dapat dibimbing lebih
awal.

Aplikasi online: https://jaya-jaya-institut-dropout-ecs63k8ckwmdxegclvmdbs.streamlit.app

## Fitur

- **Satu siswa:** formulir dalam tiga kelompok, hasil berupa probabilitas dropout, tingkat risiko (Rendah, Sedang,
  Tinggi), dan saran tindak lanjut. Masukan yang tidak sah ditolak dengan pesan yang jelas.
- **Banyak siswa:** unggah CSV (pemisah titik koma atau koma), hasil diurutkan dari risiko tertinggi dan dapat diunduh.
  Templat CSV tersedia pada tab tersebut.

## Menjalankan secara lokal

Dibutuhkan Python 3.11.

```
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501.

## Isi repositori

| Berkas | Fungsi |
|--------|--------|
| `app.py` | Antarmuka Streamlit |
| `prediction.py` | Memuat model, memeriksa masukan, menghitung probabilitas |
| `model/model_dropout.joblib` | Pipeline scikit-learn (pra-pemrosesan dan klasifier) |
| `model/fitur.json` | Daftar isian, rentang, pilihan, dan metrik model; formulir dibangun dari berkas ini |
| `requirements.txt` | Versi pustaka yang dipakai |

Hasil model adalah perkiraan statistik dari data siswa terdahulu, bukan keputusan.
