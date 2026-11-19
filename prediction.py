"""Prediksi risiko dropout siswa Jaya Jaya Institut.

Berkas ini tidak bergantung pada Streamlit. Isinya: memuat model dan daftar
fitur, memeriksa masukan, menghitung probabilitas dropout, dan membaca CSV.
Dipakai oleh app.py, notebook, dan dapat dijalankan langsung:

    python prediction.py

Tanpa argumen, skrip memprediksi siswa berstatus Enrolled pada data/data.csv
dan mencetak dua baris ringkasan.
"""

import io
import json
import math
from pathlib import Path

import joblib
import pandas as pd

BERKAS_DATA = "data/data.csv"
PEMISAH = ";"
BERKAS_MODEL = "model/model_dropout.joblib"
BERKAS_FITUR = "model/fitur.json"
KOLOM_TARGET = "Status"
KELAS_POSITIF = "Dropout"
SEED = 42
AMBANG = 0.5
BATAS_SEDANG = 0.30
BATAS_TINGGI = 0.60

FOLDER_SKRIP = Path(__file__).resolve().parent
LABEL_POSITIF = "Dropout"
LABEL_NEGATIF = "Tidak dropout"


def muat_model(folder=None):
    """Muat pipeline model dan isi fitur.json; kembalikan (pipeline, fitur_dict).

    folder: folder yang berisi subfolder model/. Bila kosong, dipakai folder
    tempat berkas ini berada, sehingga dapat dijalankan dari folder mana pun.
    """
    dasar = Path(folder) if folder else FOLDER_SKRIP
    jalur_model = dasar / BERKAS_MODEL
    jalur_fitur = dasar / BERKAS_FITUR
    for jalur, nama in ((jalur_model, BERKAS_MODEL), (jalur_fitur, BERKAS_FITUR)):
        if not jalur.is_file():
            raise FileNotFoundError(
                "Berkas %s tidak ditemukan. Jalankan notebook.ipynb sampai bagian "
                "penyimpanan model, lalu letakkan hasilnya pada folder model/." % nama
            )
    pipeline = joblib.load(jalur_model)
    with open(jalur_fitur, encoding="utf-8") as f:
        fitur_dict = json.load(f)
    return pipeline, fitur_dict


def _angka(nilai):
    """Ubah nilai menjadi float, atau None bila bukan angka yang sah."""
    if isinstance(nilai, bool):
        return None
    if isinstance(nilai, (int, float)):
        x = float(nilai)
    else:
        try:
            x = float(str(nilai).strip().replace(",", "."))
        except ValueError:
            return None
    return x if math.isfinite(x) else None


def _kosong(nilai):
    """True bila nilai belum diisi (None, teks kosong, atau NaN)."""
    if nilai is None:
        return True
    if isinstance(nilai, str):
        return nilai.strip() == "" or nilai.strip().lower() == "nan"
    if isinstance(nilai, float):
        return math.isnan(nilai)
    return False


def periksa_input(baris, fitur_dict):
    """Periksa satu baris masukan terhadap fitur.json; kembalikan daftar pesan.

    Daftar kosong berarti masukan sah. Pesan memakai bahasa Indonesia dan
    menyebut label kolom: kolom hilang, bukan angka, bukan bilangan bulat,
    di luar rentang data latih, atau pilihan yang tidak dikenal.
    """
    pesan = []
    for f in fitur_dict["fitur"]:
        nama = f["nama"]
        label = f.get("label", nama)
        if nama not in baris or _kosong(baris[nama]):
            pesan.append("%s: belum diisi." % label)
            continue
        x = _angka(baris[nama])
        if x is None:
            pesan.append("%s: harus berupa angka." % label)
            continue
        if f["tipe"] in ("biner", "kategori"):
            sah = {float(o["nilai"]) for o in f.get("opsi", [])}
            if x not in sah:
                pesan.append("%s: pilihan tidak dikenal." % label)
            continue
        if f["tipe"] == "int" and x != int(x):
            pesan.append("%s: harus bilangan bulat." % label)
            continue
        kecil, besar = f.get("min"), f.get("max")
        if (kecil is not None and x < kecil) or (besar is not None and x > besar):
            pesan.append(
                "%s: nilai %g di luar rentang %g sampai %g." % (label, x, kecil, besar)
            )
    return pesan


def tingkat_risiko(p):
    """Tingkat risiko dari probabilitas: Rendah, Sedang, atau Tinggi."""
    if p >= BATAS_TINGGI:
        return "Tinggi"
    if p >= BATAS_SEDANG:
        return "Sedang"
    return "Rendah"


def prediksi(df, pipeline, fitur_dict):
    """Tambahkan probabilitas_dropout, prediksi, dan tingkat_risiko pada salinan df.

    Kolom fitur dibaca sebagai angka (koma desimal diterima). Baris yang
    belum diperiksa dengan periksa_input sebaiknya tidak dikirim ke sini.
    """
    nama = [f["nama"] for f in fitur_dict["fitur"]]
    x = pd.DataFrame(
        {
            k: pd.to_numeric(df[k].astype(str).str.strip().str.replace(",", ".", regex=False))
            for k in nama
        }
    )
    kelas = list(pipeline.classes_)
    kolom_positif = kelas.index(1)
    proba = pipeline.predict_proba(x)[:, kolom_positif]
    ambang = fitur_dict.get("ambang", AMBANG)
    hasil = df.copy()
    hasil["probabilitas_dropout"] = [round(float(p), 4) for p in proba]
    hasil["prediksi"] = [
        LABEL_POSITIF if p >= ambang else LABEL_NEGATIF for p in hasil["probabilitas_dropout"]
    ]
    hasil["tingkat_risiko"] = [tingkat_risiko(p) for p in hasil["probabilitas_dropout"]]
    return hasil


def baca_csv(berkas, fitur_dict=None):
    """Baca CSV dengan pemisah ; atau , (dideteksi dari baris judul).

    berkas: jalur berkas atau objek berkas (misalnya unggahan Streamlit).
    Semua sel dibaca sebagai teks; angka diperiksa oleh periksa_input.
    Bila fitur_dict diberikan dan ada kolom yang hilang, ValueError
    menyebut nama kolomnya.
    """
    mentah = berkas.read() if hasattr(berkas, "read") else Path(berkas).read_bytes()
    try:
        teks = mentah.decode("utf-8-sig")
    except UnicodeDecodeError:
        teks = mentah.decode("latin-1")
    if not teks.strip():
        raise ValueError("Berkas kosong.")
    judul = teks.splitlines()[0]
    pemisah = "," if judul.count(",") > judul.count(";") else PEMISAH
    df = pd.read_csv(io.StringIO(teks), sep=pemisah, dtype=str, keep_default_na=False)
    df.columns = [str(k).strip() for k in df.columns]
    if fitur_dict is not None:
        hilang = [f["nama"] for f in fitur_dict["fitur"] if f["nama"] not in df.columns]
        if hilang:
            raise ValueError("Kolom berikut tidak ditemukan pada berkas: %s." % ", ".join(hilang))
    if len(df) == 0:
        raise ValueError("Berkas tidak memuat baris data.")
    return df


def format_ringkasan(jumlah, berisiko):
    """Dua baris ringkasan yang dicetak saat skrip dijalankan langsung."""
    persen = 100.0 * berisiko / jumlah if jumlah else 0.0
    return [
        "Siswa aktif diprediksi: %d" % jumlah,
        "Berisiko dropout: %d (%.1f %%)" % (berisiko, persen),
    ]


def main():
    """Prediksi siswa berstatus Enrolled pada data/data.csv dan cetak ringkasannya."""
    pipeline, fitur_dict = muat_model()
    data = pd.read_csv(FOLDER_SKRIP / BERKAS_DATA, sep=PEMISAH, encoding="utf-8-sig")
    aktif = data[data[KOLOM_TARGET] == "Enrolled"]
    hasil = prediksi(aktif, pipeline, fitur_dict)
    berisiko = int((hasil["prediksi"] == LABEL_POSITIF).sum())
    for baris in format_ringkasan(len(hasil), berisiko):
        print(baris)


if __name__ == "__main__":
    main()
