"""Prototype Streamlit: perkiraan risiko dropout siswa Jaya Jaya Institut.

Hanya antarmuka. Memuat model, memeriksa masukan, dan menghitung probabilitas
dilakukan oleh prediction.py. Formulir dibangun dari model/fitur.json, jadi
daftar kolom pada layar selalu sama dengan daftar kolom yang dipakai model.

Jalankan dengan:  streamlit run app.py
"""

import importlib.metadata as metadata
import platform

import pandas as pd
import streamlit as st

import prediction as pr

JUDUL_KELOMPOK = {
    "akademik": "Data akademik",
    "keuangan": "Data keuangan",
    "profil": "Profil siswa",
}

TINDAK_LANJUT = {
    "Rendah": "Belum perlu tindakan khusus. Tetap pantau hasil semester berikutnya.",
    "Sedang": (
        "Perlu perhatian. Hubungi siswa, tanyakan kendala belajar atau biaya, "
        "dan periksa lagi setelah semester berikutnya."
    ),
    "Tinggi": (
        "Prioritaskan untuk bimbingan. Jadwalkan pertemuan dengan dosen wali, "
        "tinjau beasiswa atau keringanan biaya, dan susun rencana studi bersama siswa."
    ),
}

KOLOM_HASIL = ["probabilitas_dropout", "prediksi", "tingkat_risiko"]


@st.cache_resource
def muat():
    """Muat model dan fitur.json satu kali untuk seluruh sesi."""
    return pr.muat_model()


def versi(paket):
    """Versi paket yang terpasang, atau tanda strip bila tidak ditemukan."""
    try:
        return metadata.version(paket)
    except metadata.PackageNotFoundError:
        return "-"


def format_persen(p):
    """Probabilitas 0..1 menjadi teks persen satu desimal."""
    return "%.1f %%" % (100.0 * p)


def buat_masukan(f):
    """Buat satu widget masukan dari satu entri fitur.json dan kembalikan nilainya."""
    nama, label, bantuan = f["nama"], f["label"], f.get("bantuan") or None
    kunci = "in_" + nama
    if f["tipe"] == "int":
        return st.number_input(
            label, min_value=int(f["min"]), max_value=int(f["max"]),
            value=int(f["default"]), step=1, help=bantuan, key=kunci,
        )
    if f["tipe"] == "float":
        return st.number_input(
            label, min_value=float(f["min"]), max_value=float(f["max"]),
            value=float(f["default"]), step=0.1, help=bantuan, key=kunci,
        )
    nilai = [o["nilai"] for o in f["opsi"]]
    teks = {o["nilai"]: o["label"] for o in f["opsi"]}
    indeks = nilai.index(f["default"]) if f["default"] in nilai else 0
    if f["tipe"] == "biner":
        return st.radio(
            label, nilai, index=indeks, format_func=lambda v: teks.get(v, str(v)),
            horizontal=True, help=bantuan, key=kunci,
        )
    return st.selectbox(
        label, nilai, index=indeks, format_func=lambda v: teks.get(v, str(v)),
        help=bantuan, key=kunci,
    )


def tampilkan_hasil(p, tingkat):
    """Tampilkan probabilitas, tingkat risiko, dan saran tindak lanjut."""
    st.subheader("Hasil perkiraan")
    st.metric("Probabilitas dropout", format_persen(p))
    st.progress(min(max(p, 0.0), 1.0))
    if tingkat == "Tinggi":
        st.error("Tingkat risiko: Tinggi")
    elif tingkat == "Sedang":
        st.warning("Tingkat risiko: Sedang")
    else:
        st.success("Tingkat risiko: Rendah")
    st.markdown("**Saran tindak lanjut.** " + TINDAK_LANJUT[tingkat])
    st.caption(
        "Hasil ini perkiraan statistik dari data siswa terdahulu, bukan keputusan. "
        "Gunakan sebagai bahan percakapan dengan siswa."
    )


def tab_satu_siswa(pipeline, fitur):
    """Formulir untuk satu siswa, dikelompokkan menurut kelompok pada fitur.json."""
    st.markdown(
        "Isi data siswa pada tiga kelompok di bawah, lalu tekan tombol prediksi. "
        "Nilai awal adalah nilai tengah data latih."
    )
    daftar = fitur["fitur"]
    with st.form("form_satu_siswa"):
        isian = {}
        for kelompok in ("akademik", "keuangan", "profil"):
            anggota = [f for f in daftar if f["kelompok"] == kelompok]
            if not anggota:
                continue
            st.subheader(JUDUL_KELOMPOK[kelompok])
            for awal in range(0, len(anggota), 2):
                # satu baris kolom per pasangan: urutan di layar sempit sama dengan urutan fitur.json
                kolom = st.columns(2)
                for k, f in zip(kolom, anggota[awal:awal + 2]):
                    with k:
                        isian[f["nama"]] = buat_masukan(f)
        kirim = st.form_submit_button("Prediksi risiko dropout")
    if not kirim:
        return
    pesan = pr.periksa_input(isian, fitur)
    if pesan:
        st.error("Masukan belum sah. Perbaiki hal berikut lalu coba lagi:")
        for m in pesan:
            st.warning(m)
        return
    hasil = pr.prediksi(pd.DataFrame([isian]), pipeline, fitur)
    tampilkan_hasil(float(hasil["probabilitas_dropout"].iloc[0]), hasil["tingkat_risiko"].iloc[0])


def teks_templat(fitur):
    """Isi CSV templat: satu baris judul dan satu baris contoh berisi nilai awal."""
    nama = [f["nama"] for f in fitur["fitur"]]
    contoh = [str(f["default"]) for f in fitur["fitur"]]
    return (";".join(nama) + "\n" + ";".join(contoh) + "\n").encode("utf-8")


def tampilkan_tabel(hasil, fitur):
    """Tampilkan tabel hasil; bila pustaka tabel Streamlit (pyarrow) tidak dapat dimuat, pakai tabel teks."""
    try:
        st.dataframe(hasil, hide_index=True)
    except ImportError:
        st.caption("Tabel interaktif tidak tersedia di lingkungan ini; menampilkan 200 baris pertama dengan kolom hasil dan kolom model.")
        kolom = KOLOM_HASIL + [f["nama"] for f in fitur["fitur"]]
        teks = hasil[kolom].head(200).astype(str)
        baris = ["| " + " | ".join(teks.columns) + " |", "|" + " --- |" * len(teks.columns)]
        baris += ["| " + " | ".join(r) + " |" for r in teks.values.tolist()]
        st.markdown("\n".join(baris))


def tab_banyak_siswa(pipeline, fitur):
    """Unggah CSV berisi banyak siswa, tampilkan hasil menurut risiko, dan unduh."""
    st.markdown(
        "Unggah berkas CSV (pemisah titik koma atau koma) yang memuat kolom yang sama "
        "dengan templat. Kolom lain pada berkas dibiarkan."
    )
    st.download_button(
        "Unduh templat CSV", data=teks_templat(fitur),
        file_name="templat_siswa.csv", mime="text/csv",
    )
    berkas = st.file_uploader("Unggah berkas CSV", type=["csv"])
    if berkas is None:
        st.info("Belum ada berkas yang diunggah.")
        return
    try:
        data = pr.baca_csv(berkas, fitur)
    except ValueError as e:
        st.error(str(e))
        return
    sah, masalah = [], []
    for indeks, baris in enumerate(data.to_dict("records")):
        pesan = pr.periksa_input(baris, fitur)
        if pesan:
            masalah.append("Baris %d: %s" % (indeks + 2, " ".join(pesan)))
        else:
            sah.append(indeks)
    if masalah:
        st.warning(
            "%d dari %d baris dilewati karena masukannya belum sah." % (len(masalah), len(data))
        )
        with st.expander("Lihat baris yang dilewati"):
            for m in masalah[:50]:
                st.markdown("- " + m)
            if len(masalah) > 50:
                st.caption("Hanya 50 baris pertama yang ditampilkan.")
    if not sah:
        st.error("Tidak ada baris yang dapat diprediksi.")
        return
    hasil = pr.prediksi(data.iloc[sah], pipeline, fitur)
    hasil = hasil.sort_values("probabilitas_dropout", ascending=False)
    hasil = hasil[KOLOM_HASIL + [k for k in hasil.columns if k not in KOLOM_HASIL]]
    st.subheader("Ringkasan")
    kolom = st.columns(3)
    for k, tingkat in zip(kolom, ("Tinggi", "Sedang", "Rendah")):
        k.metric("Risiko " + tingkat, int((hasil["tingkat_risiko"] == tingkat).sum()))
    st.subheader("Daftar siswa, risiko tertinggi di atas")
    tampilkan_tabel(hasil, fitur)
    st.download_button(
        "Unduh hasil (CSV)", data=hasil.to_csv(sep=";", index=False).encode("utf-8-sig"),
        file_name="hasil_prediksi.csv", mime="text/csv",
    )


def sidebar(fitur):
    """Penjelasan singkat alat, metrik uji model, dan versi pustaka yang berjalan."""
    with st.sidebar:
        st.header("Tentang alat ini")
        st.markdown(
            "Alat ini memperkirakan peluang siswa Jaya Jaya Institut mengalami dropout, "
            "supaya siswa yang berisiko dapat didekati lebih awal."
        )
        st.markdown(
            "Model dilatih pada siswa yang statusnya sudah final (lulus atau dropout), "
            "memakai %d ciri siswa, sama dengan isian pada formulir." % len(fitur["fitur"])
        )
        m = fitur.get("metrik", {})
        st.subheader("Hasil pada data uji")
        st.markdown("Model: **%s**" % fitur.get("model", "-"))
        st.markdown(
            "Akurasi %s, precision %s, recall %s, F1 %s (kelas dropout)."
            % tuple("%.2f" % m.get(k, 0.0) for k in ("akurasi", "precision", "recall", "f1"))
        )
        with st.expander("Arti tingkat risiko"):
            st.markdown(
                "- Rendah: probabilitas di bawah %.0f %%\n"
                "- Sedang: %.0f %% sampai di bawah %.0f %%\n"
                "- Tinggi: %.0f %% atau lebih"
                % (100 * pr.BATAS_SEDANG, 100 * pr.BATAS_SEDANG, 100 * pr.BATAS_TINGGI, 100 * pr.BATAS_TINGGI)
            )
        st.caption(
            "Python %s | scikit-learn %s | pandas %s | streamlit %s"
            % (platform.python_version(), versi("scikit-learn"), versi("pandas"), versi("streamlit"))
        )


def main():
    """Susun halaman: judul, sidebar, dan dua tab."""
    st.set_page_config(page_title="Risiko Dropout Siswa", layout="wide")
    st.title("Perkiraan Risiko Dropout Siswa")
    st.caption("Jaya Jaya Institut")
    try:
        pipeline, fitur = muat()
    except (FileNotFoundError, ValueError, OSError) as e:
        st.error(
            "Model belum dapat dimuat. %s Pastikan folder model/ berisi "
            "model_dropout.joblib dan fitur.json." % str(e)
        )
        return
    sidebar(fitur)
    tab1, tab2 = st.tabs(["Satu siswa", "Banyak siswa"])
    with tab1:
        tab_satu_siswa(pipeline, fitur)
    with tab2:
        tab_banyak_siswa(pipeline, fitur)


main()
