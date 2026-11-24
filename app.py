
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="NutriCluster",
    page_icon="🍔",
    layout="wide"
)

# ==========================================
# 2. MEMUAT DATA DAN MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("hasil_clustering_mcdonalds.csv")

@st.cache_resource
def load_model():
    return joblib.load("model_clustering_mcdonalds.pkl")

try:
    df = load_data()
    bundle = load_model()

    model = bundle["model"]
    features = list(bundle["features"])
    nama_segmen = bundle["nama_segmen"]

except Exception as e:
    st.error(f"Gagal memuat data atau model: {e}")
    st.stop()

# Memastikan fitur model tersedia di dataset
fitur_valid = [f for f in features if f in df.columns]

if len(fitur_valid) != len(features):
    st.error("Ada fitur model yang tidak ditemukan dalam dataset.")
    st.write("Fitur model:", features)
    st.stop()

# ==========================================
# 3. FUNGSI NAMA SEGMEN
# ==========================================

def get_nama_segmen(cluster):
    try:
        if isinstance(nama_segmen, dict):
            return nama_segmen.get(
                cluster,
                nama_segmen.get(str(cluster), f"Cluster {cluster}")
            )

        if isinstance(nama_segmen, (list, tuple)):
            return nama_segmen[int(cluster)]

    except (IndexError, ValueError, TypeError):
        pass

    return f"Cluster {cluster}"

# ==========================================
# 4. HEADER APLIKASI
# ==========================================

st.title("🍔 NutriCluster")
st.subheader("Pengelompokan Menu McDonald's Berdasarkan Profil Gizi")

st.write(
    "Aplikasi analisis dan pengelompokan menu McDonald's "
    "menggunakan algoritma K-Means dan Hierarchical Clustering."
)

st.divider()

# ==========================================
# 5. NAVIGASI
# ==========================================

menu = st.sidebar.radio(
    "MENU UTAMA",
    [
        "Dashboard",
        "Data Menu",
        "Prediksi Cluster",
        "Analisis Cluster",
        "Perbandingan Algoritma",
        "Rekomendasi"
    ]
)

# ==========================================
# 6. DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("Dashboard NutriCluster")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Menu", len(df))

    col2.metric(
        "Jumlah Cluster K-Means",
        df["Cluster_KMeans"].nunique()
    )

    col3.metric(
        "Rata-rata Kalori",
        f"{df['Calories'].mean():.1f} kcal"
    )

    st.divider()

    st.subheader("Distribusi Cluster K-Means")

    distribusi = (
        df["Cluster_KMeans"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(distribusi)

    st.subheader("Contoh Data Menu")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ==========================================
# 7. DATA MENU
# ==========================================

elif menu == "Data Menu":

    st.header("Data Menu McDonald's")

    kategori = st.selectbox(
        "Pilih Kategori",
        ["Semua"] + sorted(
            df["Category"].dropna().astype(str).unique().tolist()
        )
    )

    if kategori == "Semua":
        data_tampil = df.copy()
    else:
        data_tampil = df[
            df["Category"].astype(str) == kategori
        ]

    st.write("Jumlah menu:", len(data_tampil))

    st.dataframe(
        data_tampil,
        use_container_width=True
    )

    st.download_button(
        "Download Data Menu",
        data=data_tampil.to_csv(index=False).encode("utf-8"),
        file_name="data_menu_mcdonalds.csv",
        mime="text/csv"
    )

# ==========================================
# 8. PREDIKSI CLUSTER
# ==========================================

elif menu == "Prediksi Cluster":

    st.header("Prediksi Cluster Menu")

    metode = st.radio(
        "Pilih metode prediksi",
        [
            "Menu dari Dataset",
            "Input Nutrisi Manual"
        ]
    )

    if metode == "Menu dari Dataset":

        pilihan = st.selectbox(
            "Pilih Menu McDonald's",
            sorted(df["Item"].dropna().unique().tolist())
        )

        if st.button("Prediksi Cluster", type="primary"):

            data_menu = df[
                df["Item"] == pilihan
            ].iloc[[0]]

            hasil = model.predict(data_menu[features])
            cluster = int(hasil[0])

            st.success("Prediksi berhasil!")

            st.metric(
                "Hasil Cluster",
                f"Cluster {cluster}"
            )

            st.subheader(get_nama_segmen(cluster))

            st.write("Informasi nutrisi menu:")

            st.dataframe(
                data_menu,
                use_container_width=True
            )

    else:

        st.write(
            "Masukkan nilai nutrisi untuk mengetahui "
            "kelompok menu berdasarkan model K-Means."
        )

        nilai = {}

        with st.form("form_prediksi"):

            for fitur in features:

                kolom = pd.to_numeric(
                    df[fitur],
                    errors="coerce"
                )

                nilai[fitur] = st.number_input(
                    fitur,
                    min_value=0.0,
                    value=float(kolom.median()),
                    step=0.1
                )

            submit = st.form_submit_button(
                "Prediksi Cluster",
                type="primary"
            )

        if submit:

            data_baru = pd.DataFrame(
                [nilai],
                columns=features
            )

            hasil = model.predict(data_baru)
            cluster = int(hasil[0])

            st.success("Prediksi berhasil!")

            st.metric(
                "Hasil Cluster",
                f"Cluster {cluster}"
            )

            st.subheader(get_nama_segmen(cluster))

            st.write("Data nutrisi yang dimasukkan:")

            st.dataframe(data_baru)

# ==========================================
# 9. ANALISIS CLUSTER
# ==========================================

elif menu == "Analisis Cluster":

    st.header("Analisis Karakteristik Cluster")

    st.subheader("Profil Cluster K-Means")

    profil = (
        df.groupby("Cluster_KMeans")[features]
        .mean()
        .round(2)
    )

    st.dataframe(
        profil,
        use_container_width=True
    )

    st.subheader("Visualisasi Profil Nutrisi")

    fitur_pilihan = st.selectbox(
        "Pilih kandungan nutrisi",
        features
    )

    grafik = (
        df.groupby("Cluster_KMeans")[fitur_pilihan]
        .mean()
    )

    st.bar_chart(grafik)

    st.subheader("Distribusi Cluster Hierarchical")

    distribusi_h = (
        df["Cluster_Hierarchical"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(distribusi_h)

# ==========================================
# 10. PERBANDINGAN ALGORITMA
# ==========================================

elif menu == "Perbandingan Algoritma":

    st.header("Perbandingan Algoritma Clustering")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("K-Means")
        st.write(
            "Jumlah cluster:",
            df["Cluster_KMeans"].nunique()
        )
        st.bar_chart(
            df["Cluster_KMeans"].value_counts().sort_index()
        )

    with col2:
        st.subheader("Hierarchical")
        st.write(
            "Jumlah cluster:",
            df["Cluster_Hierarchical"].nunique()
        )
        st.bar_chart(
            df["Cluster_Hierarchical"].value_counts().sort_index()
        )

    st.subheader("Hasil Evaluasi Algoritma")

    evaluasi = pd.read_csv("perbandingan_algoritma.csv")

    st.dataframe(
        evaluasi,
        use_container_width=True
    )

# ==========================================
# 11. REKOMENDASI
# ==========================================

elif menu == "Rekomendasi":

    st.header("Rekomendasi Menu McDonald's")

    rekomendasi = pd.read_csv("rekomendasi_cluster.csv")

    st.write(
        "Berikut hasil rekomendasi berdasarkan "
        "profil cluster dari penelitian."
    )

    st.dataframe(
        rekomendasi,
        use_container_width=True
    )

# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "NutriCluster | Proyek Akhir Sains Data"
)

st.caption(
    "Dikembangkan oleh Kelompok Proyek Akhir"
)

