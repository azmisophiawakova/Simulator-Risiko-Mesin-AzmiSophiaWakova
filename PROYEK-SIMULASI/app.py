from pathlib import Path
import streamlit as st
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Simulator Risiko Sistem",
    page_icon="📊",
    layout="wide",
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background-color:#F4F7FC;
}

.block-container{
    padding-top:2rem;
}

.title{
    font-size:38px;
    font-weight:bold;
    color:#1F4E79;
}

.subtitle{
    color:#555;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.08);
}

.result{
    background:#EEF6FF;
    border-left:6px solid #1F77B4;
    padding:20px;
    border-radius:10px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=80
)

st.sidebar.title("Tentang Aplikasi")

st.sidebar.info(
"""
Aplikasi ini digunakan untuk mensimulasikan
Skor Risiko Kegagalan Sistem menggunakan
Machine Learning.

Fitur:
- Prediksi Risiko
- Monitoring Drift
- Inference menggunakan Joblib
"""
)

st.sidebar.success("Status Model : Aktif")

st.sidebar.markdown("---")

st.sidebar.subheader("📦 Informasi Model")

st.sidebar.write("Versi Model : v1.0")

st.sidebar.write("Framework : Scikit-Learn")

st.sidebar.write("Model : Linear Regression")

st.sidebar.write("Scaler : StandardScaler")

st.sidebar.write("Model File : model_risiko_v1.joblib")

st.sidebar.write("Scaler File : scaler_risiko_v1.joblib")

st.sidebar.write("Deployment : Streamlit Cloud Ready")

st.sidebar.write("Status : Production")

# =====================================================
# MODEL LOADER (Validation Script)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_ml_assets():
    model_path = BASE_DIR / "model_risiko_v1.joblib"
    scaler_path = BASE_DIR / "scaler_risiko_v1.joblib"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

try:
    model, scaler = load_ml_assets()

except Exception as e:
    st.error("Model gagal dimuat.")
    st.exception(e)
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<div class="title">
📊 Simulator Risiko Kegagalan Sistem
</div>
""",
unsafe_allow_html=True)

st.markdown(
"""
<div class="subtitle">
Sistem simulasi berbasis Machine Learning
untuk memprediksi risiko kegagalan sistem produksi.
</div>
""",
unsafe_allow_html=True)

st.write("")

# =====================================================
# PIPELINE MLOps
# =====================================================

st.markdown("## 🔄 Pipeline MLOps Minggu 15")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.success("📦 Exporting")
    st.caption("Model & Scaler Joblib")

with col2:
    st.success("✅ Validation")
    st.caption("Load Model tanpa Training")

with col3:
    st.success("📈 Monitoring")
    st.caption("Drift Detection")

with col4:
    st.success("🧠 MCDM")
    st.caption("Mesin Keputusan")

with col5:
    st.success("⚖ Ethical Audit")
    st.caption("Audit Model")

st.markdown("---")

# =====================================================
# INPUT
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🌡 Suhu Mesin")

    suhu = st.slider(
        "",
        0,
        300,
        85
    )

    st.metric("Nilai", f"{suhu} °C")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚙ Getaran Mesin")

    getaran = st.slider(
        "",
        0,
        100,
        7
    )

    st.metric("Nilai", getaran)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# DRIFT
# =====================================================

if suhu > 120 or suhu < 10:

    st.warning(
        "⚠ Data berada di luar rentang data pelatihan. "
        "Prediksi mungkin kurang akurat."
    )

# =====================================================
# MONITORING MODULE
# =====================================================

X_train = np.array([
    [60,2],
    [70,4],
    [80,6],
    [90,8],
    [100,10]
])

TRAIN_MEAN = np.mean(X_train)

def check_data_drift(
        new_data,
        train_mean,
        threshold=2.0):

    drift = np.abs(
        np.mean(new_data)
        -
        np.mean(train_mean)
    )

    if drift > threshold:
        return True, drift

    return False, drift

# =====================================================
# MCDM DECISION ENGINE
# =====================================================

def mcdm_decision(score):

    if score < 30:
        status = "Risiko Rendah"
        rekomendasi = "Operasi Mesin Aman"

    elif score < 70:
        status = "Risiko Sedang"
        rekomendasi = "Lakukan Pemeriksaan Berkala"

    else:
        status = "Risiko Tinggi"
        rekomendasi = "Segera Hentikan Operasi"

    return status, rekomendasi

def clean_sensitive_data(df):

    cols = [
        "Nama_Operator",
        "NIK_Petugas"
    ]

    return df.drop(
        columns=[c for c in cols if c in df.columns],
        errors="ignore"
    )

# =====================================================
# BUTTON
# =====================================================

st.write("")

status = "-"
rekomendasi = "-"

if st.button("🚀 Jalankan Prediksi", use_container_width=True):

    data = np.array([[suhu, getaran]])

    data_scaled = scaler.transform(data)

    hasil = model.predict(data_scaled)[0]
    
    # Monitoring Drift

    drift_status, drift_value = check_data_drift(
    data,
    TRAIN_MEAN
    )

    if drift_status:

        st.warning(
            f"⚠ WARNING : Data Drift Terdeteksi ({drift_value:.2f})\n\n"
            "Hasil simulasi mungkin kurang akurat."
        )

    else:

        st.success("✅ Data masih sesuai dengan distribusi data pelatihan.")

    # Mesin Keputusan (MCDM)

    status, rekomendasi = mcdm_decision(hasil)

    st.write("")

    st.markdown(
        f"""
        <div class="result">
        <h3>Hasil Prediksi</h3>

        <h1 style="color:#1F77B4;">
        {hasil:.2f}
        </h1>

        </div>
        """,
        unsafe_allow_html=True
    )

    progress = max(0, min(int(hasil),100))

    st.progress(progress)

    st.write("")

    if hasil < 30:

        st.success(
            "🟢 Risiko Rendah\n\n"
            "Kondisi mesin masih aman."
        )

    elif hasil < 70:

        st.warning(
            "🟡 Risiko Sedang\n\n"
            "Perlu dilakukan pemeriksaan berkala."
        )

    else:

        st.error(
            "🔴 Risiko Tinggi\n\n"
            "Disarankan segera melakukan inspeksi mesin."
        )

    st.info(
        """
Interpretasi

Semakin besar skor maka semakin tinggi
potensi kegagalan sistem.
"""
    )

st.subheader("📋 Keputusan Sistem (MCDM)")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Kategori Risiko",
        status
    )

with c2:
    st.metric(
        "Rekomendasi",
        rekomendasi
    )
    
# =====================================================
# ETHICAL AUDIT
# =====================================================

st.markdown("---")

with st.expander("⚖ Ethical Audit"):

    st.markdown("""

### Keterbatasan Model

- Model hanya menggunakan dua variabel:
  - Suhu Mesin
  - Getaran Mesin

- Model dilatih menggunakan data historis sehingga
  akurasi dapat menurun apabila karakteristik data
  berubah secara signifikan (Data Drift).

- Hasil simulasi merupakan prediksi Machine Learning
  dan tidak menggantikan keputusan teknisi.

### Potensi Bias

Model tidak menggunakan:

- Nama Operator
- NIK
- Jenis Kelamin
- Alamat

Sehingga keputusan model tidak dipengaruhi oleh
identitas personal pengguna.

### Keamanan

✔ Model dimuat menggunakan Joblib

✔ Tidak menyimpan API Key atau Password
  secara langsung pada kode.

✔ Jika aplikasi menggunakan database,
  kredensial disimpan melalui Streamlit Secrets.

✔ Training tidak dilakukan ulang saat aplikasi berjalan

✔ Siap untuk deployment ke Streamlit Cloud

""")
    
# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div class="footer">

Praktikum Pemodelan dan Simulasi • MLOps Minggu 15

Dibangun menggunakan Streamlit & Scikit-Learn

Developed by Azmi Sophia Wakova 🚀

</div>
""",
unsafe_allow_html=True)