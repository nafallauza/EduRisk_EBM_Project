import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "output/models/ebm_model.pkl"
ENCODER_PATH = "output/models/label_encoder.pkl"
FEATURE_PATH = "output/models/feature_columns.pkl"
DATA_PATH = "data/processed/student_data_processed.csv"


st.set_page_config(
    page_title="Dashboard Evaluasi Akademik",
    layout="wide"
)


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURE_PATH)
    data = pd.read_csv(DATA_PATH)
    return model, encoder, features, data


def base_row(data, features):
    row = {}

    for col in features:
        if col in data.columns:
            row[col] = data[col].median()
        else:
            row[col] = 0

    return row


def convert_ip(ip_value):
    return ip_value * 5


def build_input(data, features, values):
    row = base_row(data, features)

    row["Age at enrollment"] = values["age"]
    row["Gender"] = values["gender"]
    row["Debtor"] = values["debtor"]
    row["Tuition fees up to date"] = values["tuition"]
    row["Scholarship holder"] = values["scholarship"]
    row["Admission grade"] = values["admission_grade"]
    row["Curricular units 1st sem (approved)"] = values["approved_1"]
    row["Curricular units 2nd sem (approved)"] = values["approved_2"]
    row["Curricular units 1st sem (grade)"] = convert_ip(values["ip_1"])
    row["Curricular units 2nd sem (grade)"] = convert_ip(values["ip_2"])

    input_df = pd.DataFrame([row])
    return input_df[features]


def predict_model(model, encoder, input_df):
    pred_encoded = model.predict(input_df)
    prediction = encoder.inverse_transform(pred_encoded)[0]

    probabilities = model.predict_proba(input_df)[0]

    probability_df = pd.DataFrame({
        "Kelas": encoder.classes_,
        "Probabilitas": probabilities
    })

    probability_df["Probabilitas (%)"] = (
        probability_df["Probabilitas"] * 100
    ).round(2)

    keluar_probability = probability_df.loc[
        probability_df["Kelas"] == "Keluar",
        "Probabilitas"
    ].values[0]

    return prediction, probability_df, keluar_probability


def analyze_risk(values, keluar_probability):
    score = 0
    factors = []

    if values["ip_1"] < 2.00:
        score += 3
        factors.append("IP Semester 1 berada di bawah 2.00")

    if values["ip_2"] < 2.00:
        score += 3
        factors.append("IP Semester 2 berada di bawah 2.00")

    if values["ip_2"] < values["ip_1"]:
        score += 1
        factors.append("IP Semester 2 menurun dibanding Semester 1")

    if values["approved_1"] <= 8:
        score += 2
        factors.append("Jumlah mata kuliah/SKS lulus Semester 1 rendah")

    if values["approved_2"] <= 8:
        score += 2
        factors.append("Jumlah mata kuliah/SKS lulus Semester 2 rendah")

    if values["tuition"] == 0:
        score += 2
        factors.append("Status pembayaran UKT belum up to date")

    if values["debtor"] == 1:
        score += 2
        factors.append("Mahasiswa memiliki tunggakan")

    if values["scholarship"] == 0:
        score += 1
        factors.append("Mahasiswa tidak menerima beasiswa")

    if keluar_probability >= 0.70:
        score += 3
        factors.append("Probabilitas model terhadap risiko keluar tinggi")
    elif keluar_probability >= 0.40:
        score += 1
        factors.append("Probabilitas model terhadap risiko keluar sedang")

    if score >= 7:
        risk_level = "Risiko Tinggi"
    elif score >= 4:
        risk_level = "Risiko Sedang"
    else:
        risk_level = "Risiko Rendah"

    if risk_level in ["Risiko Tinggi", "Risiko Sedang"] or keluar_probability >= 0.50:
        final_status = "Perlu Pemantauan Akademik"
    else:
        final_status = "Performa Relatif Stabil"

    return final_status, risk_level, factors


def get_recommendation(final_status, risk_level):
    if final_status == "Perlu Pemantauan Akademik" and risk_level == "Risiko Tinggi":
        return (
            "Mahasiswa disarankan masuk daftar prioritas pemantauan. "
            "Tindak lanjut yang dapat dilakukan meliputi konseling akademik, "
            "evaluasi capaian IP, pemantauan administrasi UKT, serta pendampingan belajar."
        )

    if final_status == "Perlu Pemantauan Akademik":
        return (
            "Mahasiswa perlu dimonitor secara berkala, terutama pada capaian IP, "
            "jumlah mata kuliah/SKS yang berhasil diselesaikan, dan kondisi administrasi akademik."
        )

    return (
        "Mahasiswa menunjukkan pola akademik yang relatif stabil. "
        "Pemantauan rutin tetap diperlukan agar performa akademik tetap konsisten."
    )


def make_template():
    return pd.DataFrame({
        "age": [19, 20],
        "gender": [1, 0],
        "debtor": [0, 1],
        "tuition_fees_up_to_date": [1, 0],
        "scholarship_holder": [1, 0],
        "admission_grade": [130.0, 115.0],
        "approved_semester_1": [18, 7],
        "approved_semester_2": [20, 6],
        "ip_semester_1": [3.40, 1.80],
        "ip_semester_2": [3.50, 1.70],
    })


def show_status(final_status, risk_level):
    if risk_level == "Risiko Tinggi":
        st.error(f"{final_status} — {risk_level}")
    elif risk_level == "Risiko Sedang":
        st.warning(f"{final_status} — {risk_level}")
    else:
        st.success(f"{final_status} — {risk_level}")


def show_factors(factors):
    if len(factors) == 0:
        st.write("- Tidak ada indikator risiko akademik yang kuat.")
    else:
        for factor in factors:
            st.write(f"- {factor}")


def get_manual_values():
    left, right = st.columns(2)

    with left:
        code = st.text_input("Kode Anonim", value="MHS-004", disabled=True)
        age = st.number_input("Usia saat masuk", min_value=15, max_value=60, value=19)
        gender_label = st.selectbox("Gender", ["Perempuan", "Laki-laki"])
        admission_grade = st.number_input("Nilai masuk", min_value=0.0, max_value=200.0, value=120.0)
        scholarship_label = st.selectbox("Penerima beasiswa", ["Tidak", "Ya"])

    with right:
        tuition_label = st.selectbox("Status pembayaran UKT", ["Tidak up to date", "Up to date"])
        debtor_label = st.selectbox("Memiliki tunggakan", ["Tidak", "Ya"])
        approved_1 = st.number_input("Jumlah mata kuliah/SKS lulus Semester 1", min_value=0, max_value=30, value=18)
        approved_2 = st.number_input("Jumlah mata kuliah/SKS lulus Semester 2", min_value=0, max_value=30, value=18)
        ip_1 = st.number_input("IP Semester 1", min_value=0.0, max_value=4.0, value=3.00, step=0.01)
        ip_2 = st.number_input("IP Semester 2", min_value=0.0, max_value=4.0, value=3.00, step=0.01)

    values = {
        "age": age,
        "gender": 1 if gender_label == "Laki-laki" else 0,
        "admission_grade": admission_grade,
        "scholarship": 1 if scholarship_label == "Ya" else 0,
        "tuition": 1 if tuition_label == "Up to date" else 0,
        "debtor": 1 if debtor_label == "Ya" else 0,
        "approved_1": approved_1,
        "approved_2": approved_2,
        "ip_1": ip_1,
        "ip_2": ip_2,
    }

    return code, values


def render_overview():
    st.subheader("Ringkasan Studi Kasus")

    st.info(
        "Sistem ini digunakan sebagai alat bantu evaluasi akademik untuk mengidentifikasi "
        "mahasiswa yang perlu dipantau berdasarkan performa semester awal dan hasil prediksi model. "
        "Hasil sistem tidak digunakan sebagai keputusan akhir."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Alur Penggunaan")
        st.write("1. Data mahasiswa dimasukkan melalui form manual atau CSV.")
        st.write("2. Model EBM memprediksi kecenderungan Lulus atau Keluar.")
        st.write("3. Sistem membaca indikator akademik seperti IP, jumlah mata kuliah/SKS lulus, UKT, dan beasiswa.")
        st.write("4. Sistem menampilkan status pemantauan, tingkat risiko, dan rekomendasi.")

    with right:
        st.markdown("#### Indikator Risiko")
        st.write("- IP Semester 1 dan Semester 2")
        st.write("- Jumlah mata kuliah/SKS lulus")
        st.write("- Status pembayaran UKT")
        st.write("- Status tunggakan")
        st.write("- Status beasiswa")
        st.write("- Probabilitas prediksi dari model EBM")


def render_individual(model, encoder, features, data):
    st.subheader("Analisis Individual")

    st.info(
        "Form ini digunakan untuk menganalisis satu mahasiswa. "
        "Kode mahasiswa dibuat anonim agar hasil analisis tidak menampilkan identitas pribadi."
    )

    code, values = get_manual_values()

    if st.button("Analisis Mahasiswa", type="primary", use_container_width=True):
        input_df = build_input(data, features, values)

        model_prediction, probability_df, keluar_probability = predict_model(
            model,
            encoder,
            input_df
        )

        final_status, risk_level, factors = analyze_risk(
            values,
            keluar_probability
        )

        recommendation = get_recommendation(final_status, risk_level)

        st.divider()

        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown("### Hasil Evaluasi")
            show_status(final_status, risk_level)

            st.write("**Kode Anonim:**", code)
            st.write("**Prediksi Model:**", model_prediction)
            st.write("**Probabilitas Risiko Keluar:**", f"{keluar_probability * 100:.2f}%")

            st.markdown("#### Indikator yang Terdeteksi")
            show_factors(factors)

            st.markdown("#### Rekomendasi")
            st.write(recommendation)

            if values["ip_1"] < 2.00 or values["ip_2"] < 2.00:
                st.warning(
                    "Terdapat IP semester di bawah 2.00. Mahasiswa tetap perlu dipantau "
                    "meskipun model menunjukkan kecenderungan lulus."
                )

        with right:
            st.markdown("### Probabilitas Model")
            st.dataframe(
                probability_df[["Kelas", "Probabilitas (%)"]],
                use_container_width=True,
                hide_index=True
            )

            chart_df = probability_df.set_index("Kelas")
            st.bar_chart(chart_df["Probabilitas (%)"])


def render_batch(model, encoder, features, data):
    st.subheader("Analisis Batch melalui CSV")

    st.info(
        "Fitur ini digunakan untuk menganalisis banyak mahasiswa sekaligus dan menghasilkan "
        "daftar prioritas pemantauan akademik."
    )

    template = make_template()

    st.download_button(
        label="Download Template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_edurisk.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is None:
        return

    uploaded_df = pd.read_csv(uploaded_file)

    required_columns = list(template.columns)
    missing_columns = [
        col for col in required_columns
        if col not in uploaded_df.columns
    ]

    if missing_columns:
        st.error("Format CSV belum sesuai.")
        st.write("Kolom yang belum tersedia:")
        st.write(missing_columns)
        return

    results = []

    for index, row_data in uploaded_df.iterrows():
        values = {
            "age": row_data["age"],
            "gender": row_data["gender"],
            "admission_grade": row_data["admission_grade"],
            "scholarship": row_data["scholarship_holder"],
            "tuition": row_data["tuition_fees_up_to_date"],
            "debtor": row_data["debtor"],
            "approved_1": row_data["approved_semester_1"],
            "approved_2": row_data["approved_semester_2"],
            "ip_1": row_data["ip_semester_1"],
            "ip_2": row_data["ip_semester_2"],
        }

        input_df = build_input(data, features, values)

        model_prediction, probability_df, keluar_probability = predict_model(
            model,
            encoder,
            input_df
        )

        final_status, risk_level, factors = analyze_risk(
            values,
            keluar_probability
        )

        results.append({
            "kode_anonim": f"MHS-{index + 1:03d}",
            "prediksi_model": model_prediction,
            "status_pemantauan": final_status,
            "tingkat_risiko": risk_level,
            "probabilitas_keluar": round(keluar_probability * 100, 2),
            "jumlah_indikator": len(factors),
            "indikator_terdeteksi": "; ".join(factors),
            "rekomendasi": get_recommendation(final_status, risk_level)
        })

    result_df = pd.DataFrame(results)

    priority_order = {
        "Risiko Tinggi": 1,
        "Risiko Sedang": 2,
        "Risiko Rendah": 3
    }

    result_df["prioritas"] = result_df["tingkat_risiko"].map(priority_order)

    result_df = result_df.sort_values(
        by=["prioritas", "probabilitas_keluar"],
        ascending=[True, False]
    )

    final_df = result_df.drop(columns=["prioritas"])

    st.success("Analisis CSV berhasil dilakukan.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", len(final_df))
    c2.metric("Risiko Tinggi", (final_df["tingkat_risiko"] == "Risiko Tinggi").sum())
    c3.metric("Perlu Pemantauan", (final_df["status_pemantauan"] == "Perlu Pemantauan Akademik").sum())

    st.markdown("#### Daftar Prioritas Pemantauan")
    st.dataframe(final_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Hasil Analisis",
        data=final_df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_analisis_edurisk.csv",
        mime="text/csv",
        use_container_width=True
    )


model, encoder, features, data = load_assets()

st.title("Dashboard Evaluasi Risiko Studi Mahasiswa")
st.caption(
    "Sistem pendukung evaluasi akademik untuk mengidentifikasi mahasiswa yang perlu dipantau "
    "berdasarkan performa semester awal dan hasil prediksi model EBM."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "EBM")
m2.metric("Akurasi", "90.91%")
m3.metric("Output", "2 Kelas")
m4.metric("Fokus", "Evaluasi Akademik")

st.sidebar.title("Dashboard Akademik")
st.sidebar.caption("Evaluasi Risiko Studi")
st.sidebar.markdown("---")
st.sidebar.write("Model: EBM")
st.sidebar.write("Akurasi: 90.91%")
st.sidebar.write("Output: 2 kelas")
st.sidebar.markdown("---")
st.sidebar.info("Hasil sistem bersifat pendukung evaluasi, bukan keputusan akhir.")

tab_overview, tab_individual, tab_batch = st.tabs([
    "Ringkasan",
    "Analisis Individual",
    "Analisis Batch"
])

with tab_overview:
    render_overview()

with tab_individual:
    render_individual(model, encoder, features, data)

with tab_batch:
    render_batch(model, encoder, features, data)
