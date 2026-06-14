import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
from google import genai
from google.genai.errors import APIError

st.set_page_config(layout="wide", page_title="Prediksi Kualitas Air")

# Load data hal 1
@st.cache_data
def load_data1():
    """Memuat data Panel A dan Panel B."""
    try:
        dfA = pd.read_csv("dataset/dfA.csv")
        dfA = dfA.set_index("id")
        dfB = pd.read_csv("dataset/dfB.csv") 
        dfB = dfB.set_index("id")
        return dfA, dfB
    except FileNotFoundError:
        return None, None

# Distribusi data hal 1
@st.cache_data
def create_distribution_plot(df, title, color_map):
    """Menghitung value count dan membuat distribusinya."""
    distribusi = df["quality_label"].value_counts().reset_index()
    distribusi.columns = ["Kualitas Air", "Jumlah"]
    total_sampel = distribusi["Jumlah"].sum()
    distribusi["Proporsi"] = (distribusi["Jumlah"] / total_sampel) * 100
    distribusi["Label Teks"] = distribusi.apply(lambda row: f"{row["Jumlah"]} ({row["Proporsi"]:.2f}%)",axis=1)
    fig = px.bar(
            distribusi, 
            x="Kualitas Air", 
            y="Jumlah", 
            color="Kualitas Air", 
            title=title,
            labels={"Kualitas Air": "Kategori Kualitas", "Jumlah": "Banyak Sampel"},
            template="streamlit",
            color_discrete_map=color_map,
            text = "Label Teks"
        )
    return fig    

# Load data hal 2
@st.cache_data
def load_data2():
    """Memuat data Akurasi dan Prediksi."""
    try:
        acc_dfA = pd.read_csv("dataset/acc_dfA.csv")
        acc_dfB = pd.read_csv("dataset/acc_dfB.csv")
        acc_dfA = acc_dfA.set_index("Model Name")
        acc_dfA = acc_dfA.drop("Unnamed: 0", axis=1)
        acc_dfB = acc_dfB.set_index("Model Name")
        acc_dfB = acc_dfB.drop("Unnamed: 0", axis=1)

        df_predsA = pd.read_csv("dataset/df_predsA.csv")
        df_predsB = pd.read_csv("dataset/df_predsB.csv")
        return acc_dfA, acc_dfB, df_predsA, df_predsB
    except FileNotFoundError:
        return None, None, None, None

# Matrix hal 2
@st.cache_data
def generate_confusion_matrix_plot(df, y_true_col, y_pred_col, labels):
    """Membuat objek Matplotlib Figure untuk Confusion Matrix."""
    try:
        cm = confusion_matrix(df[y_true_col], df[y_pred_col], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4)) # Ukuran plot yang lebih kecil
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title(f"CM")
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.close(fig)
        return fig
    except Exception as e:
        st.error(f"Gagal membuat plot CM: {e}")
        return None

# Load model ML hal 3    
@st.cache_resource
def load_model(model_path):
    """Memuat model Machine Learning yang tersimpan."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan: {model_path}")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Halaman 1
def render_data_info_page(dfA, dfB):
    st.title("📊 Halaman Data & Info")

    if dfA is None:
        st.error("Data tidak tersedia.")
        return
     
    st.header("Ringkasan Dataset")
    st.write("Aplikasi ini menggunakan data Internet of Things (IoT) yang dikumpulkan dari Water Treatment Plant (WTP) di Jatinangor ITB. Tujuan utama data ini adalah untuk memantau dan mengevaluasi efektivitas proses filtrasi dengan membandingkan kualitas air pada titik masuk dan titik distribusi.")
    st.write("Data pengawasan kualitas air terbagi menjadi dua dataset: Panel A dan Panel B. Dataset Panel A (Pra-Filtrasi) merekam kondisi air baku yang diambil dari Danau sebelum memasuki unit filtrasi. Sementara itu, dataset Panel B (Pasca-Filtrasi) mencatat kondisi air bersih setelah proses filtrasi selesai dan siap dialirkan menuju Asrama.")
    st.markdown("Sistem monitoring kualitas air ini bergantung pada lima fitur utama untuk memastikan proses pengolahan berjalan efektif, yaitu: "
    "`flow1` yang mengukur debit air baku sebelum diproses, " \
    "`turbidity` untuk memantau tingkat kekeruhan air, " \
    "`tds` atau total dissolved solids untuk menilai kandungan zat terlarut, " \
    "`ph` yang menunjukkan tingkat keasaman atau kebasaan air, dan " \
    "khusus pada dataset2 terdapat fitur `flow2` yang mengukur debit air bersih yang siap didistribusikan ke pengguna.")

    # Dataset
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Panel A")
        st.write(f"Baris: {len(dfA)}, Kolom: {len(dfA.columns)}")
        st.dataframe(dfA.head(6))
    with col2:
        st.subheader("Panel B")
        st.write(f"Baris: {len(dfB)}, Kolom: {len(dfB.columns)}")
        st.dataframe(dfB.head(6))
    
    st.markdown("----")

    # Distribusi Kualitas Air
    st.header("Distribusi Kualitas Air")
    st.markdown("Visualisasi ini menunjukkan perbandingan banyak sampel berdasarkan kategori kualitas air (`quality_label`) di Panel A dan Panel B.")
    color_map = {
    "Putih": "#E0E0E0",      
    "Non-putih": "#4169E1", 
    }
    
    col_grafik_1, col_grafik_2 = st.columns(2)
    with col_grafik_1:
        st.subheader("Panel A")
        fig_A = create_distribution_plot(dfA, "Distribusi Kategori Air di Panel A", color_map)
        st.plotly_chart(fig_A, use_container_width=True)
    with col_grafik_2:
        st.subheader("Panel B")
        fig_B = create_distribution_plot(dfB, "Distribusi Kategori Air di Panel B", color_map)
        st.plotly_chart(fig_B, use_container_width=True)
    
    st.write("Dataset A didominasi oleh label 'Non-Putih' (sekitar 80%), sementara Dataset B menunjukkan mayoritas label 'Putih' (sekitar 60%).")

# Halaman 2
def render_analysis_page(acc_dfA, acc_dfB, df_predsA, df_predsB):
    st.title("📈 Halaman Model & Evaluasi")
    st.write("Model yang digunakan adalah XGBoost, Logistic Regression, dan SVM. Berikut ditampilkan metrik evaluasi dari ketiga model tersebut.")

    # Tabel Akurasi
    st.subheader("Tabel Akurasi Model")
    st.write("Akurasi adalah metrik yang menunjukkan proporsi total prediksi yang benar.")
    st.write("Berikut ditampilkan tabel nilai akurasi setiap model.")
    col_data1, col_data2 = st.columns(2)
    config_kolom = {
        "Before Tuning": st.column_config.NumberColumn("Before Tuning", format="%.8f"),
        "After Tuning": st.column_config.NumberColumn("After Tuning",format="%.8f")
    }
    with col_data1:
        st.subheader("Data Panel A")
        if acc_dfA is not None:
            st.dataframe(acc_dfA,use_container_width=True, column_config=config_kolom)
        else:
            st.error("Data nilai akurasi Panel A tidak tersedia.")
    with col_data2:
        st.subheader("Data Panel B")
        if acc_dfB is not None:
            st.dataframe(acc_dfB,use_container_width=True, column_config=config_kolom)
        else:
            st.error("Data nilai akurasi Panel B tidak tersedia.")

    st.write("Akurasi terbaik pada model Panel A dan Panel B adalah XGBoost.") 

    st.markdown("----")
    
    # COnfusion Matrix
    st.subheader("Confusion Matrix")
    st.write("Confusion Matrix adalah tabel yang menggambarkan kinerja model klasifikasi.")
    st.write("Berikut ditampilkan Confusion Matrix dari tiga model terbaik. Model-model ini dipilih berdasarkan nilai akurasi tertinggi dari total enam kandidat (tiga model awal dan versi tuningnya).")
    labels = ["Non-putih", "Putih"]

    st.subheader("Panel A")
    colsA = st.columns(3)
    if df_predsA is not None:
        models_A = [
            ("XGBoost", "y_pred_XGBoost_tuned", colsA[0]),
            ("Logistic Regression", "y_pred_LogisticRegression_tuned", colsA[1]),
            ("SVM", "y_pred_SVM", colsA[2])
        ]

        for name, col_pred, col in models_A:
            with col:
                st.markdown(f"**{name}**")
                fig = generate_confusion_matrix_plot(df_predsA, "y_true", col_pred, labels)
                if fig:
                    st.pyplot(fig)

    st.subheader("Panel B")
    colsB = st.columns(3)
    if df_predsB is not None:
        models_B = [
            ("XGBoost", "y_pred_XGBoost_tuned", colsB[0]),
            ("Logistic Regression", "y_pred_Logistic_Regression", colsB[1]),
            ("SVM", "y_pred_SVM", colsB[2])
        ]
        for name, col_pred, col in models_B:
            with col:
                st.markdown(f"**{name}**")
                fig = generate_confusion_matrix_plot(df_predsB, "y_true", col_pred, labels)
                if fig:
                    st.pyplot(fig)
    st.markdown("Dalam klasifikasi kualitas air, Kasus Positif didefinisikan sebagai air yang terklasifikasi 'Non-Putih'. " \
    "Dalam konteks ini, sel matriks yang harus dimaksimalkan penekanannya adalah False Negative (FN), karena merepresentasikan risiko terbesar. " \
    "False Positive (FN) merepresentasikan situasi di mana air aktualnya Non-Putih (bermasalah/terkontaminasi), tetapi diprediksi oleh model sebagai Putih. " \
    "Tingginya nilai FN berakibat fatal karena menyebabkan 'Safety Failure'. Model gagal mendeteksi kontaminasi dan memberikan jaminan keamanan yang palsu. " \
    "Kesalahan ini berpotensi menyebabkan kerugian besar atau risiko kesehatan publik. " \
    "Dari ketiga model, XGBoost dipilih karena berhasil meminimalkan False Negative (hanya mencatat sekitar 140 kasus FN).")
    st.markdown("----")
    st.markdown("Dari dua metrik evaluasi, model terbaik untuk memprediksi kualitas air, baik pada model data Panel A maupun data panel B, adalah `XGBoost`")

# Fungsi LLM hal 3
@st.cache_resource
def initialize_gemini_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"]) 
    except KeyError:
        st.error("Kunci API Gemini tidak ditemukan. Harap tambahkan 'GEMINI_API_KEY' ke secrets.toml Anda.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat inisialisasi Gemini Client: {e}")
        return None
client = initialize_gemini_client()

def generate_water_quality_analysis(flow1, turbidity, tds, ph, flow2, prediction_label, model_used):
    global client
    if client is None:
        return "ERROR API: Gagal terhubung ke Gemini API. Harap periksa kunci API di secrets.toml."
    
    # Menyiapkan input Flow2 (hanya jika digunakan)
    flow2_input = f"- Flow2: {flow2} m³/s" if flow2 > 0 else ""

    # Prompt Engineering: Instruksi untuk LLM
    prompt = f"""
    Anda adalah seorang ahli pengolahan air. Anda baru saja mendapatkan hasil prediksi kualitas air menggunakan model {model_used}.

    Data Input Sensor:
    - Flow1: {flow1} m³/s
    - Kekeruhan (Turbidity): {turbidity} NTU
    - Total Padatan Terlarut (TDS): {tds} ppm
    - Keasaman (pH): {ph}
    {flow2_input}

    Keterangan data input sensor Flow1 dan Flow2:
    - Flow1 adalah debit air yang ditarik dari danau sebelum memasuki unit filtrasi.
    - Flow2 adalah debit air yang didistribusikan ke asrama setelah proses filtrasi.

    Label Kualitas Air Hasil Prediksi: **{prediction_label}**

    Berdasarkan data dan label tersebut, lakukan dua hal:
    1. Interpretasi kualitas air. Jika labelnya "Putih", jelaskan mengapa itu baik. Jika "Non-Putih", jelaskan potensi masalah utama berdasarkan nilai sensor.
    2. Rekomendasi treatment spesifik.
    3. Kesimpulan singkat dari interpretasi dan rekomendasi yang sudah dijelaskan.

    Format Output HARUS dalam format teks berikut:
    Interpretasi: [Tulis interpretasi Anda di sini. Contoh: "Air berwarna Non-Putih, indikasi sangat keruh dan tidak layak konsumsi langsung."]
    Rekomendasi Treatment:
    - [Langkah rekomendasi 1]
    - [Langkah rekomendasi 2]
    - [Langkah rekomendasi 3, dst]
    Kesimpulan: [Tulis kesimpulan Anda di sini.]
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"ERROR API: Gagal terhubung ke Gemini API. Detail: {e}"
    except Exception as e:
        return f"ERROR UMUM: Terjadi kesalahan. Detail: {e}"

# Halaman 3
def render_prediction_page():
    st.title("🔮 Simulasi Prediksi Kualitas Air")
    try:
        modelA = load_model("dataset/best_model_A.joblib")
        modelB = load_model("dataset/best_model_B.joblib")
        scalerA = load_model("dataset/scaler_panelA.joblib") 
        scalerB = load_model("dataset/scaler_panelB.joblib")
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file model/scaler: {e}. Pastikan file joblib berada di direktori yang benar.")
        return
    
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None
    
    if "analysis_output" not in st.session_state:
        st.session_state["analysis_output"] = None
    
    # INput data
    st.write("Masukkan parameter air untuk diprediksi.")
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            flow1 = st.number_input("Flow1 (m³/s)", min_value=0.0, max_value=100.0, step=0.1, value=0.5)
            turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.1, value=50.0)
            ph = st.slider("pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)
        with col2:
            flow2 = st.number_input("Flow2 (m³/s) - Optional", min_value=0.0, step=0.1, value=0.0)
            tds = st.number_input("TDS (ppm)", min_value=0.0, step=1.0, value=300.0)
        submitted_predict = st.form_submit_button("Prediksi Kategori Air")

    # Predict data
    if submitted_predict:
        st.session_state["analysis_output"] = None
        label_map = {0: "Non-Putih", 1: "Putih"}

        if flow2 > 0:  # pakai Panel B
            model_used = "Panel B"
            X_input = np.array([[flow1, turbidity, tds, ph, flow2]])
            X_scaled = scalerB.transform(X_input)
            y_pred = modelB.predict(X_scaled)
        else:  # pakai Panel A
            model_used = "Panel A"
            X_input = np.array([[flow1, turbidity, tds, ph]])
            X_scaled = scalerA.transform(X_input)
            y_pred = modelA.predict(X_scaled)
        
        prediction_label = label_map[y_pred[0]]
        
        st.session_state["prediction_result"] = {
            "label" : prediction_label,
            "model_used" : model_used,
            "params":{"flow1": flow1,
                      "turbidity": turbidity,
                      "tds":tds,
                      "ph":ph,
                      "flow2":flow2}
        }
        st.success(f"Kategori Warna Air adalah **{prediction_label}**")

    if "request_count" not in st.session_state:
        st.session_state["request_count"] = 0
    
    MAX_REQUESTS = 3

    if st.session_state["prediction_result"]:
        sisa = MAX_REQUESTS - st.session_state["request_count"]

        if sisa <=0:
            st.warning("Batas maksimal analisis AI sudah tercapai. Silakan refresh halaman untuk melakukan analisis baru.")
        else:
            st.info(f"Anda memiliki {sisa} kali kesempatan untuk melakukan analisis AI untuk sesi ini.")
            if st.button("Tampilkan Analisis Kualitas Air dan Rekomendasi AI"):
                st.session_state["request_count"] += 1
                label_saat_ini = st.session_state["prediction_result"]["label"]
                # Tampilan memanggil AI
                with st.spinner(f"Memanggil Gemini API untuk analisis air yang berkategori warna **{label_saat_ini}**..."):
                    params = st.session_state["prediction_result"]["params"]
                    analysis_output = generate_water_quality_analysis(
                        params["flow1"],
                        params["turbidity"],
                        params["tds"],
                        params["ph"],
                        params["flow2"],
                        label_saat_ini, 
                        st.session_state["prediction_result"]["model_used"] 
                    )
                    st.session_state["analysis_output"] = analysis_output
                    st.rerun()

        if st.session_state["analysis_output"]:
            analysis_output = st.session_state["analysis_output"]
            st.subheader("Analisis Kualitas Air & Rekomendasi Treatment")
            label = st.session_state["prediction_result"]["label"]
            st.success(f"Kategori Warna Air adalah **{label}**")
            try:
                parts = analysis_output.split("Rekomendasi Treatment:")
                interpretasi = parts[0].replace("Interpretasi:", "").strip()
                rekomendasi_dan_kesimpulan = parts[1].strip() 
                rekomendasi = rekomendasi_dan_kesimpulan.split("Kesimpulan:")[0].strip()
                kesimpulan = rekomendasi_dan_kesimpulan.split("Kesimpulan:")[1].strip()

                st.markdown("**Interpretasi Kualitas Air**")
                st.info(interpretasi)
                st.markdown("**Rekomendasi Treatment:**")
                st.markdown(rekomendasi)
                st.markdown("**Kesimpulan:**")
                st.info(kesimpulan)

            except IndexError:
                st.warning("Gagal memecah output LLM. Menampilkan output mentah:")
                st.code(analysis_output)

    if st.session_state["analysis_output"]:
        if st.button("Reset Input & Hasil"):
            if 'prediction_result' in st.session_state:
                del st.session_state['prediction_result']
            if 'analysis_output' in st.session_state:
                del st.session_state['analysis_output']
            st.rerun()

# MAIN MENU
# Menu pilihan di sidebar
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Data & Info", "Model & Evaluasi", "Simulasi Prediksi"],
        icons=["house-door-fill", "database-fill", "robot", "activity"],
        menu_icon="water",
        default_index=0
    )

if selected == "Home":
    st.title("Halo! 👋 Selamat Datang di Aplikasi Prediksi Kualitas Air")
    st.write("Silakan pilih menu di sidebar untuk melihat info data, evaluasi model, atau simulasi prediksi.")
    
elif selected == "Data & Info":
    data_a, data_b = load_data1()
    if data_a is None or data_b is None:
        st.error("Data tidak tersedia.")
    else:
        render_data_info_page(data_a, data_b)

elif selected == "Model & Evaluasi":
    data_a, data_b, data_c, data_d = load_data2()
    if data_a is None or data_b is None:
        st.error("Data tidak tersedia.")
    else:
        render_analysis_page(data_a, data_b, data_c, data_d)
    
elif selected == "Simulasi Prediksi":
    render_prediction_page()
