import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from transformers import BertTokenizer, BertModel
import torch
import altair as alt

# --- Bagian Loading Model dan Data Awal ---
try:
    model = joblib.load('violence_justification_lgbm_model.joblib')
except FileNotFoundError:
    st.error("Error: File model 'violence_justification_lgbm_model.joblib' tidak ditemukan. "
             "Pastikan skrip pelatihan telah dijalankan dan model disimpan.")
    st.stop()

# Muat dataset asli untuk mendapatkan nilai unik untuk dropdown
try:
    df_original = pd.read_csv('Violence Against Women Girls Data-1.csv')
    df_original.dropna(subset=['Value'], inplace=True)
    df_original['Survey Year'] = pd.to_datetime(df_original['Survey Year']).dt.year
except FileNotFoundError:
    st.error("Error: File data asli 'Violence Against Women Girls Data-1.csv' tidak ditemukan. "
             "Pastikan berada di direktori yang sama.")
    st.stop()

# --- Fungsi untuk Simulasi Embedding BERT ---
# Dimensi embedding BERT umum (misalnya, bert-base-uncased)
BERT_EMBEDDING_DIM = 768

# Ini adalah fungsi placeholder untuk mensimulasikan embedding BERT.

@st.cache_data # Cache hasil agar tidak dihitung ulang setiap kali Streamlit me-refresh
def get_bert_embedding_simulated(text, dim=BERT_EMBEDDING_DIM):
    """
    Fungsi placeholder untuk mensimulasikan embedding BERT.
    Dalam implementasi nyata, ini akan memuat model BERT dan menggunakannya.
    Contoh implementasi BERT nyata (membutuhkan 'transformers' dan 'torch'):
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy() # Mengambil [CLS] token embedding
    """
    # Mengembalikan vektor acak sebagai simulasi embedding BERT
    np.random.seed(hash(text) % (2**32 - 1)) # Untuk konsistensi hasil acak per teks
    return np.random.rand(dim)

# --- Mendapatkan Nilai Unik untuk Dropdown ---
countries = sorted(df_original['Country'].dropna().unique().tolist())
genders = sorted(df_original['Gender'].dropna().unique().tolist())
demographic_questions = sorted(df_original['Demographics Question'].dropna().unique().tolist())
demographic_responses = sorted(df_original['Demographics Response'].dropna().unique().tolist())
questions = sorted(df_original['Question'].dropna().unique().tolist()) # Bisa juga menggunakan st.text_input untuk pertanyaan baru
survey_years = sorted(df_original['Survey Year'].dropna().unique().tolist())

# --- Pengaturan Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("Analisis Faktor Demografi Terhadap Pembenaran Kekerasan Berbasis Gender")
st.markdown("Aplikasi ini memprediksi persentase pembenaran kekerasan berdasarkan faktor demografi dan alasan.")
st.markdown("*(Catatan: Embedding BERT disimulasikan dalam aplikasi demo ini karena keterbatasan sumber daya lingkungan.)*")

# --- Antarmuka Prediksi ---
st.header("Prediksi Pembenaran Kekerasan Berbasis Gender")

col1, col2, col3 = st.columns(3)

with col1:
    selected_country = st.selectbox("Pilih Negara:", countries)
    selected_gender = st.selectbox("Pilih Jenis Kelamin:", genders)
    selected_demog_question = st.selectbox("Pilih Pertanyaan Demografi:", demographic_questions)

with col2:
    # Filter respons demografi berdasarkan pertanyaan yang dipilih
    filtered_responses = df_original[df_original['Demographics Question'] == selected_demog_question]['Demographics Response'].dropna().unique().tolist()
    selected_demog_response = st.selectbox("Pilih Respon Demografi:", sorted(filtered_responses) if filtered_responses else demographic_responses)
    selected_survey_year = st.selectbox("Pilih Tahun Survei:", survey_years)

with col3:
    # Menggunakan st.text_input agar user bisa memasukkan pertanyaan baru,
    selected_question = st.selectbox("Pilih Alasan Kekerasan (dari data historis):", questions)
    # Atau untuk input teks bebas: selected_question_text = st.text_input("Masukkan Alasan Kekerasan:")

if st.button("Prediksi"):
    # Buat DataFrame untuk prediksi
    input_data = pd.DataFrame([{
        'Country': selected_country,
        'Gender': selected_gender,
        'Demographics Question': selected_demog_question,
        'Demographics Response': selected_demog_response,
        'Question': selected_question, 
        'Survey Year': selected_survey_year
    }])
    
    # Hasilkan embedding BERT untuk pertanyaan input
    input_data['bert_embeddings'] = input_data['Question'].apply(get_bert_embedding_simulated)

    # Pisahkan embedding BERT menjadi kolom-kolom terpisah
    bert_embedding_input_df = pd.DataFrame(input_data['bert_embeddings'].tolist())
    bert_embedding_input_df.columns = [f'bert_embed_{i}' for i in range(BERT_EMBEDDING_DIM)]

    # Gabungkan embedding BERT dengan input_data dan pastikan urutan kolom sesuai pelatihan
    # Kolom yang digunakan untuk pelatihan: ['Country', 'Gender', 'Demographics Question', 'Demographics Response', 'Survey Year'] + list(bert_embedding_df.columns)
    base_columns_for_prediction = ['Country', 'Gender', 'Demographics Question', 'Demographics Response', 'Survey Year']
    
    # Gabungkan fitur dasar dengan embedding BERT
    final_input_for_prediction = pd.concat([
        input_data[base_columns_for_prediction].reset_index(drop=True),
        bert_embedding_input_df.reset_index(drop=True)
    ], axis=1)

    try:
        predicted_value = model.predict(final_input_for_prediction)
        st.success(f"**Persentase Prediksi Persetujuan: {predicted_value[0]:.2f}%**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.write(f"Detail error: {e}")
        st.write("Pastikan semua pustaka yang diperlukan terinstal dan model dimuat dengan benar.")

st.markdown("---")

# --- Bagian Visualisasi Data ---
st.header("Visualisasi Data Pembenaran Kekerasan")

visualization_choice = st.selectbox(
    "Pilih Jenis Visualisasi:",
    ["Distribusi 'Value'", "Rata-rata 'Value' berdasarkan Gender", "Rata-rata 'Value' berdasarkan Pertanyaan Demografi",
     "Rata-rata 'Value' berdasarkan Negara Teratas"]
)

if visualization_choice == "Distribusi 'Value'":
    st.subheader("Distribusi Persentase Persetujuan ('Value')")
    chart = alt.Chart(df_original).mark_bar().encode(
        alt.X('Value:Q', bin=alt.Bin(maxbins=20), title='Persentase Persetujuan'),
        alt.Y('count()', title='Jumlah Data'),
        tooltip=['count()']
    ).properties(
        title='Distribusi Nilai Persentase Persetujuan'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

elif visualization_choice == "Rata-rata 'Value' berdasarkan Gender":
    st.subheader("Rata-rata Persentase Persetujuan berdasarkan Gender")
    avg_by_gender = df_original.groupby('Gender')['Value'].mean().reset_index()
    chart = alt.Chart(avg_by_gender).mark_bar().encode(
        x=alt.X('Value:Q', title='Rata-rata Persentase Persetujuan'),
        y=alt.Y('Gender:N', sort='-x', title='Jenis Kelamin'),
        tooltip=['Gender', alt.Tooltip('Value', format='.2f')]
    ).properties(
        title='Rata-rata Persetujuan per Gender'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

elif visualization_choice == "Rata-rata 'Value' berdasarkan Pertanyaan Demografi":
    st.subheader("Rata-rata Persentase Persetujuan berdasarkan Pertanyaan Demografi")
    avg_by_demog_q = df_original.groupby('Demographics Question')['Value'].mean().reset_index()
    chart = alt.Chart(avg_by_demog_q).mark_bar().encode(
        x=alt.X('Value:Q', title='Rata-rata Persentase Persetujuan'),
        y=alt.Y('Demographics Question:N', sort='-x', title='Pertanyaan Demografi'),
        tooltip=['Demographics Question', alt.Tooltip('Value', format='.2f')]
    ).properties(
        title='Rata-rata Persetujuan per Pertanyaan Demografi'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

elif visualization_choice == "Rata-rata 'Value' berdasarkan Negara Teratas":
    st.subheader("Rata-rata Persentase Persetujuan berdasarkan Negara (Top 10)")
    avg_by_country = df_original.groupby('Country')['Value'].mean().reset_index()
    top_10_countries = avg_by_country.nlargest(10, 'Value')
    chart = alt.Chart(top_10_countries).mark_bar().encode(
        x=alt.X('Value:Q', title='Rata-rata Persentase Persetujuan'),
        y=alt.Y('Country:N', sort='-x', title='Negara'),
        tooltip=['Country', alt.Tooltip('Value', format='.2f')]
    ).properties(
        title='Top 10 Negara dengan Rata-rata Persetujuan Tertinggi'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)