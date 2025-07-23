# Analisis-Faktor-Demografi-Terhadap-Pembenaran-Kekerasan-Berbasis-Gender
<img width="1135" height="759" alt="image" src="https://github.com/user-attachments/assets/70d77875-1c33-402c-8969-797ade47f6ee" />
<img width="1148" height="822" alt="image" src="https://github.com/user-attachments/assets/3b4373af-535b-4a20-bcc8-298053bd48db" />
Ini adalah proyek data mining dan Natural Language Processing (NLP) yang bertujuan untuk menganalisis dan memprediksi persentase pembenaran kekerasan berbasis gender berdasarkan faktor demografi dan alasan kekerasan yang diberikan.

## 1. Pendahuluan
Proyek ini adalah bagian dari tugas akhir untuk mata kuliah data mining, dengan fokus pada penelitian NLP. Tema yang diangkat adalah "Analisis Faktor Demografi Terhadap Pembenaran Kekerasan Berbasis Gender". Dengan menggunakan dataset yang berisi survei tentang pandangan masyarakat terhadap pembenaran kekerasan, kami membangun model prediktif untuk mengidentifikasi korelasi antara faktor demografi, alasan kekerasan, dan tingkat pembenaran.

## 2. Dataset
Dataset yang digunakan berjudul Violence Against Women Girls Data-1.csv. Dataset ini berisi rincian survei terkait pandangan tentang pembenaran kekerasan berbasis gender.
Rincian Kolom:
a. RecordID: ID unik untuk setiap catatan.
b. Country: Negara tempat survei dilakukan.
c. Gender: Jenis kelamin responden (F/M).
d. Demographics Question: Kategori pertanyaan demografi (contoh: Education, Marital Status, Employment, Age, Residence).
e. Demographics Response: Respon spesifik untuk pertanyaan demografi tersebut (contoh: Never Married, Higher, Secondary, Primary, Widowed, Divorced, Separated, Employed for kind, 15-24, Unemployed, Rural, 25-34, 35-49, No Education, Employe for cash, Urban, Married or Living Together).
f. Question: Alasan spesifik mengapa kekerasan dapat dibenarkan (contoh: "...if she burns the food", "... if she argues with him", "... if she refuses to have sex with him"). Ini adalah kolom teks yang akan dianalisis menggunakan teknik NLP.
g. Survey Year: Tahun survei dilakukan (rentang: 2000-01-01 hingga 2018-01-01).
h. Value: Persentase responden dalam kelompok demografi relevan yang setuju dengan alasan kekerasan yang diberikan (target variabel).

## 3. Tujuan Proyek
Menganalisis hubungan antara faktor demografi (negara, gender, pendidikan, status perkawinan, pekerjaan, usia, tempat tinggal) dan alasan kekerasan terhadap persentase pembenaran kekerasan berbasis gender.
-- Membangun model machine learning yang dapat memprediksi persentase pembenaran kekerasan berdasarkan fitur-fitur yang diberikan.
-- Membuat aplikasi web interaktif menggunakan Streamlit untuk visualisasi data dan demonstrasi model.

## 4. Metodologi
Proyek ini mengikuti alur kerja data science standar, mencakup pemahaman data, pra-pemrosesan, analisis eksploratif, rekayasa fitur, pelatihan model, evaluasi, dan deployment.

### 4.1. Data Understanding (Pemahaman Data)
Memuat dataset dan memeriksa informasi dasar seperti tipe data, jumlah nilai non-null, dan beberapa baris pertama data (df.info(), df.head()).
Mengidentifikasi kolom target (Value) dan fitur (Country, Gender, Demographics Question, Demographics Response, Question, Survey Year).

### 4.2. Data Preprocessing (Pra-pemrosesan Data)
Penanganan Missing Values: Baris dengan nilai NaN pada kolom Value (variabel target) dihapus untuk memastikan integritas data pelatihan.
Konversi Tipe Data: Kolom Survey Year dikonversi dari format tanggal ke tahun integer.
Pra-pemrosesan Teks (Question):Teks diubah menjadi huruf kecil (.lower()) untuk standardisasi.

(Catatan: Pada lingkungan lokal dengan NLTK terinstal, langkah ini akan diperluas untuk mencakup penghapusan tanda baca/angka, tokenisasi, penghapusan stopwords, dan lemmatisasi untuk pra-pemrosesan teks yang lebih mendalam. Namun, karena keterbatasan lingkungan komputasi tertentu, langkah-langkah ini dapat disederhanakan.)

### 4.3. Exploratory Data Analysis (EDA)
EDA dilakukan untuk mendapatkan wawasan tentang distribusi data dan hubungan antar variabel. Output visualisasi disimpan sebagai file gambar.
1. Statistik Deskriptif: Ringkasan statistik untuk kolom numerik (Value, Survey Year).
2. Jumlah Nilai (Value Counts): Untuk kolom kategorikal utama seperti Gender, Demographics Question, Demographics Response, Country, dan Question untuk memahami distribusi kategori.
3. Visualisasi Distribusi Value: Histogram dengan KDE untuk melihat sebaran persentase pembenaran.
4. Visualisasi Rata-rata Value vs. Fitur Kategorikal: Bar plot rata-rata Value berdasarkan Gender. Bar plot rata-rata Value berdasarkan Demographics Question. Bar plot rata-rata Value berdasarkan 10 Country teratas.

### 4.4. Feature Engineering
1. Transformasi Fitur Kategorikal: Kolom Country, Gender, Demographics Question, dan Demographics Response diubah menjadi representasi numerik menggunakan OneHotEncoder. Ini penting karena algoritma machine learning bekerja dengan angka. Opsi handle_unknown='ignore' digunakan untuk menangani kategori yang tidak terlihat selama pelatihan.
2. Transformasi Fitur Teks: Kolom Processed_Question diubah menjadi vektor numerik menggunakan TfidfVectorizer. TF-IDF (Term Frequency-Inverse Document Frequency) adalah teknik yang memberikan bobot pada kata berdasarkan frekuensinya dalam dokumen dan kebalikannya dalam korpus, merefleksikan pentingnya kata tersebut.
3. Fitur Numerik: Kolom Survey Year dilewatkan secara langsung ke model.
4. ColumnTransformer: Digunakan untuk menggabungkan semua transformasi ini dalam satu langkah pra-pemrosesan yang terintegrasi dalam pipeline.

### 4.5. Model Training (Pelatihan Model)
Pemilihan Model: LightGBM Regressor dipilih sebagai algoritma regresi. LightGBM adalah kerangka kerja gradient boosting yang sangat efisien dan berkinerja tinggi, cocok untuk data tabular, serta dikenal karena kecepatan dan akurasinya.
1. Pipeline: Sebuah pipeline scikit-learn dibangun untuk mengintegrasikan langkah pra-pemrosesan (preprocessor) dan model regresi (LGBMRegressor).
2. Pembagian Data: Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan train_test_split untuk memastikan evaluasi model yang tidak bias.
3. Hyperparameter Tuning: Hyperparameter dasar untuk LGBMRegressor diatur (misalnya, random_state=42, n_estimators, learning_rate, max_depth). Optimalisasi lebih lanjut dapat dilakukan menggunakan teknik seperti Grid Search atau Random Search.
4. Pelatihan: Model dilatih menggunakan data pelatihan.

### 4.6. Model Evaluation (Evaluasi Model)
Setelah pelatihan, model dievaluasi menggunakan metrik regresi standar pada set data pengujian:
a. Mean Absolute Error (MAE): Rata-rata absolut dari perbedaan antara prediksi dan nilai sebenarnya.
b. Mean Squared Error (MSE): Rata-rata dari kuadrat perbedaan antara prediksi dan nilai sebenarnya.
c. Root Mean Squared Error (RMSE): Akar kuadrat dari MSE, memberikan metrik dalam unit yang sama dengan variabel target.
d. R-squared (R2): Mengukur proporsi varians dalam variabel dependen yang dapat diprediksi dari variabel independen. Nilai mendekati 1 menunjukkan model yang sangat baik.
Model yang sudah dilatih kemudian disimpan sebagai file .joblib (misalnya, violence_justification_lgbm_model.joblib) untuk digunakan kembali tanpa perlu melatih
