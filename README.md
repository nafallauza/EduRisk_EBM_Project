# Dashboard Evaluasi Risiko Studi Mahasiswa Menggunakan Explainable Boosting Machine

## 1. Deskripsi Proyek

Proyek ini merupakan implementasi data mining untuk memprediksi risiko keluar studi mahasiswa berdasarkan data akademik awal. Sistem dibangun menggunakan algoritma **Explainable Boosting Machine (EBM)** karena model ini tidak hanya menghasilkan prediksi, tetapi juga mendukung interpretabilitas sehingga hasil prediksi dapat lebih mudah dipahami.

Output sistem tidak digunakan sebagai keputusan akhir terhadap status mahasiswa. Sistem ini berfungsi sebagai alat bantu evaluasi akademik untuk mengidentifikasi mahasiswa yang perlu dipantau berdasarkan performa semester awal.

## 2. Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Membangun model klasifikasi untuk memprediksi status mahasiswa menjadi dua kelas, yaitu **Lulus** dan **Keluar**.
2. Menggunakan algoritma Explainable Boosting Machine untuk memperoleh model yang tetap dapat dijelaskan.
3. Menampilkan hasil evaluasi model melalui accuracy, precision, recall, F1-score, confusion matrix, dan feature importance.
4. Mengimplementasikan model ke dalam dashboard berbasis Streamlit untuk analisis individual dan batch melalui CSV.

## 3. Dataset

Dataset yang digunakan adalah **Predict Students' Dropout and Academic Success** yang tersedia pada UCI Machine Learning Repository dan Kaggle.

Dataset awal memiliki tiga target kelas:

- Graduate
- Dropout
- Enrolled

Pada proyek ini, data difokuskan menjadi dua kelas:

| Target Asli | Target Baru |
|---|---|
| Graduate | Lulus |
| Dropout | Keluar |
| Enrolled | Tidak digunakan |

Kelas **Enrolled** tidak digunakan karena mahasiswa dengan status tersebut masih aktif dan belum memiliki status akhir.

## 4. Struktur Folder

```text
EduRisk_EBM_Project/
├── data/
│   ├── raw/
│   │   └── students_dropout_academic_success.csv
│   └── processed/
│       ├── student_data_processed.csv
│       ├── dataset_info.txt
│       └── target_distribution.csv
│
├── output/
│   ├── models/
│   │   ├── ebm_model.pkl
│   │   ├── label_encoder.pkl
│   │   └── feature_columns.pkl
│   │
│   ├── reports/
│   │   ├── training_metrics.txt
│   │   ├── classification_report_train.txt
│   │   ├── evaluation_report.txt
│   │   └── feature_importance_ebm.csv
│   │
│   └── plots/
│       ├── confusion_matrix_ebm.png
│       └── feature_importance_ebm.png
│
├── src/
│   ├── prepare.py
│   ├── train_ebm.py
│   ├── evaluate_ebm.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

## 5. Tahapan Proyek

### 5.1 Data Preparation

Tahap ini dilakukan pada file:

```bash
src/prepare.py
```

Proses yang dilakukan:

- Membaca dataset mentah dari folder `data/raw`.
- Memilih data dengan status akhir Graduate dan Dropout.
- Mengubah label Graduate menjadi Lulus.
- Mengubah label Dropout menjadi Keluar.
- Menyimpan dataset hasil preprocessing ke folder `data/processed`.

Jalankan perintah berikut:

```bash
python src/prepare.py
```

### 5.2 Training Model

Tahap training dilakukan pada file:

```bash
src/train_ebm.py
```

Model yang digunakan adalah **Explainable Boosting Machine**. Beberapa konfigurasi interaksi fitur diuji, kemudian model terbaik disimpan ke folder `output/models`.

Jalankan perintah berikut:

```bash
python src/train_ebm.py
```

### 5.3 Evaluasi Model

Tahap evaluasi dilakukan pada file:

```bash
src/evaluate_ebm.py
```

Evaluasi menghasilkan:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Feature importance

Jalankan perintah berikut:

```bash
python src/evaluate_ebm.py
```

### 5.4 Implementasi Dashboard

Dashboard dibuat menggunakan Streamlit pada file:

```bash
src/app.py
```

Dashboard memiliki tiga menu utama:

1. **Ringkasan**
2. **Analisis Individual**
3. **Analisis Batch**

Jalankan dashboard dengan perintah:

```bash
python -m streamlit run src/app.py
```

## 6. Hasil Evaluasi Model

Model terbaik diperoleh menggunakan algoritma EBM dengan hasil sebagai berikut:

| Metrik | Nilai |
|---|---:|
| Accuracy | 90.91% |
| Precision | 91.10% |
| Recall | 90.91% |
| F1-score | 90.79% |

Classification report:

| Kelas | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Keluar | 0.94 | 0.82 | 0.88 |
| Lulus | 0.89 | 0.96 | 0.93 |

Hasil tersebut menunjukkan bahwa model memiliki performa yang baik dalam membedakan mahasiswa dengan status Lulus dan Keluar.

## 7. Fitur Dashboard

Dashboard menyediakan beberapa fitur utama:

### 7.1 Ringkasan

Menu ini menjelaskan studi kasus, alur penggunaan sistem, dan indikator risiko yang digunakan.

### 7.2 Analisis Individual

Menu ini digunakan untuk menganalisis satu mahasiswa berdasarkan data akademik awal. Input yang digunakan antara lain:

- Usia saat masuk
- Gender
- Nilai masuk
- Status beasiswa
- Status pembayaran UKT
- Status tunggakan
- Jumlah mata kuliah/SKS lulus Semester 1
- Jumlah mata kuliah/SKS lulus Semester 2
- IP Semester 1
- IP Semester 2

Output yang ditampilkan:

- Prediksi model
- Probabilitas risiko keluar
- Status pemantauan
- Tingkat risiko
- Indikator yang terdeteksi
- Rekomendasi tindak lanjut

### 7.3 Analisis Batch

Menu ini digunakan untuk menganalisis banyak mahasiswa melalui file CSV. Sistem akan menghasilkan daftar prioritas pemantauan akademik yang dapat diunduh.

## 8. Interpretasi Hasil

Hasil prediksi tidak digunakan sebagai keputusan akhir. Apabila sistem menampilkan status **Perlu Pemantauan Akademik**, hal tersebut berarti mahasiswa memiliki indikator yang perlu diperhatikan, bukan berarti mahasiswa pasti keluar dari studi.

Pendekatan ini digunakan agar sistem lebih realistis sebagai alat bantu evaluasi akademik.

## 9. Tools dan Library

Library yang digunakan dalam proyek ini:

- Python
- Pandas
- Scikit-learn
- InterpretML
- Streamlit
- Matplotlib
- Joblib
- MLflow

## 10. Cara Menjalankan Project

Aktifkan virtual environment:

```bash
.venv\Scripts\activate
```

Jalankan preprocessing:

```bash
python src/prepare.py
```

Jalankan training:

```bash
python src/train_ebm.py
```

Jalankan evaluasi:

```bash
python src/evaluate_ebm.py
```

Jalankan dashboard:

```bash
python -m streamlit run src/app.py
```

## 11. Kesimpulan

Proyek ini berhasil membangun model prediksi risiko keluar studi mahasiswa menggunakan Explainable Boosting Machine. Model menghasilkan accuracy sebesar 90.91% pada klasifikasi dua kelas, yaitu Lulus dan Keluar. Selain itu, model diimplementasikan ke dalam dashboard Streamlit untuk mendukung analisis individual dan batch.

Dengan adanya dashboard ini, hasil prediksi dapat digunakan sebagai pendukung evaluasi akademik, terutama untuk membantu mengidentifikasi mahasiswa yang perlu mendapat pemantauan lebih lanjut.

## 12. Referensi Singkat

1. V. Realinho, M. V. Martins, J. Machado, and L. Baptista, “Predict Students’ Dropout and Academic Success,” UCI Machine Learning Repository.
2. Kaggle, “Higher Education Predictors of Student Retention.”
3. H. Nori, S. Jenkins, P. Koch, and R. Caruana, “InterpretML: A Unified Framework for Machine Learning Interpretability.”
4. C. Rudin, “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.”
5. A. Barredo Arrieta et al., “Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI.”
