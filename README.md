# Early Warning Student Dropout Prediction

Sistem **Early Warning** untuk memprediksi mahasiswa yang berisiko **dropout** menggunakan Machine Learning dengan pendekatan **end-to-end MLOps**.

Project ini mencakup training, experiment tracking, model registry, hingga model serving melalui API.

---

## Deskripsi

Model memprediksi status mahasiswa (**Dropout / Non-Dropout**) berdasarkan data:

- **Demografi**: usia, jenis kelamin, status pernikahan
- **Pendaftaran**: jalur masuk, program studi, urutan pilihan
- **Akademik Semester 1**: SKS diambil, lulus, nilai
- **Keuangan**: status SPP, beasiswa, tunggakan

Sistem ini dapat digunakan sebagai **alat pendukung keputusan** bagi institusi pendidikan untuk melakukan intervensi dini.

---

## Dataset

- **Sumber**: UCI Machine Learning Repository  
- **Jumlah Data**: 4.424 mahasiswa  
- **Jumlah Fitur**: 27 fitur  
- **Target**:  
  - `Dropout`  
  - `Non-Dropout`

---

## Model

- **Algoritma**: Random Forest Classifier  
- **Optimasi**: GridSearchCV  
- **Class Weight**: Balanced  

### Evaluasi Model
- **Accuracy (Test)**: 85%
- **F1-Score**: 0.77

---

## Tech Stack (MLOps)

| Komponen | Tool |
|--------|------|
| Code Versioning | GitHub |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Model Serving | FastAPI |
| Containerization | Docker |
| Image Registry | Docker Hub |

---

## Struktur Project

