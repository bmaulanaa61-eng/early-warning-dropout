# Early Warning Student Dropout Prediction

Sistem **Early Warning** untuk memprediksi mahasiswa yang berisiko **dropout** menggunakan Machine Learning dengan pendekatan **end-to-end MLOps**.

Project ini mencakup proses training, experiment tracking, model registry, hingga model serving melalui REST API.

---

## Deskripsi

Model memprediksi status mahasiswa (**Dropout / Non-Dropout**) berdasarkan fitur berikut:

- **Demografi**: usia, jenis kelamin, status pernikahan  
- **Pendaftaran**: jalur masuk, program studi, urutan pilihan  
- **Akademik Semester 1**: jumlah SKS diambil, SKS lulus, nilai  
- **Keuangan**: status pembayaran SPP, beasiswa, tunggakan  

Sistem ini ditujukan sebagai **decision support system** bagi institusi pendidikan untuk melakukan **intervensi dini** terhadap mahasiswa berisiko tinggi.

---

## Dataset

- **Sumber**: UCI Machine Learning Repository  
- **Jumlah Data**: 4.424 mahasiswa  
- **Jumlah Fitur**: 27 fitur  
- **Target Kelas**:
  - `Dropout`
  - `Non-Dropout`

---

## Model

- **Algoritma**: Random Forest Classifier  
- **Optimasi Hyperparameter**: GridSearchCV  
- **Penanganan Imbalance**: Class Weight (Balanced)  

### Evaluasi Model
- **Accuracy (Test Set)**: ± 85%  
- **F1-Score**: ± 0.77  

Model terbaik diregistrasikan ke **MLflow Model Registry** dan digunakan langsung pada API inference.

---

## Tech Stack (MLOps)

| Komponen | Tools |
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

```text
early_warning/
├── api/                     # FastAPI
│   ├── main.py
│   └── schemas.py
├── data/                    # Dataset (DVC)
│   └── data.csv
├── models/                  # Model & feature
│   ├── model_dropout.pkl
│   └── fitur.pkl
├── src/                     # Training pipeline
│   └── train.py
├── utils/                   # Utility modules
│   ├── logging.py
│   └── config_loader.py
├── mlruns/                  # MLflow
├── config.yaml              
├── Dockerfile
├── requirements.txt
└── README.md
