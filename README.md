# Early Warning Student Dropout Prediction

Sistem prediksi mahasiswa berisiko dropout menggunakan Machine Learning dengan pendekatan MLOps.

## Deskripsi

Model ini memprediksi apakah mahasiswa akan dropout atau tidak berdasarkan data:
- Demografi (usia, jenis kelamin, status pernikahan)
- Pendaftaran (jalur masuk, program studi)
- Akademik Semester 1 (SKS lulus, nilai)
- Keuangan (SPP, beasiswa)

## Dataset

- **Sumber:** UCI Machine Learning Repository
- **Jumlah Data:** 4.424 mahasiswa
- **Jumlah Fitur:** 27 fitur
- **Target:** Dropout / Non-Dropout

## Model

- **Algoritma:** Random Forest Classifier
- **Akurasi:** 85.20%
- **F1-Score:** 0.77

## Tech Stack (MLOps)

| Komponen | Tool |
|----------|------|
| Code Versioning | GitHub |
| Data Versioning | DVC |
| Model Versioning | MLflow |
| Image Versioning | Docker Hub |
| Model Serving | FastAPI |
| Containerization | Docker |

## Struktur Project
```
early_warning/
├── api/                  # FastAPI
│   ├── main.py
│   └── schemas.py
├── data/                 # Dataset (DVC)
│   └── data.csv
├── models/               # Model (DVC)
│   ├── model_dropout.pkl
│   └── fitur.pkl
├── src/                  # Training script
│   └── train.py
├── mlruns/               # MLflow tracking
├── config.yaml           # Konfigurasi
├── Dockerfile            # Docker config
├── requirements.txt      # Dependencies
└── README.md
```

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/bmaulanaa61-eng/early-warning-dropout.git
cd early-warning-dropout
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Training
```bash
python src/train.py
```

### 4. Jalankan API
```bash
uvicorn api.main:app --reload
```

### 5. Jalankan dengan Docker
```bash
docker pull mllanaa/early-warning-student:v1
docker run -d -p 8000:8000 mllanaa/early-warning-student:v1
```

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Status API |
| GET | `/health` | Health check |
| POST | `/predict` | Prediksi dropout |

### Contoh Request
```json
POST /predict
{
    "Status_Pernikahan": 1,
    "Kewarganegaraan": 1,
    "Jenis_Kelamin": 1,
    "Usia_Saat_Daftar": 20,
    "Jalur_Pendaftaran": 1,
    "Urutan_Pilihan": 1,
    "Program_Studi": 1,
    "Kelas_Siang_Malam": 1,
    "Pendidikan_Sebelumnya": 1,
    "Nilai_Pendidikan_Sebelumnya": 120.0,
    "Nilai_Masuk": 130.0,
    "Pendidikan_Ibu": 1,
    "Pendidikan_Ayah": 1,
    "Pekerjaan_Ibu": 1,
    "Pekerjaan_Ayah": 1,
    "Pindahan": 0,
    "Berkebutuhan_Khusus": 0,
    "Punya_Tunggakan": 0,
    "SPP_Lunas": 1,
    "Penerima_Beasiswa": 0,
    "Mahasiswa_Internasional": 0,
    "SKS_Sem1_Diakui": 0,
    "SKS_Sem1_Diambil": 6,
    "SKS_Sem1_Evaluasi": 6,
    "SKS_Sem1_Lulus": 6,
    "Nilai_Sem1": 14.0,
    "SKS_Sem1_Tanpa_Evaluasi": 0
}
```

### Contoh Response
```json
{
    "prediction": "Non-Dropout",
    "probability_dropout": 0.16,
    "probability_non_dropout": 0.83
}
```

## Docker Hub
```bash
docker pull mllanaa/early-warning-student:v1
```

## Author

**Maulana**

## License

MIT License