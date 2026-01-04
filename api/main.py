import logging
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from api.schemas import StudentData, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "student_dropout"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    # Prioritas: Production
    model_uri = f"models:/{MODEL_NAME}@production"
    model = mlflow.sklearn.load_model(model_uri)
    logging.info("Model berhasil dimuat dari stage Production")

except Exception as e:
    logging.warning(f"Model Production tidak ditemukan: {e}")
    try:
        # Fallback: versi terbaru
        model_uri = f"models:/{MODEL_NAME}@latest"
        model = mlflow.sklearn.load_model(model_uri)
        logging.info("Model berhasil dimuat dari versi terbaru (latest)")
    except Exception as err:
        logging.error("Gagal memuat model dari MLflow Registry")
        raise RuntimeError("Gagal memuat model MLflow") from err

model_metadata = mlflow.models.get_model_info(model_uri)

if model_metadata.signature is None:
    raise RuntimeError(
        "Signature model tidak ditemukan. Pastikan model disimpan dengan signature."
    )

feature_names = model_metadata.signature.inputs.input_names()

logging.info(f"Daftar fitur model: {feature_names}")

app = FastAPI(
    title="Early Warning Student Dropout API",
    description="Prediksi Dropout Mahasiswa Menggunakan MLflow",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "status": 200,
        "message": "Success"
    }

@app.get("/health")
def health_check():
    return {
        "status": 200,
        "message": "Success"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    try:
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[feature_names]

        # Prediksi
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        hasil = {
            "prediction": "Dropout" if prediction == 1 else "Non-Dropout",
            "probability_dropout": round(float(probabilities[1]), 4),
            "probability_non_dropout": round(float(probabilities[0]), 4),
        }

        logging.info(f"Prediksi berhasil: {hasil}")

        return {
            "status": 200,
            "message": "Success",
            "data": hasil
        }

    except KeyError as e:
        logging.error(f"Kesalahan fitur input: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Fitur input tidak sesuai dengan model: {str(e)}"
        )

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat prediksi: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan pada proses prediksi"
        )
