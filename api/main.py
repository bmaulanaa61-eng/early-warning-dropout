import joblib
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.config_loader import load_config
from utils.logging import setup_logger


# ======================================================
# APP & LOGGER
# ======================================================
app = FastAPI(title="Student Dropout Prediction")
logger = setup_logger("api")


# ======================================================
# LOAD CONFIG
# ======================================================
config = load_config()
logger.info("Config berhasil dimuat")


# ======================================================
# MLFLOW SETUP
# ======================================================
mlflow.set_tracking_uri(config["mlflow"]["tracking_url"])
REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]


# ======================================================
# LOAD MODEL & FEATURES
# ======================================================
try:
    logger.info("Memuat model dari MLflow Registry...")

    model_uri = "models:/student_dropout@production"
    model = mlflow.sklearn.load_model(model_uri)

    feature_names = joblib.load(config["output"]["features_path"])

    logger.info("Model & fitur berhasil dimuat")

except Exception as e:
    logger.error("Gagal memuat model dari MLflow", exc_info=True)
    raise RuntimeError("Model tidak dapat dimuat")


# ======================================================
# REQUEST SCHEMA
# ======================================================
class PredictionRequest(BaseModel):
    data: dict


# ======================================================
# HEALTH CHECK
# ======================================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": REGISTERED_MODEL_NAME
    }


# ======================================================
# PREDICTION ENDPOINT
# ======================================================
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = request.data

        # Validasi fitur
        missing_features = set(feature_names) - set(input_data.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Fitur tidak lengkap: {missing_features}"
            )

        # Urutan fitur harus sama
        df = pd.DataFrame([input_data])[feature_names]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = {
            "prediction_label": "Dropout" if prediction == 1 else "Non-Dropout",
            "prediction_score": round(float(probability), 2)
        }

        logger.info(f"Prediksi berhasil: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Terjadi error saat prediksi", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
