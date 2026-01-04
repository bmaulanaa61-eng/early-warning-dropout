import os
import pandas as pd
import joblib
from fastapi import FastAPI
from api.schemas import StudentData, PredictionResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_dropout.pkl")
FITUR_PATH = os.path.join(BASE_DIR, "models", "fitur.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FITUR_PATH)

app = FastAPI(
    title="Early Warning Student Dropout",
    description="Prediction for Student Dropout",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "status": 200, 
        "message": "Early Warning Student Dropout API"
        }

@app.get("/health")
def health_check():
    return {
        "status": 200, 
        "message": "health"
        }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    input_df = pd.DataFrame([data.model_dump()])

    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": "Dropout" if prediction == 1 else "Non-Dropout",
        "probability_dropout": round(float(probabilities[1]), 4),
        "probability_non_dropout": round(float(probabilities[0]), 4),
    }