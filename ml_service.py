import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "house_price_prediction_model.joblib"
)

model = joblib.load(MODEL_PATH)

FEATURES = [
    "area",
    "flooers",
    "bedrooms",
    "bathrooms",
    "bhk",
    "property type",
    "furnished",
    "parking",
    "city",
]


def predict_price(features: dict) -> float:
    input_df = pd.DataFrame([features], columns=FEATURES)
    prediction = model.predict(input_df)
    return float(prediction[0])