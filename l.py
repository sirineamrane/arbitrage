# MONITORING

import joblib
import pandas as pd
import numpy as np
import datetime
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ 1️⃣ Initialiser l’API de Monitoring
app = FastAPI(title="ML Monitoring API", description="API de monitoring pour la détection des dérives", version="1.0")

# ✅ 2️⃣ Charger le modèle & preprocessing
model = joblib.load("best_model_optimized.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("expected_features.pkl")

# ✅ 3️⃣ Stockage des prédictions pour recalibrage
PREDICTION_LOG_FILE = "monitoring_predictions.json"

# ✅ 4️⃣ Définition des classes pour FastAPI
class InputData(BaseModel):
    data: List[Dict[str, float]]

class PredictionLog(BaseModel):
    timestamp: str
    input_data: Dict[str, float]
    prediction: float

# ✅ 5️⃣ Fonction pour stocker les prédictions
def log_prediction(input_data, prediction):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_data": input_data,
        "prediction": prediction
    }
    try:
        with open(PREDICTION_LOG_FILE, "r") as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)
    with open(PREDICTION_LOG_FILE, "w") as file:
        json.dump(logs, file, indent=4)

# ✅ 6️⃣ Fonction de détection de `Data Drift`
def detect_data_drift(new_data):
    try:
        with open("training_data_distribution.json", "r") as file:
            training_distribution = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return "⚠️ Pas de baseline trouvée pour comparer le drift."

    drift_results = {}
    for feature in new_data.columns:
        if feature in training_distribution:
            drift_score = wasserstein_distance(new_data[feature], training_distribution[feature])
            drift_results[feature] = drift_score

    return drift_results

# ✅ 7️⃣ Endpoint pour prédictions + monitoring
@app.post("/predict", response_model=Dict[str, List[float]])
def predict(input_data: InputData):
    try:
        df = pd.DataFrame(input_data.data)

        # Vérifier les features
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Features manquantes: {missing_features}")

        # Appliquer le scaling
        df_scaled = scaler.transform(df)

        # Faire la prédiction
        prediction = model.predict(df_scaled).tolist()

        # Stocker la prédiction
        for i in range(len(df)):
            log_prediction(df.iloc[i].to_dict(), prediction[i])

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 8️⃣ Endpoint pour détecter le drift
@app.get("/detect_drift", response_model=Dict[str, float])
def drift_check():
    try:
        with open(PREDICTION_LOG_FILE, "r") as file:
            logs = json.load(file)

        if not logs:
            raise HTTPException(status_code=400, detail="Aucune donnée enregistrée pour vérifier le drift.")

        # Charger les dernières données enregistrées
        last_data = pd.DataFrame([log["input_data"] for log in logs])
        drift_results = detect_data_drift(last_data)

        return {"drift_scores": drift_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 9️⃣ Endpoint pour vérifier les performances
@app.get("/evaluate_performance")
def evaluate_model():
    try:
        with open(PREDICTION_LOG_FILE, "r") as file:
            logs = json.load(file)

        if not logs:
            raise HTTPException(status_code=400, detail="Aucune donnée enregistrée pour évaluer la performance.")

        df_logs = pd.DataFrame(logs)
        y_pred = df_logs["prediction"].values
        y_true = df_logs["input_data"].apply(lambda x: x["spread"]).values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 🔟 Lancer l'API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
