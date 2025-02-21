# DEPLOYEMENT

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# ✅ 1️⃣ Initialiser FastAPI
app = FastAPI(title="ML Model API", description="API de prédiction pour l'arbitrage WTI-Brent", version="1.0")

# ✅ 2️⃣ Charger le modèle optimisé et le scaler
model = joblib.load("best_model_optimized.pkl")
scaler = joblib.load("scaler.pkl")  # Scaler utilisé pour normaliser les données
expected_features = joblib.load("expected_features.pkl")  # Liste des features utilisées

# ✅ 3️⃣ Définir le format attendu avec `Pydantic`
class InputData(BaseModel):
    data: List[Dict[str, float]]  # Liste d'observations sous forme de dictionnaire

# ✅ 4️⃣ Endpoint pour faire des prédictions
@app.post("/predict", response_model=Dict[str, List[float]])
def predict(input_data: InputData):
    try:
        # 🔹 Convertir l'entrée en DataFrame
        df = pd.DataFrame(input_data.data)

        # 🔹 Vérifier que toutes les features sont présentes
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Features manquantes: {missing_features}")

        # 🔹 Appliquer le scaling
        df_scaled = scaler.transform(df)

        # 🔹 Faire la prédiction
        prediction = model.predict(df_scaled)

        # 🔹 Retourner le résultat
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 5️⃣ Endpoint pour vérifier l'état du serveur
@app.get("/")
def root():
    return {"message": "API de prédiction fonctionnelle. Utilisez /predict pour envoyer des données."}

# ✅ 6️⃣ Lancer l'API si exécuté en local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
