# INFERENCE 

import joblib
import pandas as pd
import numpy as np

# ✅ 1️⃣ Charger le modèle de production
MODEL_PATH = "best_model_production.pkl"
SCALER_PATH = "scaler.pkl"
EXPECTED_FEATURES_PATH = "expected_features.pkl"

print("\n🚀 Chargement du modèle de production...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_features = joblib.load(EXPECTED_FEATURES_PATH)

# ✅ 2️⃣ Fonction de préprocessing des nouvelles données
def preprocess_input(data: pd.DataFrame):
    # Vérifier que toutes les features attendues sont présentes
    missing_features = [feat for feat in expected_features if feat not in data.columns]
    if missing_features:
        raise ValueError(f"❌ Features manquantes : {missing_features}")

    # Appliquer le scaling
    data_scaled = scaler.transform(data)
    return data_scaled

# ✅ 3️⃣ Fonction d'inférence
def make_prediction(new_data: pd.DataFrame):
    try:
        # Préprocessing
        processed_data = preprocess_input(new_data)

        # Prédiction avec le modèle
        prediction = model.predict(processed_data)

        return prediction

    except Exception as e:
        return f"Erreur d'inférence : {str(e)}"

# ✅ 4️⃣ Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de nouvelles données
    sample_data = pd.DataFrame({
        "wti_close": [85.2],
        "brent_close": [88.4],
        "usdindex_close": [102.3],
        "spread": [3.2],
        "vol_wti": [2.5],
        "vol_brent": [2.7],
        "momentum_wti": [1.2],
        "momentum_brent": [1.5]
    })

    print("\n📊 Prédiction en cours...")
    result = make_prediction(sample_data)
    print(f"\n✅ Prédiction obtenue : {result}")
