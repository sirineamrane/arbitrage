import joblib
import pandas as pd
import numpy as np
import argparse
import time

# ✅ 1️⃣ Charger le modèle de production et le scaler
MODEL_PATH = "best_model_production.pkl"
SCALER_PATH = "scaler.pkl"
EXPECTED_FEATURES_PATH = "expected_features.pkl"

print("\n🚀 Chargement du modèle de production...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_features = joblib.load(EXPECTED_FEATURES_PATH)

# ✅ 2️⃣ Fonction de préprocessing pour le batch processing
def preprocess_batch(data: pd.DataFrame):
    # Vérifier les features
    missing_features = [feat for feat in expected_features if feat not in data.columns]
    if missing_features:
        raise ValueError(f"❌ Features manquantes : {missing_features}")

    # Appliquer le scaling
    data_scaled = scaler.transform(data)
    return data_scaled

# ✅ 3️⃣ Fonction de batch prediction
def batch_predict(input_file: str, output_file: str, file_format: str = "csv", batch_size: int = 10000):
    print(f"\n📂 Chargement du fichier : {input_file}")
    
    # Détecter le format
    if file_format == "csv":
        df = pd.read_csv(input_file)
    elif file_format == "parquet":
        df = pd.read_parquet(input_file)
    else:
        raise ValueError("❌ Format non supporté. Utilisez 'csv' ou 'parquet'.")

    print(f"\n📊 Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Initialiser la liste de prédictions
    predictions = []

    # Diviser en batchs pour éviter la surcharge mémoire
    num_batches = (len(df) // batch_size) + 1
    start_time = time.time()
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))

        if start_idx >= end_idx:
            break

        print(f"\n🚀 Prédiction batch {i+1}/{num_batches} ({start_idx} → {end_idx})")

        batch_data = df.iloc[start_idx:end_idx].copy()
        processed_batch = preprocess_batch(batch_data)
        batch_pred = model.predict(processed_batch)

        predictions.extend(batch_pred.tolist())

    end_time = time.time()
    print(f"\n✅ Batch Prediction terminée en {end_time - start_time:.2f} sec.")

    # ✅ 4️⃣ Sauvegarde des résultats
    df["prediction"] = predictions

    if file_format == "csv":
        df.to_csv(output_file, index=False)
    elif file_format == "parquet":
        df.to_parquet(output_file, index=False)

    print(f"\n📂 Résultats sauvegardés dans : {output_file}")

# ✅ 5️⃣ Exécution en ligne de commande
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Prediction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Chemin du fichier d'entrée")
    parser.add_argument("--output", type=str, required=True, help="Chemin du fichier de sortie")
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv", help="Format du fichier (csv/parquet)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Taille des batchs (défaut: 10,000)")

    args = parser.parse_args()

    batch_predict(args.input, args.output, args.format, args.batch_size)
