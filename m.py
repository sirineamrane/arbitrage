# RETRAINING 

import joblib
import pandas as pd
import numpy as np
import datetime
import json
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ 1️⃣ Charger les logs de prédiction
PREDICTION_LOG_FILE = "monitoring_predictions.json"

try:
    with open(PREDICTION_LOG_FILE, "r") as file:
        logs = json.load(file)
    df_logs = pd.DataFrame(logs)
except (FileNotFoundError, json.JSONDecodeError):
    print("⚠️ Aucune donnée de monitoring trouvée. Pas de retraining nécessaire.")
    exit()

if df_logs.empty:
    print("⚠️ Pas de nouvelles données pour entraîner un modèle.")
    exit()

# ✅ 2️⃣ Construire un dataset avec les nouvelles données
df_new = pd.DataFrame([entry["input_data"] for entry in logs])
df_new["spread"] = [entry["prediction"] for entry in logs]  # Utiliser les valeurs prédites comme cibles

# ✅ 3️⃣ Vérifier si la qualité du modèle a chuté
RMSE_THRESHOLD = 5  # Seuil critique de RMSE pour retraining

y_true = df_new["spread"].values
y_pred = df_logs["prediction"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\n📊 Évaluation actuelle du modèle :")
print(f"✅ RMSE  : {rmse:.4f}")
print(f"✅ MAE   : {mae:.4f}")
print(f"✅ R²    : {r2:.4f}")

if rmse < RMSE_THRESHOLD:
    print("\n✅ Performance acceptable. Pas de retraining nécessaire.")
    exit()

print("\n🚀 Dégradation détectée ! Lancement du retraining...")

# ✅ 4️⃣ Ajouter les nouvelles données au dataset d'entraînement
X_old = pd.read_parquet("train_X_final.parquet")
y_old = pd.read_parquet("train_y_final.parquet")["spread"]

X = pd.concat([X_old, df_new.drop(columns=["spread"])], axis=0).reset_index(drop=True)
y = np.concatenate([y_old, df_new["spread"].values])

# ✅ 5️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# ✅ 6️⃣ Hyperparameter Tuning avec Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10),
    }
    model = xgb.XGBRegressor(**params, random_state=42)
    score = np.mean(cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5))
    return -score  # Minimiser RMSE

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ✅ 7️⃣ Meilleurs hyperparamètres trouvés
best_params = study.best_params
print("\n🏆 Meilleurs hyperparamètres après tuning :", best_params)

# ✅ 8️⃣ Entraînement du modèle recalibré
retrained_model = xgb.XGBRegressor(**best_params, random_state=42)
retrained_model.fit(X_train, y_train)

# ✅ 9️⃣ Évaluer la performance du modèle recalibré
y_pred_new = retrained_model.predict(X_test)
rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_new))
r2_new = r2_score(y_test, y_pred_new)

print(f"\n📊 Performance du modèle recalibré :")
print(f"✅ RMSE : {rmse_new:.4f}")
print(f"✅ R²   : {r2_new:.4f}")

# ✅ 🔟 Sauvegarde du modèle recalibré
if rmse_new < rmse:  # Vérifier si le modèle recalibré est meilleur
    joblib.dump(retrained_model, "best_model_retrained.pkl")
    print("\n✅ Nouveau modèle entraîné et enregistré sous 'best_model_retrained.pkl'")
else:
    print("\n❌ Le modèle recalibré n'est pas meilleur. On conserve l'ancien modèle.")

print("\n✅ Retraining terminé ! 🚀")
