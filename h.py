# MODEL TRAINING

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neural_network import MLPRegressor

# ✅ 1️⃣ Charger les données finales après Feature Engineering
X_train = pd.read_parquet("train_X_final.parquet")
X_test = pd.read_parquet("test_X_final.parquet")
y_train = pd.read_parquet("train_y_final.parquet")["spread"]
y_test = pd.read_parquet("test_y_final.parquet")["spread"]

print("\n📊 Shape des données pour l'entraînement :", X_train.shape)

# ✅ 2️⃣ Définition des modèles individuels
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
    "CatBoost": cb.CatBoostRegressor(n_estimators=200, learning_rate=0.05, depth=6, random_state=42, verbose=0),
    "SVM": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=500, random_state=42)
}

# ✅ 3️⃣ Entraînement et validation des modèles individuels
model_scores = {}
for name, model in models.items():
    print(f"\n🚀 Entraînement du modèle {name}...")
    model.fit(X_train, y_train)
    score = np.mean(cross_val_score(model, X_train, y_train, scoring="r2", cv=5))
    model_scores[name] = score
    print(f"✅ Score R² : {score:.4f}")

# ✅ 4️⃣ Sélection des 3 meilleurs modèles pour le Stacking
top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
stacking_estimators = [(name, models[name]) for name, _ in top_models]

# ✅ 5️⃣ Création du Stacking Model
stacking_model = StackingRegressor(estimators=stacking_estimators, final_estimator=Ridge())

# ✅ 6️⃣ Entraînement du Stacking Model
print("\n🚀 Entraînement du Stacking Model...")
stacking_model.fit(X_train, y_train)

# ✅ 7️⃣ Évaluation des performances
stacking_score = np.mean(cross_val_score(stacking_model, X_train, y_train, scoring="r2", cv=5))
print(f"\n📊 Performance du Stacking Model (R²) : {stacking_score:.4f}")

# ✅ 8️⃣ Sélection du modèle final
best_model_name, best_score = max(model_scores.items(), key=lambda x: x[1])
best_model = models[best_model_name] if best_score > stacking_score else stacking_model

print(f"\n🏆 Meilleur modèle sélectionné : {best_model_name if best_score > stacking_score else 'Stacking Model'}")

# ✅ 9️⃣ Sauvegarde du modèle final
joblib.dump(best_model, "best_model.pkl")
print("\n✅ Modèle final enregistré sous 'best_model.pkl'")
