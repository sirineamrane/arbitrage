# EVALUATION

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ 1️⃣ Charger les données de test
X_test = pd.read_parquet("test_X_final.parquet")
y_test = pd.read_parquet("test_y_final.parquet")["spread"]

# ✅ 2️⃣ Charger le modèle optimisé
best_model = joblib.load("best_model_optimized.pkl")
print(f"\n🚀 Modèle chargé : {type(best_model).__name__}")

# ✅ 3️⃣ Faire des prédictions sur le `test set`
y_pred = best_model.predict(X_test)

# ✅ 4️⃣ Calcul des métriques de performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

print("\n📊 **Métriques de performance sur le test set** :")
print(f"✅ RMSE  : {rmse:.4f}")
print(f"✅ MAE   : {mae:.4f}")
print(f"✅ R²    : {r2:.4f}")
print(f"✅ MAPE  : {mape:.2f}%")

# ✅ 5️⃣ Visualisation des prédictions vs. les vraies valeurs
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("📊 Prédictions vs. Valeurs réelles")
plt.show()

# ✅ 6️⃣ Analyse des erreurs (résidus)
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel("Erreur (Résidu)")
plt.ylabel("Fréquence")
plt.title("📊 Distribution des erreurs")
plt.show()

# ✅ 7️⃣ Plot des résidus pour vérifier les patterns
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Prédictions")
plt.ylabel("Résidu")
plt.title("📊 Plot des résidus")
plt.show()

print("\n✅ Évaluation complète du modèle terminée ! 🚀")
