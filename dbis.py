# CORRELATION, STATIONNARITE 

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb

# ✅ 1️⃣ Charger les données après preprocessing (avant feature selection)
file_path = "train_X_scaled.parquet"  # Fichier après preprocessing mais avant feature engineering
df = pd.read_parquet(file_path)

y_path = "train_y.parquet"
y = pd.read_parquet(y_path)["spread"]  # Variable cible (arbitrage WTI-Brent)

print("\n📊 Shape des données :", df.shape)

# ✅ 2️⃣ Calcul des corrélations linéaires et non linéaires
correlation_results = []
for col in df.columns:
    pearson_corr, _ = pearsonr(df[col], y)
    spearman_corr, _ = spearmanr(df[col], y)
    kendall_corr, _ = kendalltau(df[col], y)
    
    correlation_results.append({
        "Feature": col,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "Kendall": kendall_corr
    })

correlation_df = pd.DataFrame(correlation_results).set_index("Feature")

print("\n📊 Corrélation linéaire avec `spread` (WTI-Brent)")
print(correlation_df)

# 📊 Visualisation Heatmap des corrélations
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_df, annot=True, cmap="coolwarm", center=0)
plt.title("📊 Heatmap des Corrélations avec `spread` (WTI-Brent)")
plt.show()

# ✅ 3️⃣ Mutual Information (capture des dépendances non linéaires)
mi_scores = mutual_info_regression(df, y, random_state=42)
mi_results = pd.DataFrame({"Feature": df.columns, "Mutual Information": mi_scores})
mi_results = mi_results.sort_values(by="Mutual Information", ascending=False)

print("\n📊 Mutual Information avec `spread` (WTI-Brent)")
print(mi_results)

# 📊 Visualisation de MI
plt.figure(figsize=(12, 6))
sns.barplot(x="Mutual Information", y="Feature", data=mi_results, palette="viridis")
plt.title("📊 Importance des Features (Mutual Information)")
plt.show()

# ✅ 4️⃣ Importance des Features avec SHAP
print("\n🚀 Entraînement d'un modèle XGBoost pour calculer l'importance SHAP...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(df, y)

explainer = shap.Explainer(model)
shap_values = explainer(df)

# 📊 Affichage de l'importance des features avec SHAP
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, df, show=False)
plt.title("📊 SHAP - Importance des Features pour `spread` (WTI-Brent)")
plt.show()

# ✅ 5️⃣ Test de stationnarité (ADF - Augmented Dickey-Fuller)
stationarity_results = []
for col in df.columns:
    adf_test = adfuller(df[col])
    p_value = adf_test[1]
    stationarity_results.append({
        "Feature": col,
        "ADF p-value": p_value,
        "Stationary": p_value < 0.05  # Si p < 0.05, on rejette H0 et la série est stationnaire
    })

stationarity_df = pd.DataFrame(stationarity_results)
print("\n📊 Test de Stationnarité ADF")
print(stationarity_df)

# 📊 Visualisation des séries non stationnaires
non_stationary_features = stationarity_df[stationarity_df["Stationary"] == False]["Feature"].tolist()
if non_stationary_features:
    df[non_stationary_features].plot(subplots=True, figsize=(10, len(non_stationary_features) * 2), title="📊 Séries non stationnaires")
    plt.show()

print("\n✅ Analyse avancée terminée ! 🚀")
