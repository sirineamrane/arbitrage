# FEATURE ENGINEERING
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller

# ✅ 1️⃣ Charger les données après sélection des features
X_train = pd.read_parquet("train_X_selected.parquet")
X_test = pd.read_parquet("test_X_selected.parquet")

# Charger la variable cible pour la différenciation
y_train = pd.read_parquet("train_y.parquet")["spread"]
y_test = pd.read_parquet("test_y.parquet")["spread"]

print("\n📊 Shape des données AVANT feature engineering :", X_train.shape)

# ✅ 2️⃣ Détection des séries non stationnaires (ADF Test)
def check_stationarity(series):
    p_value = adfuller(series.dropna())[1]
    return p_value < 0.05  # Si p < 0.05, la série est stationnaire

stationary_cols = [col for col in X_train.columns if check_stationarity(X_train[col])]
non_stationary_cols = list(set(X_train.columns) - set(stationary_cols))

print("\n📊 Features stationnaires :", stationary_cols)
print("\n🚀 Features non stationnaires :", non_stationary_cols)

# ✅ 3️⃣ Transformation des séries non stationnaires
for col in non_stationary_cols:
    X_train[f"{col}_diff"] = X_train[col].diff().fillna(0)
    X_test[f"{col}_diff"] = X_test[col].diff().fillna(0)

    # Transformation logarithmique (si valeurs > 0)
    if (X_train[col] > 0).all():
        X_train[f"{col}_log"] = np.log1p(X_train[col])
        X_test[f"{col}_log"] = np.log1p(X_test[col])

# ✅ 4️⃣ Lag Features (Créer un historique pour capturer la dynamique)
lag_periods = [3, 7, 14, 30]
for lag in lag_periods:
    for col in X_train.columns:
        X_train[f"{col}_lag_{lag}"] = X_train[col].shift(lag).fillna(0)
        X_test[f"{col}_lag_{lag}"] = X_test[col].shift(lag).fillna(0)

# ✅ 5️⃣ Momentum & Volatility Features
for col in ["wti_close", "brent_close"]:
    X_train[f"{col}_momentum"] = X_train[col] - X_train[col].shift(5)
    X_test[f"{col}_momentum"] = X_test[col] - X_test[col].shift(5)

    X_train[f"{col}_volatility"] = X_train[col].rolling(10).std().fillna(0)
    X_test[f"{col}_volatility"] = X_test[col].rolling(10).std().fillna(0)

# ✅ 6️⃣ Fourier Features (Détection des cycles)
def fourier_features(df, col, periods=[10, 20, 50]):
    for period in periods:
        df[f"{col}_sin_{period}"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos_{period}"] = np.cos(2 * np.pi * df[col] / period)
    return df

for col in ["wti_close", "brent_close"]:
    X_train = fourier_features(X_train, col)
    X_test = fourier_features(X_test, col)

# ✅ 7️⃣ Interaction Features (Multiplication des variables clés)
X_train["spread_x_ratio"] = X_train["spread"] * X_train["ratio"]
X_test["spread_x_ratio"] = X_test["spread"] * X_test["ratio"]

X_train["vol_wti_x_vol_brent"] = X_train["vol_wti"] * X_train["vol_brent"]
X_test["vol_wti_x_vol_brent"] = X_test["vol_wti"] * X_test["vol_brent"]

# ✅ 8️⃣ Features temporelles (Mois, Jour de la semaine)
X_train["month"] = X_train.index.month
X_test["month"] = X_test.index.month

X_train["day_of_week"] = X_train.index.dayofweek
X_test["day_of_week"] = X_test.index.dayofweek

# ✅ 9️⃣ Sauvegarde des datasets enrichis
X_train.to_parquet("train_X_engineered.parquet", engine="pyarrow", compression="snappy")
X_test.to_parquet("test_X_engineered.parquet", engine="pyarrow", compression="snappy")

print("\n✅ Feature Engineering terminé ! 🚀")
print("📂 Données enregistrées :")
print("- train_X_engineered.parquet")
print("- test_X_engineered.parquet")
