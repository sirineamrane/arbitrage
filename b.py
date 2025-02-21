# FEATURE EXTRACTION AND CREATION

import pandas as pd
import numpy as np

# ✅ 1️⃣ Chargement des données nettoyées depuis le fichier A
file_path = "a.parquet"
df = pd.read_parquet(file_path)

# ✅ 2️⃣ Création des nouvelles features
# Spread entre WTI et Brent
df["spread"] = df["brent_close"] - df["wti_close"]

# Ratio WTI/Brent
df["ratio"] = df["wti_close"] / df["brent_close"]

# Différence des volumes
df["volume_diff"] = df["brent_volume"] - df["wti_volume"]

# Volatilité WTI & Brent (rolling std sur 10 jours)
df["vol_wti"] = df["wti_close"].rolling(window=10).std()
df["vol_brent"] = df["brent_close"].rolling(window=10).std()

# Moyennes mobiles
df["wti_sma_5"] = df["wti_close"].rolling(window=5).mean()
df["wti_sma_20"] = df["wti_close"].rolling(window=20).mean()
df["brent_sma_5"] = df["brent_close"].rolling(window=5).mean()
df["brent_sma_20"] = df["brent_close"].rolling(window=20).mean()
df["spread_sma_5"] = df["spread"].rolling(window=5).mean()
df["spread_sma_20"] = df["spread"].rolling(window=20).mean()

# Momentum
df["momentum_wti"] = df["wti_close"].diff(5)
df["momentum_brent"] = df["brent_close"].diff(5)

# RSI (Relative Strength Index)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["rsi_wti"] = calculate_rsi(df["wti_close"], 14)
df["rsi_brent"] = calculate_rsi(df["brent_close"], 14)

# Corrélation USD et pétrole (rolling correlation sur 30 jours)
df["usd_wti_corr"] = df["usdindex_close"].rolling(window=30).corr(df["wti_close"])
df["usd_brent_corr"] = df["usdindex_close"].rolling(window=30).corr(df["brent_close"])

# Différence des variations USD-Pétrole
df["usd_wti_change"] = df["usdindex_close"].pct_change() - df["wti_close"].pct_change()
df["usd_brent_change"] = df["usdindex_close"].pct_change() - df["brent_close"].pct_change()

# Ajout des variables temporelles
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# Tendance globale du marché (différence des prix sur 30 jours)
df["trend_30_wti"] = df["wti_close"].diff(30)
df["trend_30_brent"] = df["brent_close"].diff(30)

# ✅ 3️⃣ Sauvegarde des nouvelles features
feature_file = "b.parquet"
df.to_parquet(feature_file, engine="pyarrow", compression="snappy")

print("✅ Feature Engineering terminé et fichier enregistré !")
print("\n Voici le dataset complet : ")

# Charger le fichier
file_path = "b.parquet"
df = pd.read_parquet(file_path)

# Afficher les 5 premières lignes
print("\n✅ Aperçu du dataset enrichi :")
print(df.head())

# Afficher toutes les colonnes disponibles
print("\n📊 Colonnes du dataset :")
print(df.columns.tolist())

# Vérifier la taille des données
print("\n📊 Shape du dataset :", df.shape)
