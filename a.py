import numpy as np
import pandas as pd
import datetime
import os

# ✅ 1️⃣ Définition des paramètres
start_date = datetime.datetime(2008, 1, 1)  
end_date = datetime.datetime.now()
date_range = pd.date_range(start=start_date, end=end_date, freq="B")  # Jours ouvrables

np.random.seed(42)  # Réplicabilité

# ✅ 2️⃣ Fonction avancée de génération des prix avec volatilité adaptative et chocs
def generate_oil_data_advanced(base_price, volatility, trend=0.0005, max_price=200, shock_prob=0.005):
    """ Génère des prix réalistes avec des chocs de marché, une volatilité adaptative et des tendances. """
    prices = [base_price]
    vol = volatility  

    for i in range(1, len(date_range)):
        change = np.random.normal(loc=trend, scale=vol)

        # 📉📈 Simulation de chocs de marché (événements géopolitiques, OPEP, crises)
        if np.random.rand() < shock_prob:
            shock = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15)  # Choc de -5% à +15%
            change += shock
            vol *= 1.5  # Augmente la volatilité après un choc

        # 🔄 Volatilité diminue progressivement après un choc
        vol = max(vol * 0.99, volatility)

        # 🛢️ Appliquer le changement et limiter
        new_price = prices[-1] * np.exp(change)
        new_price = min(new_price, max_price)
        new_price = max(new_price, 10)  

        prices.append(new_price)

    return np.array(prices)

# ✅ 3️⃣ Création des données simulées avec corrélations réalistes
oil_data = {
    "date": date_range,
    "wti_close": generate_oil_data_advanced(base_price=70, volatility=0.02, max_price=200),
    "brent_close": generate_oil_data_advanced(base_price=75, volatility=0.02, max_price=200),
    "usd_close": generate_oil_data_advanced(base_price=90, volatility=0.005, trend=-0.0005, max_price=150),
    "ovx_close": generate_oil_data_advanced(base_price=35, volatility=0.03, max_price=100),
    "vix_close": generate_oil_data_advanced(base_price=20, volatility=0.025, max_price=100),
    "wti_volume": np.random.randint(50000, 150000, size=len(date_range)),
    "brent_volume": np.random.randint(50000, 150000, size=len(date_range)),
}

df_fake = pd.DataFrame(oil_data)

# ✅ 4️⃣ Ajout de corrélations réalistes
# 🔄 Corrélation négative entre le pétrole et le dollar
df_fake["usd_close"] = df_fake["usd_close"] - 0.1 * (df_fake["wti_close"] - df_fake["wti_close"].mean())

# 🔄 Corrélation positive entre OVX (volatilité du pétrole) et VIX (volatilité marché global)
df_fake["ovx_close"] = df_fake["ovx_close"] + 0.3 * (df_fake["vix_close"] - df_fake["vix_close"].mean())

# ✅ 5️⃣ Vérification des valeurs extrêmes
print(f"Max WTI Close: {max(df_fake['wti_close'])}")
print(f"Max Brent Close: {max(df_fake['brent_close'])}")
print(f"Max OVX Close: {max(df_fake['ovx_close'])}")
print(f"Max VIX Close: {max(df_fake['vix_close'])}")

# ✅ 6️⃣ Sauvegarde en CSV
file_path_csv = "a.csv"
df_fake.to_csv(file_path_csv, index=False)

# ✅ 7️⃣ Vérification automatique du fichier CSV
if os.path.exists(file_path_csv):
    print(f"✅ Le fichier {file_path_csv} a bien été créé avec {df_fake.shape[0]} lignes.")
else:
    print(f"❌ Erreur : le fichier {file_path_csv} n'a pas été créé.")

# ✅ 8️⃣ Sauvegarde en Parquet
try:
    import pyarrow
    file_path_parquet = "a.parquet"
    df_fake.to_parquet(file_path_parquet, engine="pyarrow", compression="snappy")

    if os.path.exists(file_path_parquet):
        print(f"✅ Le fichier {file_path_parquet} a bien été créé avec {df_fake.shape[0]} lignes et {df_fake.shape[1]} colonnes.")
    else:
        print(f"❌ Erreur : le fichier {file_path_parquet} n'a pas été créé.")

except ImportError:
    print("⚠️ pyarrow n'est pas installé. Installe-le avec : pip install pyarrow")

