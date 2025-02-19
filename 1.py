import pandas as pd
import yfinance as yf
import datetime
import os
import time

# ✅ 1️⃣ Définition des actifs
assets = {
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "USD": "DX-Y.NYB",
    "OVX": "^OVX",
    "VIX": "^VIX"
}

# ✅ 2️⃣ Période
start_date = "2010-01-01"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# ✅ 3️⃣ Téléchargement robuste avec retry
attempts = 3
for i in range(attempts):
    try:
        df_all = yf.download(list(assets.values()), start=start_date, end=end_date, group_by="ticker")
        break  # Stop retry loop if success
    except Exception as e:
        print(f"⚠️ Erreur lors du téléchargement, tentative {i+1}/{attempts}...")
        time.sleep(3)
else:
    raise Exception("🚨 Échec du téléchargement après plusieurs tentatives !")

# ✅ 4️⃣ Aplatir les colonnes proprement
if isinstance(df_all.columns, pd.MultiIndex):
    df_all.columns = ['_'.join(col).strip() for col in df_all.columns]
df_all.reset_index(inplace=True)

# ✅ 5️⃣ Nettoyage des noms de colonnes
df_all.columns = (
    df_all.columns.str.lower()
    .str.replace(r"[\^=]", "_", regex=True)
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace("__", "_")
    .str.strip("_")
)

# ✅ 6️⃣ Vérification et correction dynamique du renommage
rename_columns = {col: col.replace("bz_f", "brent").replace("cl_f", "wti").replace("dx_y_nyb", "usdindex") 
                  for col in df_all.columns if "bz_f" in col or "cl_f" in col or "dx_y_nyb" in col}

df_all.rename(columns=rename_columns, inplace=True)

# ✅ 7️⃣ Vérifier et combler les valeurs manquantes
df_all.fillna(method='ffill', inplace=True)  # Remplissage par les valeurs précédentes
df_all.fillna(method='bfill', inplace=True)  # Remplissage par les valeurs suivantes
df_all.fillna(0, inplace=True)  # Dernier recours, remplace les NaN restants par 0

# ✅ 8️⃣ Suppression finale des colonnes `ovx_*` et `vix_*`
cols_to_drop = [col for col in df_all.columns if "ovx" in col or "vix" in col]
df_all.drop(columns=cols_to_drop, errors="ignore", inplace=True)

# ✅ 9️⃣ Vérifier après suppression
print("\n✅ Colonnes après suppression OVX/VIX :")
print(df_all.columns.tolist())

# ✅ 🔥 1️⃣0️⃣ Écriture propre et sécurisée
file_path = "historical_data_cleaned.parquet"

if os.path.exists(file_path):
    df_old = pd.read_parquet(file_path)
    df_all = pd.concat([df_old, df_all]).drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

temp_file = file_path + ".tmp"
df_all.to_parquet(temp_file, engine="pyarrow", compression="snappy")
os.replace(temp_file, file_path)

# ✅ 1️⃣1️⃣ Vérification finale
print("\n✅ Données finales :")
print(df_all.head())
print("\n📊 Shape:", df_all.shape)
print("\n✅ Colonnes finales :", df_all.columns.tolist())

