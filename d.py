# FPREPROCESSING DES DONNEES BRUTS

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import os

# ✅ 1️⃣ Charger les datasets
print("\n📥 Chargement des données...")
X_train = pd.read_parquet("train_X.parquet")
X_test = pd.read_parquet("test_X.parquet")

# ✅ 2️⃣ Suppression des valeurs aberrantes (Outliers) avec IQR
print("\n🛠 Suppression des valeurs aberrantes...")
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    return df.where(mask, np.nan)  # Remplace les outliers par NaN pour un impute intelligent

X_train = remove_outliers(X_train)
X_test = remove_outliers(X_test)

# ✅ 3️⃣ Gestion des valeurs manquantes avec KNNImputer ou IterativeImputer
print("\n🔄 Imputation des valeurs manquantes...")
imputer = KNNImputer(n_neighbors=5)  # Peut être remplacé par IterativeImputer()
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# ✅ 4️⃣ Scaling intelligent
print("\n📊 Scaling des données...")
scaler = RobustScaler()  # Par défaut, on utilise RobustScaler pour gérer les outliers
# Alternative : scaler = StandardScaler()  # Pour les modèles ML classiques
# Alternative : scaler = MinMaxScaler()  # Pour les modèles DL

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ✅ 5️⃣ Sauvegarde des données prétraitées
print("\n💾 Sauvegarde des données prétraitées...")
X_train.to_parquet("train_X_preprocessed.parquet", engine="pyarrow", compression="snappy")
X_test.to_parquet("test_X_preprocessed.parquet", engine="pyarrow", compression="snappy")

print("\n✅ Prétraitement terminé !")

