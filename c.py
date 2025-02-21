# SEPERATION DES DONNES D'ENTRAINEMENT ET DE TEST

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ✅ 1️⃣ Charger les données
file_path = "b"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"🚨 ERREUR : Le fichier {file_path} n'existe pas !")

df = pd.read_parquet(file_path)

# ✅ 2️⃣ Définir la variable cible (`y`) et les features (`X`)
TARGET = "spread"
drop_cols = ["date", TARGET]  # On enlève la date et la target

X = df.drop(columns=drop_cols)  # Features
y = df[TARGET]  # Variable cible

# ✅ 3️⃣ Supprimer les colonnes constantes (même valeur partout)
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X = X.drop(columns=constant_cols)

print(f"\n🚀 Colonnes supprimées car constantes : {constant_cols}")

# ✅ 4️⃣ Séparer en train et test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# ✅ 5️⃣ Vérification
print("\n✅ Séparation terminée !")
print(f"📊 Train set : {X_train.shape[0]} lignes, {X_train.shape[1]} features")
print(f"📊 Test set  : {X_test.shape[0]} lignes, {X_test.shape[1]} features")
print("\n🎯 Features utilisées pour l'entraînement :", X.columns.tolist())

# ✅ 6️⃣ Sauvegarde des datasets
X_train.to_parquet("train_X.parquet", engine="pyarrow", compression="snappy")
X_test.to_parquet("test_X.parquet", engine="pyarrow", compression="snappy")
y_train.to_frame().to_parquet("train_y.parquet", engine="pyarrow", compression="snappy")
y_test.to_frame().to_parquet("test_y.parquet", engine="pyarrow", compression="snappy")

print("\n✅ Jeux de données enregistrés :")
print("- train_X.parquet")
print("- test_X.parquet")
print("- train_y.parquet")
print("- test_y.parquet")
