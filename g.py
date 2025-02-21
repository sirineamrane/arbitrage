# SECOND TRAIN TEST SPLIT 

import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ 1️⃣ Charger les données après Feature Engineering
X = pd.read_parquet("train_X_engineered.parquet")  # Contient train + test après feature engineering
y = pd.read_parquet("train_y.parquet")["spread"]  # Target inchangée

print("\n📊 Shape des données AVANT resplit :", X.shape)

# ✅ 2️⃣ Séparer train et test (80% - 20%) avec `shuffle=False` pour garder l’ordre temporel
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# ✅ 3️⃣ Vérification finale après le resplit
print("\n✅ Nouvelle séparation train/test terminée !")
print(f"📊 Train set : {X_train.shape[0]} lignes, {X_train.shape[1]} features")
print(f"📊 Test set  : {X_test.shape[0]} lignes, {X_test.shape[1]} features")

# ✅ 4️⃣ Sauvegarde des datasets finaux
X_train.to_parquet("train_X_final.parquet", engine="pyarrow", compression="snappy")
X_test.to_parquet("test_X_final.parquet", engine="pyarrow", compression="snappy")
y_train.to_frame().to_parquet("train_y_final.parquet", engine="pyarrow", compression="snappy")
y_test.to_frame().to_parquet("test_y_final.parquet", engine="pyarrow", compression="snappy")

print("\n✅ Jeux de données finaux enregistrés :")
print("- train_X_final.parquet")
print("- test_X_final.parquet")
print("- train_y_final.parquet")
print("- test_y_final.parquet")
