# FEATURE SELECTION 

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

# ✅ 1️⃣ Charger les données prétraitées
X_train = pd.read_parquet("train_X_scaled.parquet")
X_test = pd.read_parquet("test_X_scaled.parquet")
y_train = pd.read_parquet("train_y.parquet")["spread"]  # Target
y_test = pd.read_parquet("test_y.parquet")["spread"]

print("\n📊 Shape des données AVANT sélection :", X_train.shape)

# ✅ 2️⃣ Suppression des colonnes constantes (même valeur partout)
constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
X_train.drop(columns=constant_cols, inplace=True)
X_test.drop(columns=constant_cols, inplace=True)
print(f"\n🚀 Colonnes supprimées car constantes : {constant_cols}")

# ✅ 3️⃣ Suppression des features multicolinéaires (VIF)
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

vif_threshold = 5  # Supprime les features où VIF > 5 (forte collinéarité)
vif_data = calculate_vif(X_train)
high_vif_cols = vif_data[vif_data["VIF"] > vif_threshold]["feature"].tolist()
X_train.drop(columns=high_vif_cols, inplace=True)
X_test.drop(columns=high_vif_cols, inplace=True)
print(f"\n🚀 Colonnes supprimées car VIF > {vif_threshold} :", high_vif_cols)

# ✅ 4️⃣ Sélection des meilleures features avec SelectKBest (Mutual Information)
k_best = 20  # On garde les 20 meilleures features
selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Convertir en DataFrame avec les noms de colonnes
selected_features = X_train.columns[selector.get_support()]
X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

print("\n📊 Features sélectionnées (SelectKBest) :", selected_features.tolist())

# ✅ 5️⃣ Sélection par régularisation LASSO (L1)
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
lasso_selected = X_train.columns[np.abs(lasso.coef_) > 1e-4]  # Seulement les features avec un poids significatif
X_train_selected = X_train_selected[lasso_selected]
X_test_selected = X_test_selected[lasso_selected]

print("\n📊 Features sélectionnées après LASSO :", lasso_selected.tolist())

# ✅ 6️⃣ Sélection via SHAP (Shapley Values)
explainer = shap.Explainer(lasso, X_train_selected)
shap_values = explainer(X_train_selected)
shap_importances = np.abs(shap_values.values).mean(axis=0)

# Prendre les 15 features les plus importantes selon SHAP
shap_top_features = X_train_selected.columns[np.argsort(-shap_importances)[:15]]
X_train_selected = X_train_selected[shap_top_features]
X_test_selected = X_test_selected[shap_top_features]

print("\n📊 Features finales sélectionnées après SHAP :", shap_top_features.tolist())

# ✅ 7️⃣ Sauvegarde des datasets après sélection des features
X_train_selected.to_parquet("train_X_selected.parquet", engine="pyarrow", compression="snappy")
X_test_selected.to_parquet("test_X_selected.parquet", engine="pyarrow", compression="snappy")

print("\n✅ Sélection des features terminée ! 🚀")
print("📂 Données enregistrées :")
print("- train_X_selected.parquet")
print("- test_X_selected.parquet")
