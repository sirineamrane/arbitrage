# HYPERPARAMETERS TUNING 

import joblib
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# ✅ 1️⃣ Charger les données finales après Feature Engineering
X_train = pd.read_parquet("train_X_final.parquet")
X_test = pd.read_parquet("test_X_final.parquet")
y_train = pd.read_parquet("train_y_final.parquet")["spread"]
y_test = pd.read_parquet("test_y_final.parquet")["spread"]

# ✅ 2️⃣ Charger le modèle sélectionné précédemment
best_model = joblib.load("best_model.pkl")
print(f"\n🚀 Modèle sélectionné pour optimisation : {type(best_model).__name__}")

# ✅ 3️⃣ Définir la fonction d’optimisation avec `Optuna`
def objective(trial):
    model_type = type(best_model).__name__

    if model_type == "XGBRegressor":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10),
        }
        model = xgb.XGBRegressor(**params, random_state=42)
    
    elif model_type == "LGBMRegressor":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 10, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10),
        }
        model = lgb.LGBMRegressor(**params, random_state=42)

    elif model_type == "CatBoostRegressor":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
            "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0.0, 1.0),
        }
        model = cb.CatBoostRegressor(**params, random_state=42, verbose=0)

    elif model_type == "RandomForestRegressor":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = RandomForestRegressor(**params, random_state=42)

    else:
        raise ValueError("Modèle non pris en charge pour l'optimisation.")

    # ✅ 4️⃣ Cross-validation pour évaluer la performance du modèle
    score = np.mean(cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5))
    
    return -score  # On minimise RMSE

# ✅ 5️⃣ Lancer l'optimisation avec `Optuna`
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ✅ 6️⃣ Meilleurs hyperparamètres trouvés
best_params = study.best_params
print("\n🏆 Meilleurs hyperparamètres trouvés :", best_params)

# ✅ 7️⃣ Réentraînement du modèle avec les meilleurs hyperparamètres
if isinstance(best_model, xgb.XGBRegressor):
    optimized_model = xgb.XGBRegressor(**best_params, random_state=42)
elif isinstance(best_model, lgb.LGBMRegressor):
    optimized_model = lgb.LGBMRegressor(**best_params, random_state=42)
elif isinstance(best_model, cb.CatBoostRegressor):
    optimized_model = cb.CatBoostRegressor(**best_params, random_state=42, verbose=0)
elif isinstance(best_model, RandomForestRegressor):
    optimized_model = RandomForestRegressor(**best_params, random_state=42)
else:
    raise ValueError("Modèle non pris en charge après optimisation.")

optimized_model.fit(X_train, y_train)

# ✅ 8️⃣ Sauvegarde du modèle optimisé
joblib.dump(optimized_model, "best_model_optimized.pkl")
print("\n✅ Modèle optimisé enregistré sous 'best_model_optimized.pkl'")
