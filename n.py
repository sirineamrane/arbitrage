# MODEL SWITCHING, ROLL BACK AUTOMATIQUE 

import joblib
import os
import shutil
import json
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 1️⃣ Charger les modèles
OPTIMIZED_MODEL = "best_model_optimized.pkl"
RETRAINED_MODEL = "best_model_retrained.pkl"
PRODUCTION_MODEL = "best_model_production.pkl"
MODEL_HISTORY_FILE = "model_switching_log.json"

# ✅ 2️⃣ Vérifier si un modèle recalibré existe
if not os.path.exists(RETRAINED_MODEL):
    print("\n❌ Aucun modèle recalibré trouvé. Pas de switch nécessaire.")
    exit()

# ✅ 3️⃣ Charger les modèles
optimized_model = joblib.load(OPTIMIZED_MODEL)
retrained_model = joblib.load(RETRAINED_MODEL)

# ✅ 4️⃣ Charger les données de test pour comparer les modèles
X_test = joblib.load("X_test_evaluation.pkl")  # Assurez-vous que ce fichier est stocké après le split
y_test = joblib.load("y_test_evaluation.pkl")

# ✅ 5️⃣ Faire des prédictions avec les deux modèles
y_pred_optimized = optimized_model.predict(X_test)
y_pred_retrained = retrained_model.predict(X_test)

# ✅ 6️⃣ Comparer les performances
rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
rmse_retrained = np.sqrt(mean_squared_error(y_test, y_pred_retrained))
r2_optimized = r2_score(y_test, y_pred_optimized)
r2_retrained = r2_score(y_test, y_pred_retrained)

print("\n📊 Comparaison des modèles :")
print(f"✅ RMSE (Optimisé)  : {rmse_optimized:.4f}")
print(f"✅ RMSE (Recalibré) : {rmse_retrained:.4f}")
print(f"✅ R² (Optimisé)    : {r2_optimized:.4f}")
print(f"✅ R² (Recalibré)   : {r2_retrained:.4f}")

# ✅ 7️⃣ Décision de mise en production
if rmse_retrained < rmse_optimized and r2_retrained > r2_optimized:
    print("\n🚀 Nouveau modèle recalibré validé. Déploiement en cours...")
    
    # Sauvegarde du modèle recalibré en tant que modèle de production
    shutil.copy(RETRAINED_MODEL, PRODUCTION_MODEL)

    # Enregistrer l'historique du changement
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "previous_model": OPTIMIZED_MODEL,
        "new_model": RETRAINED_MODEL,
        "rmse_optimized": rmse_optimized,
        "rmse_retrained": rmse_retrained,
        "r2_optimized": r2_optimized,
        "r2_retrained": r2_retrained
    }

    try:
        with open(MODEL_HISTORY_FILE, "r") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(log_entry)

    with open(MODEL_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

    print("\n✅ Modèle recalibré déployé avec succès !")
else:
    print("\n❌ Le modèle recalibré n'est pas meilleur. Aucun changement effectué.")

print("\n✅ Processus de switch terminé ! 🚀")
