# AUTOMATISATION DU PIPLEINE

import os
import subprocess
import datetime

# ✅ 1️⃣ Définir les chemins des fichiers de sortie
LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_DIR = "data"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(LOG_DIR, f"pipeline_log_{timestamp}.txt")

# ✅ 2️⃣ Définir la séquence des scripts du pipeline
pipeline_steps = [
    "python a.py",
    "python b.py",
    "python c.py",
    "python d.py",
    "python f.py",
    "python g.py",
    "python h.py",
    "python i.py",
    "python j.py",
    "python k.py",
    "python l.py",
    "python m.py",
    "python o.py",
    "python p.py",

]

# ✅ 3️⃣ Exécution automatique des étapes
with open(log_file, "w") as log:
    print("\n🚀 Lancement du pipeline ML automatisé...\n")
    for step in pipeline_steps:
        print(f"\n🔹 Exécution : {step}")
        log.write(f"\n🔹 Exécution : {step}\n")
        
        process = subprocess.run(step, shell=True, stdout=log, stderr=log)
        
        if process.returncode != 0:
            print(f"❌ Erreur lors de l'exécution de {step}. Voir {log_file}")
            exit()

print("\n✅ Pipeline ML terminé avec succès ! 🚀")
