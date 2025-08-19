# TP — MLflow + Docker + FastAPI (Breast Cancer) — CI/CD après `git push`

## Contenu
- **MLflow** en conteneur (SQLite + artifacts persistants dans `./data/mlflow`)
- **Trainer** (Python) : tests `pytest` + entraînement **RandomForest** sur `sklearn.datasets.load_breast_cancer`, enregistrement dans **Model Registry** sous le nom `CancerClassifier`
- **FastAPI** en conteneur : expose `POST /predict` et charge **la dernière version** du modèle enregistré
- **CI/CD** GitHub Actions : build → run MLflow → run trainer → déploiement FastAPI, **à chaque `git push`** sur `main`

## Arborescence
```
mlflow-docker-fastapi-tp/
├─ docker-compose.yml
├─ .github/
│  └─ workflows/
│     └─ ci-docker.yml
├─ mlflow/
│  └─ Dockerfile
├─ trainer/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ entrypoint.py
│  ├─ src/
│  │  └─ train.py
│  └─ tests/
│     └─ test_imports.py
├─ serve/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ app/
│     └─ main.py
└─ data/
   └─ mlflow/          # créé au 1er run (DB + artefacts)
```

## Démarrage local (Windows PowerShell)
Pré-requis : **Docker Desktop** (Compose v2 activé). À la racine du projet :
```powershell
# 1) Démarrer MLflow
docker compose up -d --build mlflow

# 2) Lancer tests + entraînement (enregistre le modèle dans MLflow) ##
docker compose run --rm trainer

# 3) Démarrer le serveur FastAPI (charge la dernière version du modèle)
docker compose up -d --build serve

# 4) Ouvrir les interfaces
start http://127.0.0.1:5000
start http://127.0.0.1:8000/docs
```

## Appels d'API — Exemple `POST /predict`
Le modèle attend un vecteur **30 features** par échantillon (Breast Cancer). Exemple (valeurs réalistes) :
```json
{
  "instances": [
    [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
  ]
}
```
PowerShell :
```powershell
$body = @{
  instances = @(
    @(17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
  )
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method POST -ContentType 'application/json' -Body $body
```

## CI/CD (GitHub Actions)
- Assure-toi d'avoir un runner **self-hosted** avec Docker Desktop disponible.
- Pousse ce repo sur `main` → le workflow **`.github/workflows/ci-docker.yml`** :
  1. build des images
  2. `up -d --wait` MLflow
  3. `run --rm trainer` (tests + train + register)
  4. `up -d --wait` FastAPI + restart pour charger la dernière version
- URLs :
  - MLflow UI : `http://127.0.0.1:5000`
  - FastAPI docs : `http://127.0.0.1:8000/docs`

## Paramètres
- Modèle enregistré : **`CancerClassifier`**
- Variables d'env `serve` :
  - `MLFLOW_TRACKING_URI=http://mlflow:5000`
  - `MODEL_NAME=CancerClassifier`
  - (optionnel) `MODEL_STAGE=Production`

## Nettoyage
```powershell
docker compose down -v
```
