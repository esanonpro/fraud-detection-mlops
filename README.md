# Détection de fraude bancaire — pipeline MLOps

[![CI](https://github.com/esanonpro/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/esanonpro/fraud-detection-mlops/actions/workflows/ci.yml)

Un modèle de détection de fraude sur transactions bancaires, avec le cycle complet autour : préparation des données, entraînement suivi dans MLflow, porte de validation sur le recall, API FastAPI et image Docker. Chaque push lance l'intégration continue, qui réentraîne le modèle sur un échantillon et refuse un modèle moins bon que la référence.

## Le problème

Les fraudes représentent moins de 0,2 % des transactions du jeu de données ([Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Kaggle). Avec un tel déséquilibre, l'accuracy ne dit rien : un modèle qui ne détecte aucune fraude atteint 99,8 %. Le projet optimise donc le **recall** (part des fraudes détectées), en suivant la précision et l'AUC-ROC à côté.

## Architecture

```
creditcard.csv ─► preprocess ─► build_features ─► train (LogisticRegression, class_weight="balanced")
                  (schéma,       (X / y)            │
                   doublons,                        ├─► MLflow : paramètres, métriques, modèle
                   valeurs                          ├─► models/model.joblib + scaler.joblib
                   manquantes)                      └─► models/candidate_metrics.json
                                                              │
                                     gate : recall candidat ≥ recall baseline ?
                                                              │
                                          FastAPI /predict ◄──┘  (image Docker, port 8080)
```

## Intégration continue

Le workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) s'exécute à chaque push et pull request :

1. lint avec `ruff` ;
2. tests unitaires `pytest` (prétraitement, métriques, porte de validation, API) ;
3. entraînement sur l'échantillon `data/sample/creditcard_sample.csv` (≈ 10 500 transactions) ;
4. **porte de validation** : échec si le recall du modèle candidat est inférieur à celui de `models/baseline_metrics.json` ;
5. tests de l'API avec le modèle fraîchement entraîné ;
6. build de l'image Docker.

## Stack

Python 3.11 · scikit-learn · MLflow · FastAPI · Pydantic · Docker · GitHub Actions · pytest · ruff

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt httpx
export PYTHONPATH=src

# Entraîner (échantillon fourni ; pour le jeu complet, placer creditcard.csv dans data/raw/)
python src/training/pipeline.py --input-path data/sample/creditcard_sample.csv

# Comparer au modèle de référence
python src/evaluation/gate.py --candidate models/candidate_metrics.json --baseline models/baseline_metrics.json

# Tests
pytest -q

# Suivre les expériences
mlflow ui
```

## API

```bash
docker build -t fraud-detection-api .
docker run -p 8080:8080 fraud-detection-api
```

Documentation Swagger sur `http://localhost:8080/docs`. Exemple :

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"Time": 0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
       "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55,
       "V12": -0.61, "V13": 0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
       "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25, "V21": -0.01,
       "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12, "V26": -0.18,
       "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
# {"is_fraud": 0, "probability": ...}
```

L'image écoute sur le port 8080 et peut être déployée telle quelle sur Google Cloud Run (`gcloud run deploy --source .`).

## Structure

```
src/
  data_preprocessing/   validation du schéma, nettoyage
  features/             séparation X / y
  training/             pipeline et entraînement (MLflow)
  evaluation/           métriques et porte de validation
  main.py               API FastAPI
tests/                  tests unitaires et API
models/                 modèle servi, métriques baseline et candidat
data/sample/            échantillon utilisé par la CI
notebooks/01_eda.ipynb  analyse exploratoire
```

## Limites et suites

- Le modèle est une régression logistique pondérée : une base solide et interprétable, pas encore un modèle à arbres (XGBoost, LightGBM) qui ferait sans doute mieux en précision.
- Le seuil de décision est celui par défaut (0,5) ; le régler selon le coût d'un faux positif est l'étape suivante.
- Le modèle versionné dans `models/` est entraîné sur l'échantillon de la CI.
