# projet_ml_retail

Projet de classification du churn client pour un jeu de données retail.

## Structure

- `src/preprocessing.py` : préparation des données, encodage, split train/test.
- `src/train_model.py` : entraînement du modèle et sauvegarde du bundle.
- `src/predict.py` : inférence batch à partir d'un CSV.
- `data/raw/` : données brutes.
- `data/train_test/` : sorties du split train/test.
- `models/` : artefacts entraînés.

## Exécution

Prétraitement :

```bash
python src/preprocessing.py
```

Entraînement :

```bash
python src/train_model.py
```

Prédiction sur un CSV :

```bash
python src/predict.py --input-path data/raw/retail_customers_COMPLETE_CATEGORICAL.csv --output-path reports/predictions.csv
```

Le script de prédiction charge le bundle sauvegardé dans `models/churn_model_bundle.joblib`.

