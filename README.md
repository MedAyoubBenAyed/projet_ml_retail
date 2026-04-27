# Projet ML Retail - Analyse Comportementale et Churn

Projet de machine learning sur un jeu retail e-commerce (52 features) pour:

- analyser la clientele,
- preparer des donnees imparfaites,
- entrainer des modeles de churn,
- exposer une prediction via Flask.

## 1) Installation

### Prerequis

- Python 3.10+
- VS Code

### Environnement virtuel

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Linux/Mac:

```bash
python -m venv venv
source venv/bin/activate
```

### Dependances

```bash
pip install -r requirements.txt
```

Pour regenerer `requirements.txt` apres installation de packages:

```bash
pip freeze > requirements.txt
```

## 2) Structure du projet

```text
projet_ml_retail/
|- data/
|  |- raw/          # donnees brutes
|  |- processed/    # sorties nettoyees/pretraitees
|  \- train_test/   # exports X_train/X_test/y_train/y_test
|- notebooks/       # prototypage et exploration
|- src/
|  |- preprocessing.py
|  |- train_model.py
|  |- predict.py
|  \- utils.py
|- models/          # modeles sauvegardes (.joblib)
|- app/             # application Flask
|- reports/         # metriques et rapports
|- requirements.txt
|- README.md
\- .gitignore
```

## 3) Pipeline de preprocessing

Le preprocessing de `src/preprocessing.py` est leakage-safe et aligne sur le notebook:

1. suppression de colonnes redondantes/inutiles,
2. correction des valeurs aberrantes (`SupportTickets`, `Satisfaction`),
3. parsing de `RegistrationDate` en variables temporelles,
4. feature engineering (`MonetaryPerDay`, `AvgBasketValue`),
5. imputation (mediane numerique, mode categoriel),
6. encodage ordinal pour colonnes ordonnees,
7. `StandardScaler` + PCA sur numeriques,
8. `OneHotEncoder` sur nominales,
9. SMOTE optionnel dans le pipeline d'entrainement.

## 4) Entrainement des modeles

Commande simple:

```bash
python src/train_model.py
```

Commande complete (avec chemins explicites):

```bash
python src/train_model.py \
   --data data/raw/retail_customers_COMPLETE_CATEGORICAL.csv \
   --split-output-dir data/train_test \
   --model-output models/best_churn_pipeline.joblib \
   --report-output reports/model_metrics.json \
   --target Churn \
   --pca-variance 0.95
```

Sorties generees:

- `data/train_test/X_train.csv`
- `data/train_test/X_test.csv`
- `data/train_test/y_train.csv`
- `data/train_test/y_test.csv`
- `models/best_churn_pipeline.joblib`
- `reports/model_metrics.json`

## 5) Prediction locale (script Python)

Exemple d'usage de `src/predict.py` dans un script:

```python
from predict import ChurnPredictor

predictor = ChurnPredictor(model_path="models/best_churn_pipeline.joblib")
payload = {
      "Recency": 85,
      "Frequency": 5,
      "MonetaryTotal": 120.0,
      "Country": "France",
      "Gender": "F",
}
print(predictor.predict(payload))
```

## 6) API Flask

Lancer l'application:

```bash
python app/app.py
```

Endpoint principal:

- `POST /predict`

Exemple `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
   -H "Content-Type: application/json" \
   -d '{"Recency": 10, "Frequency": 15, "MonetaryTotal": 1000, "RegistrationDate": "2023-01-10", "Gender": "F", "Country": "France"}'
```

## 7) Tests

```bash
python -m unittest -v tests/test_ml_pipeline.py
```

## 8) Push GitHub

```bash
git add .
git commit -m "Setup complete retail ML project with leakage-safe pipeline"
git push origin <votre-branche>
```
