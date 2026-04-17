# projet_ml_retail

Customer churn classification for retail e-commerce.

## Main phases implemented

1. **Preprocessing** (`src/preprocessing.py`)
   - Median/most-frequent imputation
   - `StandardScaler` + PCA on **numerical features only**
   - `OneHotEncoder` for categorical features
   - Optional SMOTE inside training pipeline (no leakage)

2. **Training & comparison** (`src/train_model.py`)
   - Train/test split with stratification
   - Models: Logistic Regression, Random Forest, KNN (+ optional XGBoost if installed)
   - Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
   - Best model selected by F1-score and saved to `models/`

3. **Prediction and deployment**
   - `src/predict.py`: load saved model + predict one customer
   - `app/app.py`: Flask API with `POST /predict`

## Run

```bash
python -m pip install -r requirements.txt
python src/train_model.py
python app/app.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Recency": 10, "Frequency": 15, "MonetaryTotal": 1000, "RegistrationDate": "2023-01-10", "Gender": "F", "Country": "France"}'
```
