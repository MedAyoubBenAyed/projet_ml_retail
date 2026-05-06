"""Flask app for churn prediction interface (ChurnGuard)"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# Add src to path for preprocessing import
sys.path.insert(0, str(BASE_DIR / "src"))
import preprocessing as prep

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# ─── Model Loading ─────────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / "models" / "churn_model_bundle.joblib"
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"

bundle = None
metrics = None


def build_single_row_input(data: dict) -> pd.DataFrame:
    """
    Build a deterministic one-row input that matches training-time expected columns.
    Uses the reference_row stored in the model bundle, then overrides UI fields.
    """
    reference_row = (bundle or {}).get("reference_row") or {}
    raw_cols = (bundle or {}).get("raw_feature_columns") or list(reference_row.keys())

    row = {c: reference_row.get(c, None) for c in raw_cols}

    # Override the fields exposed in the UI (if present in training columns)
    overrides = {
        "Recency": float(data.get("recency", 60)),
        "Frequency": float(data.get("frequency", 12)),
        "MonetaryTotal": float(data.get("monetary", 850)),
        "CustomerTenureDays": float(data.get("tenure", 300)),
        "SatisfactionScore": float(data.get("satisfaction", 3)),
        "SupportTicketsCount": float(data.get("tickets", 1)),
    }
    for k, v in overrides.items():
        if k in row:
            row[k] = v

    return pd.DataFrame([row])


def prepare_single_row_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic feature engineering for one-row inference without
    dropping constant columns (which would remove almost everything on a single row).
    """
    df_out = df.copy()
    df_out = prep.parse_registration_date(df_out, "RegistrationDate")
    df_out = prep.featurize_ip(df_out, "LastLoginIP")
    df_out = prep.add_feature_engineering(df_out)
    if "CustomerID" in df_out.columns:
        df_out = df_out.drop(columns=["CustomerID"])
    return df_out

def load_model():
    """Load the model bundle at startup"""
    global bundle, metrics
    try:
        bundle = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        
        # Load metrics
        import json
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        print(f"Metrics loaded from {METRICS_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        bundle = None
        metrics = None

# ─── Prediction Function ───────────────────────────────────────────────────
def predict_churn_real(data):
    """
    Use the real trained model to predict churn.
    
    Args:
        data: dict with 'recency', 'frequency', 'monetary', 'tenure', 'satisfaction', 'tickets'
    
    Returns:
        tuple: (churn_pred, probability, success)
    """
    if bundle is None:
        return None, None, False
    
    try:
        df_input = build_single_row_input(data)

        df_prepared = prepare_single_row_for_inference(df_input)
        
        # Get preprocessor and transform
        preprocessor = bundle.get('preprocessor')
        if preprocessor is None:
            return None, None, False
        
        try:
            transformed = preprocessor.transform(df_prepared)
        except ValueError as e:
            # If preprocessor fails due to missing columns, return graceful error
            print(f"Preprocessor error: {e}")
            return None, None, False
        
        # Rebuild a feature DataFrame with the preprocessor feature names (critical for alignment)
        try:
            feature_names = preprocessor.get_feature_names_out()
            features = pd.DataFrame(transformed, index=df_prepared.index, columns=feature_names)
        except Exception:
            features = pd.DataFrame(transformed, index=df_prepared.index)
        
        # Select the exact kept feature set from training (names preferred; fallback to indices for older bundles)
        kept = bundle.get("kept_feature_names") or bundle.get("final_columns") or []
        if kept:
            if isinstance(kept[0], str):
                expected_cols = [str(col) for col in kept]
                # Strict alignment with training-time features:
                # - same order
                # - drops extra columns
                # - adds missing columns with zeros
                features = features.reindex(columns=expected_cols, fill_value=0.0)
            else:
                try:
                    kept_int = [int(c) if isinstance(c, str) and str(c).isdigit() else int(c) for c in kept]
                    features = features.iloc[:, kept_int]
                except Exception:
                    pass
        
        # Make prediction
        model = bundle.get('model')
        if model is None:
            return None, None, False
        
        # Suppress warnings about feature names
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected_features = features.iloc[[0]]
            churn_pred = model.predict(selected_features)[0]
            proba = model.predict_proba(selected_features)[0, 1]
        
        return int(churn_pred), float(proba), True
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, False

# ─── Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Render main interface"""
    model_info = {
        'algorithm': 'LogisticRegression',
        'features': 93,
        'accuracy': '98.06%',
        'roc_auc': '99.79%',
        'recall': '96.56%',
        'precision': '97.57%',
        'f1': '97.06%',
    }
    if metrics:
        model_info['accuracy'] = f"{metrics.get('accuracy', 0.98)*100:.2f}%"
        model_info['roc_auc'] = f"{metrics.get('roc_auc', 0.998)*100:.2f}%"
        model_info['recall'] = f"{metrics.get('recall', 0.97)*100:.2f}%"
        model_info['precision'] = f"{metrics.get('precision', 0.98)*100:.2f}%"
        model_info['f1'] = f"{metrics.get('f1', 0.97)*100:.2f}%"
    
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Get prediction from model
        churn, proba, success = predict_churn_real(data)
        
        if not success or proba is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Determine risk level and formatting
        if proba >= 0.7:
            risk = "Élevé"
            color = "#e74c3c"
        elif proba >= 0.4:
            risk = "Modéré"
            color = "#f39c12"
        else:
            risk = "Faible"
            color = "#27ae60"
        
        label = "Client à risque" if churn == 1 else "Client fidèle"
        
        return jsonify({
            'churn': churn,
            'probability': round(proba, 4),
            'probability_pct': f"{proba*100:.1f}%",
            'risk_level': risk,
            'risk_color': color,
            'label': label,
            'success': True
        })
    
    except Exception as e:
        print(f"Route error: {e}")
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': bundle is not None,
        'metrics_loaded': metrics is not None
    })

# ─── Error handlers ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading ChurnGuard model...")
    load_model()
    
    if bundle is None:
        print("⚠ Warning: Model not found. Running in demo mode.")
    
    print("Starting Flask app on http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
