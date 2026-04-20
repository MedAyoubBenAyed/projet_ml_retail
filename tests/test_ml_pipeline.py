import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from predict import ChurnPredictor
from preprocessing import create_training_pipeline, get_fitted_pca_component_count
from train_model import train_and_select_best_model


class TestChurnPipeline(unittest.TestCase):
    def _build_synthetic_dataset(self, n_rows=200):
        rng = np.random.default_rng(42)
        recency = rng.integers(1, 120, size=n_rows)
        frequency = rng.integers(1, 25, size=n_rows)
        spend = recency * 0.2 + frequency * 4 + rng.normal(0, 5, size=n_rows)
        country = rng.choice(["France", "Tunisia", "Italy"], size=n_rows)
        gender = rng.choice(["F", "M"], size=n_rows)

        churn = ((recency > 60) & (frequency < 8)).astype(int)

        return pd.DataFrame(
            {
                "Recency": recency,
                "Frequency": frequency,
                "MonetaryTotal": spend,
                "Country": country,
                "Gender": gender,
                "Churn": churn,
            }
        )

    def test_pipeline_applies_pca_and_predicts(self):
        df = self._build_synthetic_dataset()
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        pipeline = create_training_pipeline(
            numerical_features=["Recency", "Frequency", "MonetaryTotal"],
            categorical_features=["Country", "Gender"],
            classifier=LogisticRegression(max_iter=1000),
            pca_variance_threshold=0.95,
            use_smote=True,
        )

        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        pca_components = get_fitted_pca_component_count(pipeline)

        self.assertEqual(len(preds), len(df))
        self.assertIsNotNone(pca_components)
        self.assertGreaterEqual(pca_components, 1)
        self.assertLessEqual(pca_components, 3)

    def test_train_and_predictor_end_to_end(self):
        df = self._build_synthetic_dataset(300)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset.csv"
            model_path = tmp_path / "best_model.joblib"
            report_path = tmp_path / "metrics.json"
            df.to_csv(dataset_path, index=False)

            best_model_name, metrics = train_and_select_best_model(
                dataset_path=dataset_path,
                model_output_path=model_path,
                report_output_path=report_path,
                target_column="Churn",
                pca_variance_threshold=0.95,
                test_size=0.25,
            )

            self.assertIn(best_model_name, metrics)
            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())

            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["best_model"], best_model_name)

            predictor = ChurnPredictor(model_path=model_path)
            payload = {
                "Recency": 85,
                "Frequency": 5,
                "MonetaryTotal": 120.0,
                "Country": "France",
                "Gender": "F",
            }
            prediction = predictor.predict(payload)

            self.assertIn("churn_prediction", prediction)
            self.assertIn("churn_probability", prediction)
            self.assertIn(prediction["churn_prediction"], [0, 1])


if __name__ == "__main__":
    unittest.main()
