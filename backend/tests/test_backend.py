import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import ModelRegistry, normalize_label


class FakeClassifier:
    classes_ = ["fake", "real"]

    def predict(self, texts):
        return ["real" for _ in texts]

    def predict_proba(self, texts):
        return [[0.15, 0.85] for _ in texts]


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "VERIFINVERSE_DATABASE_PATH": os.path.join(
                    self.temporary_directory.name, "history.sqlite3"
                )
            },
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary_directory.cleanup()

    def test_analysis_uses_configured_model_and_persists_history(self):
        with TestClient(app) as client:
            registry = ModelRegistry()
            registry._adapters["logistic_regression"] = FakeClassifier()
            app.state.model_registry = registry

            response = client.post(
                "/api/analyze",
                json={"text": "A verified financial headline."},
            )

            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["prediction"], "REAL")
            self.assertAlmostEqual(result["confidence"], 0.85)
            self.assertEqual(result["models"][0]["model"], "logistic_regression")

            history = client.get("/api/history")
            self.assertEqual(history.status_code, 200)
            self.assertEqual(history.json()[0]["text"], "A verified financial headline.")
            self.assertEqual(history.json()[0]["prediction"], "REAL")

    def test_analysis_reports_unavailable_models_without_fabricating_a_result(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/analyze",
                json={"text": "A news headline with no configured model."},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("No trained models are available", response.json()["detail"]["message"])

    def test_input_rejects_blank_text_and_duplicate_model_names(self):
        with TestClient(app) as client:
            blank_text = client.post("/api/analyze", json={"text": "   "})
            duplicate_models = client.post(
                "/api/analyze",
                json={
                    "text": "A headline.",
                    "models": ["logistic_regression", "logistic_regression"],
                },
            )

        self.assertEqual(blank_text.status_code, 422)
        self.assertEqual(duplicate_models.status_code, 422)

    def test_label_normalization(self):
        self.assertEqual(normalize_label("not_real"), "FAKE")
        self.assertEqual(normalize_label("Legitimate"), "REAL")
        self.assertEqual(normalize_label("label_0"), "LABEL_0")


if __name__ == "__main__":
    unittest.main()
