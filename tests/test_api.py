import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.initialize_database = patch("api.main.initialize_database").start()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)
        self.addCleanup(patch.stopall)

    def test_serves_frontend(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("VeriFinVerse", response.text)
        self.assertEqual(self.client.get("/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_lists_trained_models(self):
        with patch("api.main.available_models", return_value=["logistic"]):
            response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["logistic"]})

    def test_predict_uses_model_and_stores_history(self):
        result = {
            "verdict": "REAL",
            "confidence": 0.8,
            "latency_ms": 1.2,
            "top_signals": [],
        }
        with (
            patch("api.main.available_models", return_value=["logistic"]),
            patch("api.main.predict_one", return_value=result) as predict_one,
            patch("api.main.explain_text", return_value=[]) as explain_text,
            patch("api.main.save_prediction") as save_prediction,
        ):
            response = self.client.post(
                "/predict",
                json={"text": "A test financial claim", "model": "logistic"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "REAL")
        predict_one.assert_called_once_with("logistic", "A test financial claim")
        explain_text.assert_called_once_with("logistic", "A test financial claim")
        save_prediction.assert_called_once()

    def test_history_returns_database_rows(self):
        rows = [{"id": 1, "verdict": "REAL"}]
        with patch("api.main.get_history", return_value=rows):
            response = self.client.get("/history?limit=5")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predictions": rows})


if __name__ == "__main__":
    unittest.main()
