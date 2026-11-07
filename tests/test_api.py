import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import main
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
        self.assertIn('href="/static/styles.css"', response.text)
        self.assertIn('src="/static/app.js"', response.text)
        css = self.client.get("/static/styles.css")
        javascript = self.client.get("/static/app.js")
        self.assertEqual(css.status_code, 200)
        self.assertIn("text/css", css.headers["content-type"])
        self.assertEqual(javascript.status_code, 200)
        self.assertIn("javascript", javascript.headers["content-type"])
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_lists_trained_models(self):
        with patch("api.main.available_models", return_value=["logistic"]):
            response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["logistic"]})

    def test_rejects_unavailable_model(self):
        with patch("api.main.available_models", return_value=[]):
            response = self.client.post(
                "/predict",
                json={"text": "A test financial claim", "model": "logistic"},
            )

        self.assertEqual(response.status_code, 503)

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

    def test_history_rejects_invalid_limit(self):
        response = self.client.get("/history?limit=101")

        self.assertEqual(response.status_code, 400)

    def test_history_returns_service_unavailable_when_mysql_is_unconfigured(self):
        with patch("api.main.get_history", side_effect=RuntimeError("Missing MySQL configuration")):
            response = self.client.get("/history?limit=5")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Missing MySQL configuration", response.json()["detail"])

    def test_startup_keeps_web_app_available_when_mysql_is_unconfigured(self):
        with patch(
            "api.main.initialize_database",
            side_effect=RuntimeError("Missing MySQL configuration"),
        ), patch("api.main.logger.warning") as warning:
            main.startup()
            warning.assert_called_once()

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
