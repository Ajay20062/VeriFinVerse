import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from app import create_app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.app = create_app(
            {
                "TESTING": True,
                "MODEL_DIR": self.root / "models",
                "DATASET_PATH": self.root / "claims.csv",
                "HISTORY_PATH": self.root / "history.csv",
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_api_v1_and_legacy_routes_are_compatible(self):
        versioned = self.client.get("/api/v1/models")
        legacy = self.client.get("/models")

        self.assertEqual(versioned.status_code, 200)
        self.assertEqual(versioned.json, legacy.json)
        self.assertEqual(versioned.json, {"models": []})

    def test_factory_serves_pages_and_versioned_algorithms(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/workbench").status_code, 200)
        response = self.client.get("/api/v1/algorithms")
        self.assertEqual(response.status_code, 200)
        self.assertIn("algorithms", response.json)

    def test_request_logging_includes_method_path_and_status(self):
        with self.assertLogs(self.app.logger.name, level=logging.INFO) as logs:
            response = self.client.get("/api/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            any("GET /api/v1/models 200" in message for message in logs.output)
        )

    def test_environment_settings_configure_factory_defaults(self):
        environment = {
            "VERIFINVERSE_HOST": "0.0.0.0",
            "VERIFINVERSE_PORT": "8080",
            "VERIFINVERSE_DEBUG": "true",
            "VERIFINVERSE_LOG_LEVEL": "WARNING",
            "VERIFINVERSE_HISTORY_PATH": "tmp/history.csv",
        }
        with patch.dict(os.environ, environment, clear=False):
            app = create_app()

        self.assertEqual(app.config["HOST"], "0.0.0.0")
        self.assertEqual(app.config["PORT"], 8080)
        self.assertTrue(app.config["DEBUG"])
        self.assertEqual(app.logger.level, logging.WARNING)
        self.assertEqual(
            app.config["HISTORY_PATH"],
            Path(__file__).resolve().parents[1] / "tmp/history.csv",
        )

    def test_invalid_port_environment_setting_is_rejected(self):
        with patch.dict(os.environ, {"VERIFINVERSE_PORT": "not-a-port"}):
            with self.assertRaisesRegex(ValueError, "VERIFINVERSE_PORT must be an integer"):
                create_app()

    def test_predict_all_returns_and_records_each_available_model(self):
        model_dir = self.root / "models"
        model_dir.mkdir()
        (model_dir / "logistic.joblib").touch()
        (model_dir / "svm.joblib").touch()

        def explain_for_model(text, model):
            return {
                "verdict": "FAKE" if model == "svm" else "REAL",
                "confidence": 0.8,
                "probabilities": {"REAL": 0.8, "FAKE": 0.2},
                "signals": [],
                "known_words": 1,
            }

        with (
            patch.object(app_module, "get_model", side_effect=lambda name: name),
            patch.object(app_module, "explain_text", side_effect=explain_for_model),
        ):
            response = self.client.post(
                "/api/v1/predict/all",
                json={"text": "A financial claim"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [result["model"] for result in response.json["results"]],
            ["logistic", "svm"],
        )
        self.assertEqual(
            [result["verdict"] for result in response.json["results"]],
            ["REAL", "FAKE"],
        )
        history = self.client.get("/api/v1/history").json["history"]
        self.assertEqual([entry["model"] for entry in history], ["svm", "logistic"])

    def test_predict_all_reports_when_no_models_are_available(self):
        response = self.client.post("/api/v1/predict/all", json={"text": "A claim"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json["results"], [])


if __name__ == "__main__":
    unittest.main()
