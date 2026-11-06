import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from db import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "MYSQL_HOST": "127.0.0.1",
                "MYSQL_PORT": "3306",
                "MYSQL_USER": "test_user",
                "MYSQL_PASSWORD": "test_password",
                "MYSQL_DATABASE": "verifinverse",
            },
            clear=True,
        )
        self.environment.start()
        self.connection = MagicMock()
        self.cursor = MagicMock()
        self.connection.cursor.return_value.__enter__.return_value = self.cursor
        self.connection.cursor.return_value.__exit__.return_value = False
        self.connector = patch(
            "db.database.mysql.connector.connect",
            return_value=self.connection,
        )
        self.connect = self.connector.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.connector.stop)

    def test_requires_mysql_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MYSQL_HOST"):
                database._connect()
        self.connect.assert_not_called()

    def test_rejects_invalid_port(self):
        os.environ["MYSQL_PORT"] = "not-a-port"
        with self.assertRaisesRegex(RuntimeError, "MYSQL_PORT"):
            database._connect()
        self.connect.assert_not_called()

    def test_initializes_mysql_table(self):
        database.initialize_database()

        statement = self.cursor.execute.call_args.args[0]
        self.assertIn("CREATE TABLE IF NOT EXISTS predictions", statement)
        self.assertIn("ENGINE=InnoDB", statement)
        self.connection.commit.assert_called_once()
        self.connection.close.assert_called_once()

    def test_saves_prediction_with_bound_parameters(self):
        database.save_prediction("claim", "REAL", 0.8, "logistic", 1.2)

        statement, parameters = self.cursor.execute.call_args.args
        self.assertIn("VALUES (%s, %s, %s, %s, %s, %s)", statement)
        self.assertEqual(parameters[:5], ("claim", "REAL", 0.8, "logistic", 1.2))
        self.assertEqual(parameters[5].tzinfo, None)
        self.connection.commit.assert_called_once()

    def test_reads_history_with_limit_parameter(self):
        expected = [
            {
                "id": 1,
                "verdict": "REAL",
                "created_at": datetime(2026, 11, 6, 10, 0),
            }
        ]
        self.cursor.fetchall.return_value = expected

        result = database.get_history(7)

        self.assertIs(result, expected)
        self.assertEqual(result[0]["created_at"].tzinfo, timezone.utc)
        self.cursor.execute.assert_called_once()
        self.assertEqual(self.cursor.execute.call_args.args[1], (7,))
        self.assertTrue(self.connection.cursor.call_args.kwargs["dictionary"])
        self.connection.close.assert_called_once()

    def test_rejects_non_positive_history_limit(self):
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            database.get_history(0)
        self.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
