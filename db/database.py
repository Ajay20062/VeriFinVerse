"""MySQL storage for prediction requests."""

import os
from datetime import datetime, timezone

import mysql.connector


def _connect():
    required = ("MYSQL_HOST", "MYSQL_USER", "MYSQL_DATABASE")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing MySQL configuration: " + ", ".join(missing)
        )

    try:
        port = int(os.getenv("MYSQL_PORT", "3306"))
    except ValueError as error:
        raise RuntimeError("MYSQL_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise RuntimeError("MYSQL_PORT must be between 1 and 65535")

    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=port,
        user=os.environ["MYSQL_USER"],
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.environ["MYSQL_DATABASE"],
        connection_timeout=10,
    )


def initialize_database() -> None:
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    text MEDIUMTEXT NOT NULL,
                    verdict VARCHAR(8) NOT NULL,
                    confidence DOUBLE NOT NULL,
                    model VARCHAR(32) NOT NULL,
                    latency_ms DOUBLE NOT NULL,
                    created_at DATETIME(6) NOT NULL,
                    INDEX idx_predictions_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        connection.commit()
    finally:
        connection.close()


def save_prediction(text, verdict, confidence, model, latency_ms) -> None:
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions
                    (text, verdict, confidence, model, latency_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    text,
                    verdict,
                    confidence,
                    model,
                    latency_ms,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def get_history(limit: int = 100):
    if limit < 1:
        raise ValueError("limit must be positive")

    connection = _connect()
    try:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT id, text, verdict, confidence, model, latency_ms, created_at
                FROM predictions ORDER BY id DESC LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            for row in rows:
                created_at = row.get("created_at")
                if isinstance(created_at, datetime) and created_at.tzinfo is None:
                    row["created_at"] = created_at.replace(tzinfo=timezone.utc)
            return rows
    finally:
        connection.close()
