"""SQLite storage for prediction requests."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.sqlite3"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                model TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction(text, verdict, confidence, model, latency_ms) -> None:
    initialize_database()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO predictions
                (text, verdict, confidence, model, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                verdict,
                confidence,
                model,
                latency_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_history(limit: int = 100):
    if limit < 1:
        raise ValueError("limit must be positive")
    initialize_database()
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, text, verdict, confidence, model, latency_ms, created_at
            FROM predictions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
