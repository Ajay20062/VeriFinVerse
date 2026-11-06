import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _database_path() -> Path:
    configured = os.getenv("VERIFINVERSE_DATABASE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data" / "history.sqlite3"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                models_json TEXT NOT NULL
            )
            """
        )


def create_history(
    text: str,
    prediction: str,
    confidence: float | None,
    models: list[dict[str, Any]],
) -> tuple[str, int]:
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history
                (created_at, text, prediction, confidence, models_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (created_at, text, prediction, confidence, json.dumps(models)),
        )
        return created_at, int(cursor.lastrowid)


def list_history(limit: int) -> list[sqlite3.Row]:
    with _connect() as connection:
        return list(
            connection.execute(
                """
                SELECT id, created_at, text, prediction, confidence, models_json
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        )
