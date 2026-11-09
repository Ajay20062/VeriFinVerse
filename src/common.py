"""Shared text and label utilities for VeriFinVerse."""

import re
from typing import Optional, Union

REAL = 0
FAKE = 1
LABEL_NAMES = {REAL: "REAL", FAKE: "FAKE"}

_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Lowercase text and remove URLs, symbols, and excess whitespace."""
    if value is None:
        return ""
    text = str(value).lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _NON_WORD_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_label(value: object) -> Optional[int]:
    """Map supported real/fake labels to REAL=0 or FAKE=1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return REAL if value else FAKE
    if isinstance(value, (int, float)):
        if value == REAL:
            return REAL
        if value == FAKE:
            return FAKE

    label = str(value).strip().lower()
    if label in {"real", "true", "0"}:
        return REAL
    if label in {"fake", "false", "1"}:
        return FAKE
    return None
