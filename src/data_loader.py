"""Configurable loaders for labeled and auxiliary financial text CSV files."""

from pathlib import Path
from typing import Mapping

import pandas as pd


_LABEL_ALIASES = {
    "real": "REAL",
    "genuine": "REAL",
    "true": "REAL",
    "fake": "FAKE",
    "false": "FAKE",
    "fabricated": "FAKE",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    return pd.read_csv(source)


def _clean_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def load_primary_dataset(
    path: str | Path,
    text_column: str,
    label_column: str,
    label_mapping: Mapping[object, str] | None = None,
) -> pd.DataFrame:
    """Return unique, non-empty labeled records with canonical REAL/FAKE labels."""
    frame = _read_csv(path)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing columns {sorted(missing)} in {path}; "
            f"available columns: {list(frame.columns)}"
        )

    data = frame[[text_column, label_column]].copy()
    data.columns = ["text", "raw_label"]
    data["text"] = _clean_text(data["text"])
    data = data.dropna(subset=["text", "raw_label"])
    data = data[data["text"] != ""]

    aliases = dict(_LABEL_ALIASES)
    if label_mapping:
        aliases.update(
            {
                str(raw).strip().casefold(): str(label).strip().upper()
                for raw, label in label_mapping.items()
            }
        )
    invalid = set(aliases.values()) - {"REAL", "FAKE"}
    if invalid:
        raise ValueError(f"Label targets must be REAL or FAKE, got {sorted(invalid)}")

    labels = data["raw_label"].map(
        lambda value: aliases.get(str(value).strip().casefold())
    )
    unknown = data.loc[labels.isna(), "raw_label"].unique().tolist()
    if unknown:
        raise ValueError(
            "Unmapped labels found. Supply --label-map (for example "
            "'0=FAKE,1=REAL' only if correct for this dataset). "
            f"Unknown values: {unknown[:10]}"
        )
    data["label"] = labels
    data = data.drop(columns=["raw_label"])
    data["_key"] = data["text"].str.casefold()
    data = data.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No usable labeled articles found in {path}")
    return data[["text", "label"]]


def load_auxiliary_text(path: str | Path, text_column: str) -> pd.DataFrame:
    """Load unlabeled FiQA/PhraseBank text without assigning fake/real labels."""
    frame = _read_csv(path)
    if text_column not in frame.columns:
        raise ValueError(
            f"Missing text column '{text_column}' in {path}; "
            f"available columns: {list(frame.columns)}"
        )
    data = pd.DataFrame({"text": _clean_text(frame[text_column])})
    data = data.dropna(subset=["text"])
    data = data[data["text"] != ""]
    data["_key"] = data["text"].str.casefold()
    return data.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
