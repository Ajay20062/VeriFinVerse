"""Download Fin-Fact and prepare its verified binary claim labels."""

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


SOURCE_URL = (
    "https://raw.githubusercontent.com/IIT-DM/Fin-Fact/refs/heads/FinFact/"
    "finfact.json"
)
OUTPUT_PATH = Path("data/raw/finfact_claims.csv")
LABEL_MAP = {"true": "REAL", "false": "FAKE"}


def main() -> None:
    with urlopen(SOURCE_URL, timeout=60) as response:
        records = json.load(response)
    if not isinstance(records, list) or not records:
        raise ValueError("Fin-Fact source returned no records")

    frame = pd.DataFrame(records)
    required = {"claim", "label", "url"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Fin-Fact records must include {sorted(required)}")

    frame["article"] = frame["claim"].astype("string").str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    frame["source_label"] = frame["label"].astype("string").str.strip().str.casefold()
    unknown = set(frame["source_label"].dropna().unique()) - {
        "true",
        "false",
        "nei",
    }
    if unknown:
        raise ValueError(f"Unexpected Fin-Fact labels: {sorted(unknown)}")

    nei_count = int((frame["source_label"] == "nei").sum())
    frame = frame[frame["source_label"].isin(LABEL_MAP)].copy()
    frame["label"] = frame["source_label"].map(LABEL_MAP)
    frame["source_url"] = frame["url"].astype("string").str.strip()
    frame = frame.dropna(subset=["article", "source_url"])
    frame = frame[(frame["article"] != "") & (frame["source_url"] != "")]

    frame["_key"] = frame["article"].str.casefold()
    conflicting = frame.groupby("_key")["label"].nunique()
    conflicting_keys = set(conflicting[conflicting > 1].index)
    frame = frame[~frame["_key"].isin(conflicting_keys)]
    frame = frame.drop_duplicates("_key")

    frame = frame[["article", "label", "source_url"]].reset_index(drop=True)
    counts = frame["label"].value_counts()
    if set(counts.index) != {"REAL", "FAKE"} or counts.min() < 3:
        raise ValueError(f"Need at least three examples per class; got {counts.to_dict()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(frame)} labeled claims to {OUTPUT_PATH}; "
        f"labels={counts.to_dict()}; excluded NEI={nei_count}; "
        f"excluded conflicting duplicate claims={len(conflicting_keys)}"
    )


if __name__ == "__main__":
    main()
