"""Download Fin-Fact JSON and export binary fact-check claims to CSV."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/IIT-DM/Fin-Fact/FinFact/finfact.json"
DEFAULT_INPUT = Path("data/raw/finfact.json")
DEFAULT_OUTPUT = Path("data/raw/finfact_claims.csv")


def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            input_path.write_bytes(response.read())

    with input_path.open("r", encoding="utf-8") as source:
        records = json.load(source)
    if isinstance(records, dict):
        records = records.get("data", records.get("records", []))
    if not isinstance(records, list):
        raise ValueError("Fin-Fact JSON must contain a list of records.")

    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        label = str(record.get("label", "")).strip().lower()
        if label not in {"true", "false"}:
            continue
        claim = record.get("claim")
        if claim is None or not str(claim).strip():
            continue
        rows.append(
            {
                "article": str(claim).strip(),
                "label": "REAL" if label == "true" else "FAKE",
                "source_url": str(record.get("url") or "").strip(),
            }
        )

    frame = pd.DataFrame(rows, columns=["article", "label", "source_url"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    counts = frame["label"].value_counts().to_dict() if not frame.empty else {}
    print(f"Wrote {len(frame)} claims to {output_path}")
    print(f"Label counts: REAL={counts.get('REAL', 0)}, FAKE={counts.get('FAKE', 0)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
