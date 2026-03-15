from pathlib import Path
import json
import csv

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "maven_raw"
OUTPUT_PATH = BASE_DIR / "data" / "event_sentences.csv"

INPUT_FILES = [
    RAW_DIR / "train.jsonl",
    RAW_DIR / "dev.jsonl",
    RAW_DIR / "test.jsonl",
]

RANDOM_STATE = 42
MAX_PER_LABEL = 500


# Map MAVEN event types to your reduced labels
EVENT_MAP = {
    "attack": "Attack",
    "hostile_encounter": "Attack",
    "bombing": "Attack",

    "arrest_jail": "Arrest",
    "arrest-jail": "Arrest",

    "catastrophe": "Disaster",
    "damaging": "Disaster",
    "destroying": "Disaster",
    "fire_explosion": "Disaster",
    "fire-explosion": "Disaster",

    "election": "Election",
    "elect": "Election",

    "injury": "Injury",
    "bodily_harm": "Injury",
    "bodily-harm": "Injury",
    "death": "Injury",
    "killing": "Injury",

    "investigation": "Investigation",
    "investigate": "Investigation",
    "check": "Investigation",
    "suspicion": "Investigation",

    "protest": "Protest",
    "demonstrate": "Protest",
}

TARGET_LABELS = [
    "Attack",
    "Arrest",
    "Disaster",
    "Election",
    "Injury",
    "Investigation",
    "Protest",
    "None",
]

# Priority when one sentence has multiple mapped event labels
LABEL_PRIORITY = [
    "Attack",
    "Arrest",
    "Disaster",
    "Election",
    "Injury",
    "Investigation",
    "Protest",
]


def normalize_event_type(event_type: str) -> str:
    event_type = str(event_type or "").strip().lower()
    event_type = event_type.replace(" ", "_")

    if event_type in EVENT_MAP:
        return EVENT_MAP[event_type]

    for key, value in EVENT_MAP.items():
        if key in event_type:
            return value

    return "None"


def choose_label(labels):
    if not labels:
        return "None"

    for label in LABEL_PRIORITY:
        if label in labels:
            return label

    return "None"


def extract_rows_from_record(record):
    rows = []

    content = record.get("content", [])
    events = record.get("events", [])

    if not isinstance(content, list) or not content:
        return rows

    sent_labels = {i: [] for i in range(len(content))}

    # Read top-level event mentions and attach labels to the correct sentence via sent_id
    for event in events:
        mapped_label = normalize_event_type(event.get("type", ""))

        if mapped_label == "None":
            continue

        mentions = event.get("mention", [])
        if not isinstance(mentions, list):
            continue

        for mention in mentions:
            sent_id = mention.get("sent_id", None)
            if isinstance(sent_id, int) and 0 <= sent_id < len(content):
                sent_labels[sent_id].append(mapped_label)

    for i, sent_obj in enumerate(content):
        sentence = str(sent_obj.get("sentence", "")).strip()
        if not sentence:
            continue

        label = choose_label(sent_labels.get(i, []))

        rows.append(
            {
                "sentence": sentence,
                "event_type": label,
            }
        )

    return rows


def load_all_rows():
    rows = []

    for file_path in INPUT_FILES:
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        print(f"Reading: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    rows.extend(extract_rows_from_record(record))
                except Exception as e:
                    print(f"Skipping bad JSON in {file_path.name} line {line_num}: {e}")

    return rows


def balance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBefore balancing:")
    print(df["event_type"].value_counts().to_string())

    parts = []

    for label in TARGET_LABELS:
        group = df[df["event_type"] == label].copy()
        if group.empty:
            continue

        n = min(len(group), MAX_PER_LABEL)
        group = group.sample(n=n, random_state=RANDOM_STATE)
        parts.append(group)

    balanced_df = pd.concat(parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nAfter balancing:")
    print(balanced_df["event_type"].value_counts().to_string())

    return balanced_df


def main():
    rows = load_all_rows()

    if not rows:
        raise ValueError("No rows extracted. Check data/maven_raw/ files.")

    df = pd.DataFrame(rows)

    df["sentence"] = (
        df["sentence"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["event_type"] = df["event_type"].astype(str).str.strip()

    df = df[
        (df["sentence"] != "")
        & (df["event_type"] != "")
        & (df["event_type"].isin(TARGET_LABELS))
    ].drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after cleaning.")

    balanced_df = balance_dataframe(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    print(f"\nSaved clean dataset to: {OUTPUT_PATH}")
    print(f"Total rows: {len(balanced_df)}")
    print("\nSample:")
    print(balanced_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()