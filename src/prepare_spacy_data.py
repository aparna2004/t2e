from pathlib import Path
import json
import random

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "maven_raw"
OUTPUT_DIR = BASE_DIR / "artifacts" / "spacy_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPACY_PATH = OUTPUT_DIR / "train.spacy"
DEV_SPACY_PATH = OUTPUT_DIR / "dev.spacy"

INPUT_FILES = [
    RAW_DIR / "train.jsonl",
    RAW_DIR / "dev.jsonl",
    RAW_DIR / "test.jsonl",
]

RANDOM_STATE = 42
TARGET_SIZE = 300

EVENT_MAP = {
    "attack": "Attack",
    "hostile_encounter": "Attack",
    "bombing": "Attack",

    "catastrophe": "Disaster",
    "fire_explosion": "Disaster",
    "fire-explosion": "Disaster",
    "crash": "Disaster",

    "bodily_harm": "Injury",
    "bodily-harm": "Injury",

    "investigation": "Investigation",
    "investigate": "Investigation",
    "check": "Investigation",

    "protest": "Protest",
    "demonstrate": "Protest",
}

TARGET_LABELS = ["Attack", "Disaster", "Injury", "Investigation", "Protest", "None"]

LABEL_PRIORITY = ["Attack", "Disaster", "Injury", "Investigation", "Protest"]

EVENTISH_WORDS = [
    "attack", "attacked", "ambush", "bomb", "bombing", "raid",
    "earthquake", "tsunami", "flood", "storm", "explosion", "fire",
    "crash", "collapse", "injured", "wounded", "killed", "dead",
    "investigation", "probe", "protest", "demonstration", "rally",
    "casualties", "damage", "loss of life"
]


def normalize_event_type(event_type: str) -> str:
    event_type = str(event_type or "").strip().lower().replace(" ", "_")

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


def looks_eventish(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in EVENTISH_WORDS)


def extract_rows_from_record(record):
    rows = []

    content = record.get("content", [])
    events = record.get("events", [])

    if not isinstance(content, list) or not content:
        return rows

    sent_labels = {i: [] for i in range(len(content))}

    for event in events:
        mapped_label = normalize_event_type(event.get("type", ""))
        if mapped_label == "None":
            continue

        mentions = event.get("mention", [])
        if not isinstance(mentions, list):
            continue

        for mention in mentions:
            sent_id = mention.get("sent_id")
            if isinstance(sent_id, int) and 0 <= sent_id < len(content):
                sent_labels[sent_id].append(mapped_label)

    for i, sent_obj in enumerate(content):
        sentence = str(sent_obj.get("sentence", "")).strip()
        if not sentence:
            continue

        label = choose_label(sent_labels.get(i, []))
        rows.append({"sentence": sentence, "event_type": label})

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df[~((df["event_type"] == "None") & (df["sentence"].apply(looks_eventish)))]
    df = df.reset_index(drop=True)

    return df


def balance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBefore balancing:")
    print(df["event_type"].value_counts().to_string())

    parts = []
    for label in TARGET_LABELS:
        group = df[df["event_type"] == label].copy()
        if group.empty:
            continue

        if len(group) >= TARGET_SIZE:
            group = group.sample(n=TARGET_SIZE, random_state=RANDOM_STATE)
        else:
            group = group.sample(n=TARGET_SIZE, replace=True, random_state=RANDOM_STATE)

        parts.append(group)

    balanced_df = pd.concat(parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nAfter balancing:")
    print(balanced_df["event_type"].value_counts().to_string())

    return balanced_df


def create_docbin(df: pd.DataFrame, labels, nlp):
    docbin = DocBin()

    for _, row in df.iterrows():
        text = str(row["sentence"]).strip()
        label = str(row["event_type"]).strip()

        if not text or not label:
            continue

        text = text.lower()
        doc = nlp.make_doc(text)
        doc.cats = {lab: 0.0 for lab in labels}
        doc.cats[label] = 1.0
        docbin.add(doc)

    return docbin


def main():
    rows = load_all_rows()
    if not rows:
        raise ValueError("No rows extracted from MAVEN JSONL files.")

    df = pd.DataFrame(rows)
    df = clean_dataframe(df)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    balanced_df = balance_dataframe(df)

    labels = sorted(balanced_df["event_type"].unique().tolist())
    print("\nLabels:")
    print(labels)

    train_df, dev_df = train_test_split(
        balanced_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=balanced_df["event_type"],
    )

    nlp = spacy.blank("en")

    train_docbin = create_docbin(train_df, labels, nlp)
    dev_docbin = create_docbin(dev_df, labels, nlp)

    train_docbin.to_disk(TRAIN_SPACY_PATH)
    dev_docbin.to_disk(DEV_SPACY_PATH)

    print(f"\nSaved: {TRAIN_SPACY_PATH}")
    print(f"Saved: {DEV_SPACY_PATH}")
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")


if __name__ == "__main__":
    main()