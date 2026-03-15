import json
import re
from pathlib import Path
from typing import Dict, List

import spacy


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "spacy_event_model"

DEFAULT_LABELS = [
    "Attack",
    "Disaster",
    "Injury",
    "Investigation",
    "Protest",
    "None",
]

TRIGGER_HINTS = {
    "Attack": [
        "attack", "attacked", "ambush", "ambushed", "bomb", "bombing",
        "raid", "assault", "fired", "shooting", "detonated"
    ],
    "Disaster": [
        "earthquake", "tsunami", "flood", "storm", "explosion",
        "fire", "landslide", "crash", "collapse", "leak"
    ],
    "Injury": [
        "injured", "injury", "hurt", "wounded", "killed", "dead", "died"
    ],
    "Investigation": [
        "investigation", "investigating", "investigated", "probe",
        "probed", "enquiry", "inquiry", "examined", "launched"
    ],
    "Protest": [
        "protest", "protested", "demonstration", "demonstrators",
        "rally", "march", "protesters", "gathered"
    ],
}

GENERIC_PARTICIPANT_WORDS = {
    "bomber", "workers", "worker", "passengers", "passenger", "students",
    "student", "police", "suspects", "suspect", "demonstrators",
    "demonstrator", "residents", "resident", "people", "officials",
    "official", "militants", "militant", "protesters", "protester",
    "officers", "officer", "victims", "victim", "rebels", "rebel",
    "soldiers", "soldier", "civilians", "civilian", "driver", "drivers",
    "firefighters", "firefighter"
}

try:
    NLP_EVENT = spacy.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception:
    NLP_EVENT = None
    MODEL_LOADED = False

try:
    NLP_NER = spacy.load("en_core_web_sm")
except Exception:
    NLP_NER = spacy.blank("en")

EVENT_LABELS = DEFAULT_LABELS


def simple_sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_dates(text: str) -> List[str]:
    vals = []

    if NLP_NER.has_pipe("ner"):
        doc = NLP_NER(text)
        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME"}:
                item = normalize_space(ent.text)
                if item and item not in vals:
                    vals.append(item)

    extra_patterns = [
        r"\bon\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+(?:morning|evening|afternoon|night))?\b",
        r"\b(last night|this morning|this evening|this afternoon|yesterday|today|tomorrow)\b",
        r"\bon\s+[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?\b",
    ]

    for pattern in extra_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            if isinstance(match, tuple):
                match = " ".join([m for m in match if m]).strip()
            item = normalize_space(str(match))
            if item and item not in vals:
                vals.append(item)

    return vals[:3]


def clean_location_text(loc: str) -> str:
    loc = normalize_space(loc)
    loc = re.sub(
        r"\s+\bon\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+(?:morning|evening|afternoon|night))?\b.*$",
        "",
        loc,
        flags=re.IGNORECASE,
    )
    loc = re.sub(
        r"\s+\b(last night|this morning|this evening|this afternoon|yesterday|today|tomorrow)\b.*$",
        "",
        loc,
        flags=re.IGNORECASE,
    )
    loc = re.sub(r"[,\s]+$", "", loc)
    return loc.strip()


def extract_locations(text: str) -> List[str]:
    vals = []

    if NLP_NER.has_pipe("ner"):
        doc = NLP_NER(text)
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC", "FAC"}:
                item = clean_location_text(ent.text)
                if item and item not in vals:
                    vals.append(item)

    location_patterns = [
        r"\b(?:in|at|near|outside|inside|around|across)\s+([A-Z][a-zA-Z\s\-]+)",
        r"\b(?:in|at|near|outside|inside|around|across)\s+the\s+([a-zA-Z][a-zA-Z\s\-]+?)(?:,|\.|$)",
    ]

    for pattern in location_patterns:
        for match in re.findall(pattern, text):
            loc = clean_location_text(match)
            if loc and loc.lower() not in {"incident", "event", "scene"} and loc not in vals:
                vals.append(loc)

    return vals[:3]


def extract_participants(text: str) -> List[str]:
    vals = []

    if NLP_NER.has_pipe("ner"):
        doc = NLP_NER(text)

        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "NORP"}:
                item = normalize_space(ent.text)
                if item and item not in vals:
                    vals.append(item)

        if doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                chunk_text = normalize_space(chunk.text)
                lowered = chunk_text.lower()

                if any(word in lowered for word in GENERIC_PARTICIPANT_WORDS):
                    if chunk_text not in vals:
                        vals.append(chunk_text)

    return vals[:5]


def infer_trigger(sentence: str, event_type: str) -> str:
    sent_lower = sentence.lower()

    for word in TRIGGER_HINTS.get(event_type, []):
        if word in sent_lower:
            return word

    fallback_patterns = [
        r"\b(?:attack|attacked|ambush|ambushed|bomb|bombing|raid|assault|detonated)\b",
        r"\b(?:earthquake|tsunami|flood|storm|explosion|fire|landslide|crash|collapse|leak)\b",
        r"\b(?:injured|injury|hurt|wounded|killed|dead|died)\b",
        r"\b(?:investigation|investigating|investigated|probe|probed|enquiry|inquiry|examined)\b",
        r"\b(?:protest|protested|demonstration|rally|march|gathered)\b",
    ]

    for pattern in fallback_patterns:
        match = re.search(pattern, sent_lower)
        if match:
            return match.group(0)

    words = re.findall(r"\b[a-zA-Z]+\b", sent_lower)
    return words[0] if words else "unknown"


def get_sample_articles() -> Dict[str, str]:
    return {
        "Warehouse fire": (
            "A fire broke out at a textile warehouse in Coimbatore on Tuesday evening. "
            "Four workers were injured and taken to the district hospital. "
            "Officials said the police opened an investigation into the incident."
        ),
        "Embassy bombing": (
            "A suicide bomber detonated explosives outside the embassy in Kabul on Monday evening, "
            "killing five people and injuring dozens."
        ),
        "City protest": (
            "Thousands of protesters gathered outside Parliament in Delhi on Friday evening. "
            "Police later opened an investigation after clashes near the main gate."
        ),
        "Industrial leak": (
            "Three workers were injured in a chemical leak at a warehouse in Mumbai last night."
        ),
    }


def load_metrics(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "trigger_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "event_type_accuracy": 0.0,
        "per_class_metrics": [],
        "confusion_summary": [],
    }


def predict_sentence(sentence: str):
    if not MODEL_LOADED or NLP_EVENT is None:
        return "None", 0.0

    doc = NLP_EVENT(sentence)
    if not doc.cats:
        return "None", 0.0

    predicted_label = max(doc.cats, key=doc.cats.get)
    confidence = float(doc.cats[predicted_label])
    return predicted_label, confidence


def extract_event_information(text: str, confidence_threshold: float = 0.35) -> Dict:
    sentences = simple_sentence_split(text)
    all_participants = extract_participants(text)
    all_dates = extract_dates(text)
    all_locations = extract_locations(text)

    events = []
    sentence_debug = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        predicted_label, confidence = predict_sentence(sentence)

        participants = extract_participants(sentence)
        dates = extract_dates(sentence)
        locations = extract_locations(sentence)

        sentence_debug.append(
            {
                "sentence": sentence,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4),
                "participants_found": ", ".join(participants) if participants else "",
                "dates_found": ", ".join(dates) if dates else "",
                "locations_found": ", ".join(locations) if locations else "",
            }
        )

        if predicted_label != "None" and confidence >= confidence_threshold:
            events.append(
                {
                    "event_type": predicted_label,
                    "trigger": infer_trigger(sentence, predicted_label),
                    "sentence": sentence,
                    "participants": participants[:5],
                    "date_time": dates[0] if dates else None,
                    "location": locations[0] if locations else None,
                    "confidence": round(confidence, 3),
                }
            )

    steps = [
        {
            "title": "Step 1 — Preprocessing",
            "description": "The article is cleaned and split into sentences.",
            "dataframe": [{"sentence_id": i + 1, "sentence": s} for i, s in enumerate(sentences)],
        },
        {
            "title": "Step 2 — Participant / date / location extraction",
            "description": "spaCy NER and lightweight phrase rules are used to collect who, when, and where information.",
            "json": {
                "participants": all_participants,
                "dates": all_dates,
                "locations": all_locations,
            },
        },
        {
            "title": "Step 3 — spaCy text classification",
            "description": "Each sentence is passed through the trained spaCy text classifier.",
            "dataframe": sentence_debug,
        },
        {
            "title": "Step 4 — Structured event assembly",
            "description": "Predicted event sentences are converted into structured event records.",
            "json": {"events": events},
        },
    ]

    return {
        "events": events,
        "stats": {
            "num_sentences": len(sentences),
            "num_entities": len(all_participants),
            "num_dates": len(all_dates),
            "num_locations": len(all_locations),
            "model_loaded": MODEL_LOADED,
        },
        "steps": steps,
    }