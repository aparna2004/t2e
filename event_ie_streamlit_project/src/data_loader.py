import json
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _get_sentence_text(sentence_obj):
    if isinstance(sentence_obj, dict):
        tokens = sentence_obj.get("tokens", [])
        return " ".join(tokens)
    if isinstance(sentence_obj, list):
        return " ".join(sentence_obj)
    return str(sentence_obj)


def build_sentence_level_samples(maven_docs: List[Dict]) -> List[Dict]:
    """
    Converts MAVEN-style documents into sentence-level samples.

    Expected rough structure:
      {
        "content": [{"tokens": [...]}, ...],
        "events": [
          {
            "type": "Attack",
            "mention": [
              {"sent_id": 0, "offset": [2, 3], "trigger_word": "attacked"}
            ]
          }
        ]
      }
    """
    samples: List[Dict] = []

    for doc in maven_docs:
        content = doc.get("content", [])
        events = doc.get("events", [])

        sent_id_to_events: Dict[int, List[Dict]] = {}

        for event in events:
            event_type = event.get("type", "None")
            mentions = event.get("mention", []) or event.get("mentions", [])
            for mention in mentions:
                sent_id = mention.get("sent_id")
                if sent_id is None:
                    continue
                offset = mention.get("offset", [0, 0])
                trigger_word = mention.get("trigger_word", "")
                sent_id_to_events.setdefault(sent_id, []).append(
                    {
                        "event_type": event_type,
                        "offset": offset,
                        "trigger_word": trigger_word,
                    }
                )

        for sent_id, sentence_obj in enumerate(content):
            sentence_text = _get_sentence_text(sentence_obj).strip()
            if not sentence_text:
                continue

            if sent_id in sent_id_to_events:
                for event_info in sent_id_to_events[sent_id]:
                    samples.append(
                        {
                            "text": sentence_text,
                            "label": event_info["event_type"],
                            "trigger_word": event_info["trigger_word"],
                            "offset": event_info["offset"],
                        }
                    )
            else:
                samples.append(
                    {
                        "text": sentence_text,
                        "label": "None",
                        "trigger_word": "",
                        "offset": [0, 0],
                    }
                )

    return samples
