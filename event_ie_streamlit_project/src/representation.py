def build_event_card(event: dict) -> dict:
    return {
        "event_type": event.get("event_type", "Unknown"),
        "trigger": event.get("trigger", "Not found"),
        "what": event.get("what", ""),
        "who": event.get("who", []),
        "where": event.get("where", []),
        "when": event.get("when", []),
        "confidence": event.get("confidence"),
    }
