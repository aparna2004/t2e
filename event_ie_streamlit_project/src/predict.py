import os
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ARTIFACTS_DIR, MAX_LENGTH
from src.preprocess import get_nlp
from src.utils import load_json


class EventPredictor:
    def __init__(self, artifacts_dir: str = str(ARTIFACTS_DIR)):
        label_map_path = os.path.join(artifacts_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                "Trained artifacts not found. Run training first so label_map.json, model, and tokenizer are created."
            )

        maps = load_json(label_map_path)
        self.label2id = maps["label2id"]
        self.id2label = {int(key): value for key, value in maps["id2label"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(artifacts_dir, "tokenizer"))
        self.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifacts_dir, "model"))
        self.model.eval()
        self.nlp = get_nlp()

    def classify_sentence(self, sentence: str):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_id].item()
        return self.id2label[pred_id], confidence

    def extract_4w(self, sentence: str) -> Dict:
        doc = self.nlp(sentence)
        who: List[str] = []
        where: List[str] = []
        when: List[str] = []

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                who.append(ent.text)
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                where.append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                when.append(ent.text)

        trigger = None
        for token in doc:
            if token.pos_ in ["VERB", "NOUN"] and not token.is_stop and token.is_alpha:
                trigger = token.text
                break

        return {
            "what": sentence,
            "who": list(dict.fromkeys(who)),
            "where": list(dict.fromkeys(where)),
            "when": list(dict.fromkeys(when)),
            "trigger": trigger,
        }

    def predict_document(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        results: List[Dict] = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue
            label, confidence = self.classify_sentence(sentence_text)
            if label != "None":
                info = self.extract_4w(sentence_text)
                info["event_type"] = label
                info["confidence"] = round(confidence, 4)
                results.append(info)

        return results
