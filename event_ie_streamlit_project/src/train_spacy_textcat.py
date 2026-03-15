from pathlib import Path
import json
import random

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "artifacts" / "spacy_data"
MODEL_DIR = BASE_DIR / "artifacts" / "spacy_event_model"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

TRAIN_PATH = DATA_DIR / "train.spacy"
DEV_PATH = DATA_DIR / "dev.spacy"

EPOCHS = 6
BATCH_SIZE = 16
RANDOM_SEED = 42


def load_docs(path: Path, vocab):
    docbin = DocBin().from_disk(path)
    return list(docbin.get_docs(vocab))


def make_examples(nlp, docs):
    examples = []
    for doc in docs:
        pred_doc = nlp.make_doc(doc.text)
        pred_doc.cats = {}
        examples.append(Example(pred_doc, doc))
    return examples


def evaluate_model(nlp, docs, labels):
    y_true = []
    y_pred = []

    for doc in docs:
        pred_doc = nlp(doc.text)

        true_label = max(doc.cats, key=doc.cats.get) if doc.cats else "None"
        pred_label = max(pred_doc.cats, key=pred_doc.cats.get) if pred_doc.cats else "None"

        y_true.append(true_label)
        y_pred.append(pred_label)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_metrics = []
    for label in labels:
        if label in report:
            per_class_metrics.append(
                {
                    "class": label,
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                    "support": int(report[label]["support"]),
                }
            )

    confusion_summary = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(cm[i][j])
            if i != j and count > 0:
                confusion_summary.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": count,
                    }
                )

    confusion_summary = sorted(confusion_summary, key=lambda x: x["count"], reverse=True)[:15]

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    acc = accuracy_score(y_true, y_pred)

    metrics = {
        "dataset": {
            "train_path": str(TRAIN_PATH),
            "dev_path": str(DEV_PATH),
            "labels": labels,
            "num_dev_docs": len(docs),
        },
        "trigger_metrics": {
            "precision": round(float(macro_precision), 4),
            "recall": round(float(macro_recall), 4),
            "f1": round(float(macro_f1), 4),
        },
        "event_type_accuracy": round(float(acc), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_summary": confusion_summary,
    }

    return metrics


def main():
    random.seed(RANDOM_SEED)

    if not TRAIN_PATH.exists() or not DEV_PATH.exists():
        raise FileNotFoundError("Missing train.spacy or dev.spacy. Run prepare_spacy_data.py first.")

    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")

    train_docs = load_docs(TRAIN_PATH, nlp.vocab)
    dev_docs = load_docs(DEV_PATH, nlp.vocab)

    labels = sorted({label for doc in train_docs for label in doc.cats.keys()})

    for label in labels:
        textcat.add_label(label)

    train_examples = make_examples(nlp, train_docs)

    optimizer = nlp.initialize(lambda: train_examples)

    print("\nTraining spaCy text classifier")
    print(f"Train docs: {len(train_docs)}")
    print(f"Dev docs: {len(dev_docs)}")
    print(f"Labels: {labels}")

    for epoch in range(EPOCHS):
        random.shuffle(train_examples)
        losses = {}

        batches = spacy.util.minibatch(train_examples, size=BATCH_SIZE)
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Losses: {losses}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)

    metrics = evaluate_model(nlp, dev_docs, labels)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nModel saved to: {MODEL_DIR}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()