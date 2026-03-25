import os

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    ARTIFACTS_DIR,
    EVAL_BATCH_SIZE,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
)
from src.data_loader import build_sentence_level_samples, load_jsonl
from src.utils import save_json, set_seed


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def encode_examples(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def train_model(train_path: str, valid_path: str, output_dir: str = str(ARTIFACTS_DIR)):
    set_seed(RANDOM_SEED)

    train_docs = load_jsonl(train_path)
    valid_docs = load_jsonl(valid_path)

    train_samples = build_sentence_level_samples(train_docs)
    valid_samples = build_sentence_level_samples(valid_docs)

    labels = sorted({sample["label"] for sample in (train_samples + valid_samples)})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    for sample in train_samples:
        sample["label_id"] = label2id[sample["label"]]
    for sample in valid_samples:
        sample["label_id"] = label2id[sample["label"]]

    os.makedirs(output_dir, exist_ok=True)
    save_json(
        {"label2id": label2id, "id2label": id2label},
        os.path.join(output_dir, "label_map.json"),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = Dataset.from_list(train_samples)
    valid_ds = Dataset.from_list(valid_samples)

    train_ds = train_ds.map(lambda batch: encode_examples(batch, tokenizer), batched=True)
    valid_ds = valid_ds.map(lambda batch: encode_examples(batch, tokenizer), batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    valid_ds = valid_ds.rename_column("label_id", "labels")

    keep_columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep_columns)
    valid_ds.set_format(type="torch", columns=keep_columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    return trainer.evaluate()


if __name__ == "__main__":
    metrics = train_model(
        train_path="data/maven/train.jsonl",
        valid_path="data/maven/valid.jsonl",
        output_dir="artifacts",
    )
    print(metrics)
