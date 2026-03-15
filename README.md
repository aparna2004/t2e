# Event Information Extraction from News Articles — spaCy + Streamlit

This project uses the MAVEN dataset, converts it into sentence-level labels, trains a spaCy `textcat` model, and serves predictions in a Streamlit app.

## Folder flow

- `data/maven_raw/` → put `train.jsonl`, `dev.jsonl`, `test.jsonl` here
- `src/convert_maven_to_csv.py` → creates `data/event_sentences.csv`
- `src/prepare_spacy_data.py` → creates `artifacts/spacy_data/train.spacy` and `dev.spacy`
- `src/train_spacy_textcat.py` → trains the model and writes `artifacts/spacy_event_model/` and `artifacts/metrics.json`
- `app.py` → runs the Streamlit interface

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run pipeline

```bash
python src/convert_maven_to_csv.py
python src/prepare_spacy_data.py
python src/train_spacy_textcat.py
streamlit run app.py
```

## Current approach

- Real human-annotated labels from MAVEN
- Reduced to a smaller set of event classes for a moderate NLP course project
- spaCy text classification for sentence-level event prediction
- spaCy NER for participants, dates, and locations
- Light post-processing for trigger selection and structured JSON output
