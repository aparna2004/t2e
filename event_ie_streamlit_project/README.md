# Event Information Extraction from News Articles

This project modernizes a classic event extraction pipeline by combining:

- MAVEN for event-type supervision
- BERT for sentence-level event classification
- spaCy for basic 4W extraction
- Streamlit for the user interface

## Project structure

```text
event_ie_streamlit_project/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   └── maven/
│       ├── train.jsonl
│       ├── valid.jsonl
│       └── test.jsonl
├── artifacts/
├── pages/
└── src/
    ├── data_loader.py
    ├── predict.py
    ├── preprocess.py
    ├── representation.py
    ├── train.py
    └── utils.py
```

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Put your MAVEN files here:

```text
data/maven/train.jsonl
data/maven/valid.jsonl
data/maven/test.jsonl
```

## Train

```bash
python -m src.train
```

This creates:

- `artifacts/label_map.json`
- `artifacts/model/`
- `artifacts/tokenizer/`
- `artifacts/checkpoints/`

## Run the UI

```bash
streamlit run app.py
```

## Notes

- This version is sentence-level classification, not exact token-level trigger extraction.
- MAVEN is mainly strong for event detection and event typing.
- The Who, Where, and When fields are extracted using spaCy NER.
- The What field is the sentence that was classified as an event.

## Suggested future upgrades

- Token-level trigger detection using BERT token classification
- Confidence filtering slider in Streamlit
- Event distribution charts
- Better argument extraction than generic NER
