# Quick run instructions

1. Put MAVEN files here:

```text
data/maven_raw/train.jsonl
data/maven_raw/dev.jsonl
data/maven_raw/test.jsonl
```

2. Install packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Build dataset:

```bash
python src/convert_maven_to_csv.py
```

4. Prepare spaCy files:

```bash
python src/prepare_spacy_data.py
```

5. Train model:

```bash
python src/train_spacy_textcat.py
```

6. Run Streamlit:

```bash
streamlit run app.py
```
