import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import (
    EVENT_LABELS,
    MODEL_LOADED,
    extract_event_information,
    get_sample_articles,
    load_metrics,
)
from src.ui_helpers import metric_card_html, tag_html, code_block_html

st.set_page_config(
    page_title="Event Information Extraction from News Articles",
    page_icon="🗞️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


def inject_css():
    st.markdown(
        """
        <style>
        .main {padding-top: 1rem;}
        .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
        .soft-card {
            background: linear-gradient(180deg, rgba(33,37,41,0.95), rgba(24,27,31,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .section-title {font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem;}
        .muted {color: #b8c0cc; font-size: 0.95rem;}
        .event-card {
            background: linear-gradient(180deg, rgba(20,20,20,0.93), rgba(33,33,33,0.93));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
        }
        .event-title {font-size: 1.05rem; font-weight: 700; color: #f5f7fb; margin-bottom: 0.3rem;}
        .kv {font-size: 0.95rem; margin-bottom: 0.18rem;}
        .kv b {color: #e7ecf5;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.title("🗞️ Event Information Extraction from News Articles")
    st.caption(
        "Interactive NLP system for extracting event triggers, participants, dates, locations, and structured event records from news text."
    )


def render_use_model_tab():
    st.subheader("Use the model")

    if MODEL_LOADED:
        st.success("Trained spaCy model loaded successfully.")
    else:
        st.warning("No trained spaCy model found yet. Run convert_maven_to_csv.py, prepare_spacy_data.py and train_spacy_textcat.py first.")

    col1, col2 = st.columns([2.2, 1.1])
    with col2:
        st.markdown("#### Controls")
        sample_articles = get_sample_articles()
        selected_sample = st.selectbox("Load sample article", ["Custom text"] + list(sample_articles.keys()))
        show_steps = st.toggle("Show step-by-step pipeline", value=True)
        show_json = st.toggle("Show JSON output", value=True)
        event_filter = st.multiselect("Filter event types", EVENT_LABELS, default=[])
        confidence_threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.35, 0.05)

    with col1:
        if selected_sample != "Custom text":
            default_text = sample_articles[selected_sample]
        else:
            default_text = (
                "A fire broke out at a textile warehouse in Coimbatore on Tuesday evening. "
                "Four workers were injured and taken to the district hospital. "
                "Officials said the police opened an investigation into the incident."
            )

        article_text = st.text_area(
            "Paste a news article",
            value=default_text,
            height=250,
            placeholder="Enter a full news article here...",
        )

    if st.button("Extract event information", use_container_width=True):
        result = extract_event_information(article_text, confidence_threshold=confidence_threshold)
        events = result["events"]
        if event_filter:
            events = [e for e in events if e["event_type"] in event_filter]

        st.markdown("### Structured output")
        if not events:
            st.warning("No events passed the selected filter/threshold.")
        else:
            for idx, event in enumerate(events, 1):
                participants = ", ".join(event.get("participants", [])) or "—"
                st.markdown(
                    f"""
                    <div class="event-card">
                        <div class="event-title">Event {idx}: {event['event_type']}</div>
                        <div class="kv"><b>Trigger:</b> {event['trigger']}</div>
                        <div class="kv"><b>Sentence:</b> {event['sentence']}</div>
                        <div class="kv"><b>Participants:</b> {participants}</div>
                        <div class="kv"><b>Date/Time:</b> {event.get('date_time') or '—'}</div>
                        <div class="kv"><b>Location:</b> {event.get('location') or '—'}</div>
                        <div class="kv"><b>Confidence:</b> {event['confidence']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card_html("Detected events", len(events)), unsafe_allow_html=True)
        c2.markdown(metric_card_html("Sentences", result["stats"]["num_sentences"]), unsafe_allow_html=True)
        c3.markdown(metric_card_html("Entities", result["stats"]["num_entities"]), unsafe_allow_html=True)
        c4.markdown(metric_card_html("Dates found", result["stats"]["num_dates"]), unsafe_allow_html=True)

        if show_steps:
            st.markdown("### Pipeline steps")
            for step in result["steps"]:
                with st.expander(step["title"], expanded=False):
                    st.write(step["description"])
                    if step.get("dataframe") is not None:
                        st.dataframe(pd.DataFrame(step["dataframe"]), use_container_width=True)
                    if step.get("json") is not None:
                        st.json(step["json"])

        if show_json:
            st.markdown("### JSON output")
            st.json({"events": events, "stats": result["stats"]})


def render_how_it_works_tab():
    st.subheader("How it works / training information")
    st.markdown(
        """
        <div class="soft-card">
            <div class="section-title">Project design</div>
            <div class="muted">
            This app uses MAVEN event annotations converted into sentence-level labels, a trained spaCy text classifier for event type prediction,
            spaCy NER for entities, and lightweight post-processing for trigger, time, and location extraction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Core concepts covered")
        for item in [
            "Event extraction from news articles",
            "Sentence-level text classification",
            "spaCy textcat training",
            "Named Entity Recognition (NER)",
            "Date and location extraction",
            "Trigger identification",
            "Confusion analysis and evaluation metrics",
            "Human-annotated dataset conversion from MAVEN",
        ]:
            st.markdown(tag_html(item), unsafe_allow_html=True)

        st.markdown("#### Workflow")
        st.markdown(
            """
            1. Convert MAVEN JSONL into a reduced-label CSV dataset.  
            2. Convert CSV rows into train/dev `.spacy` files.  
            3. Train a spaCy `textcat` classifier on sentences.  
            4. Load the trained model in Streamlit.  
            5. Classify each sentence in a news article.  
            6. Use spaCy NER to collect participants, dates, and locations.  
            7. Assemble final event records in JSON form.
            """
        )

    with right:
        st.markdown("#### Training configuration example")
        config_example = {
            "dataset": "MAVEN reduced to 8 labels",
            "task": "Sentence-level event classification",
            "framework": "spaCy textcat",
            "epochs": 12,
            "batch_size": 8,
            "labels": EVENT_LABELS,
        }
        st.markdown(code_block_html(json.dumps(config_example, indent=2)), unsafe_allow_html=True)


def render_metrics_tab():
    st.subheader("Evaluation metrics")
    metrics = load_metrics(METRICS_PATH)

    top = st.columns(4)
    top[0].metric("Macro Precision", f"{metrics['trigger_metrics']['precision']:.3f}")
    top[1].metric("Macro Recall", f"{metrics['trigger_metrics']['recall']:.3f}")
    top[2].metric("Macro F1", f"{metrics['trigger_metrics']['f1']:.3f}")
    top[3].metric("Accuracy", f"{metrics['event_type_accuracy']:.3f}")

    if "dataset" in metrics:
        st.markdown("#### Dataset summary")
        st.json(metrics["dataset"])

    st.markdown("#### Per-class metrics")
    st.dataframe(pd.DataFrame(metrics.get("per_class_metrics", [])), use_container_width=True)

    st.markdown("#### Confusion summary")
    st.dataframe(pd.DataFrame(metrics.get("confusion_summary", [])), use_container_width=True)

    st.write("These metrics come from the real trained spaCy model saved in `artifacts/metrics.json`.")


def main():
    inject_css()
    render_header()
    tab1, tab2, tab3 = st.tabs(["🔎 Use the model", "🧠 How it works / training info", "📊 Evaluation metrics"])
    with tab1:
        render_use_model_tab()
    with tab2:
        render_how_it_works_tab()
    with tab3:
        render_metrics_tab()


if __name__ == "__main__":
    main()
