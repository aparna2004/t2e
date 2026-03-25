import streamlit as st

from src.predict import EventPredictor
from src.representation import build_event_card

st.set_page_config(page_title="Event Information Extraction", layout="wide")


@st.cache_resource
def load_predictor():
    return EventPredictor("artifacts")


st.title("Event Information Extraction from News Articles")
st.caption("MAVEN-based sentence event classification with 4W extraction and Streamlit UI")

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies event-bearing sentences using a model trained on MAVEN and extracts basic Who, Where, When, and What information using spaCy."
    )
    st.info("Before running the UI, train the model so artifacts/model, artifacts/tokenizer, and artifacts/label_map.json exist.")

sample_text = """A protest broke out in Chennai on Monday after students gathered near the university. Police later detained several organizers."""
text = st.text_area(
    "Paste a news article",
    value=sample_text,
    height=220,
    placeholder="Enter a news article here...",
)

if st.button("Extract Events", use_container_width=True):
    try:
        predictor = load_predictor()
        results = predictor.predict_document(text)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not results:
        st.warning("No event detected in the provided text.")
    else:
        st.success(f"Detected {len(results)} event(s).")
        for index, event in enumerate(results, start=1):
            card = build_event_card(event)
            with st.container(border=True):
                st.subheader(f"Event {index}")
                left, right = st.columns(2)
                with left:
                    st.write(f"**Event Type:** {card['event_type']}")
                    st.write(f"**Trigger:** {card['trigger'] or 'Not found'}")
                    st.write(f"**Confidence:** {card['confidence']}")
                with right:
                    st.write(f"**When:** {', '.join(card['when']) if card['when'] else 'Not found'}")
                    st.write(f"**Where:** {', '.join(card['where']) if card['where'] else 'Not found'}")
                    st.write(f"**Who:** {', '.join(card['who']) if card['who'] else 'Not found'}")
                st.write(f"**What:** {card['what']}")
