
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
import streamlit as st
import streamlit.components.v1 as components

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import (
    clean_text,
    tokenize_text,
    tokenize_sentences,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization,
    handle_negation,
    normalize_ml_text,
    preprocess_query,
    extract_domain_entities
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SupportIQ | Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

PALETTE = [
    "#A78BFA",
    "#38BDF8",
    "#F472B6",
    "#34D399",
    "#FBBF24",
    "#818CF8",
    "#FB7185"
]


# ==========================================
# GLASSMORPHISM STYLING
# ==========================================

st.markdown("""
<style>

:root {
    --text: #F5F1FF;
    --muted: #B6ACCF;
    --border: rgba(192,169,255,0.20);
}

.stApp {
    background:
        radial-gradient(
            ellipse at 10% 0%,
            #44216D 0%,
            transparent 40%
        ),
        radial-gradient(
            ellipse at 100% 30%,
            #292052 0%,
            transparent 42%
        ),
        #100D20;

    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg,
        #211638,
        #130F26
    );

    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #E8DFFF;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.block-container {
    max-width: 1450px;
    padding: 2rem 2.5rem 2rem;
}

h1, h2, h3, h4 {
    color: #F5F1FF !important;
    letter-spacing: -0.035em;
}

p, label, [data-testid="stWidgetLabel"] p {
    color: #E9E2F7 !important;
}

[data-testid="stCaptionContainer"] p {
    color: #B6ACCF !important;
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 20px;
}

[data-testid="stVerticalBlockBorderWrapper"] {

    background: linear-gradient(
        135deg,
        rgba(104,76,159,0.15),
        rgba(32,24,62,0.40)
    );

    border: 1px solid var(--border) !important;

    box-shadow: 0 12px 34px #08051025;

    backdrop-filter: blur(16px);
}

/* METRIC CARDS */

[data-testid="stMetric"] {

    background: linear-gradient(
        145deg,
        rgba(158,117,255,0.15),
        rgba(44,30,79,0.65)
    );

    border: 1px solid var(--border);

    border-radius: 18px;

    padding: 18px 20px;

    min-height: 105px;

    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #A78BFA;
    box-shadow: 0 8px 25px #8B5CF620;
}

[data-testid="stMetricLabel"] p {

    color: #C4B7DF !important;
    font-size: 13px !important;
}

[data-testid="stMetricValue"] {

    color: #F5F1FF !important;

    font-size: clamp(
        19px,
        2vw,
        30px
    ) !important;

    font-weight: 750;

    white-space: normal !important;

    overflow-wrap: anywhere;
}

[data-testid="stMetricValue"] div {

    white-space: normal !important;

    overflow-wrap: anywhere;
}

/* BUTTONS */

.stButton button {

    background: linear-gradient(
        105deg,
        #7C3AED,
        #A855F7
    );

    border: 1px solid #C4A1FF55;

    color: white !important;

    border-radius: 12px;

    min-height: 48px;

    font-weight: 650;

    box-shadow: 0 7px 22px #7C3AED40;

    transition: all 0.2s ease;
}

.stButton button:hover {

    transform: translateY(-2px);

    box-shadow: 0 10px 30px #A855F755;

    border-color: #D8B4FE;
}

/* INPUT FIELDS */

.stTextArea textarea,
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {

    background: #19142E !important;

    color: #F5F1FF !important;

    border-color: #65538B !important;

    border-radius: 12px !important;
}

.stTextArea textarea {

    font-size: 16px;

    line-height: 1.6;
}

/* SIDEBAR NAVIGATION */

[data-testid="stRadio"] label {

    padding: 0.4rem 0.3rem;

    border-radius: 8px;
}

[data-testid="stRadio"] label:hover {

    background: rgba(
        167,139,250,0.12
    );
}

/* TABS */

[data-baseweb="tab-list"] {
    gap: 12px;
}

[data-baseweb="tab"] {

    color: #D8C9F8 !important;

    font-weight: 600;
}

/* DATAFRAMES */

[data-testid="stDataFrame"] {

    border-radius: 12px;

    overflow: hidden;
}

/* HERO HEADER */

.hero {

    position: relative;

    overflow: hidden;

    padding: 28px 32px;

    margin-bottom: 20px;

    border: 1px solid var(--border);

    border-radius: 24px;

    background: linear-gradient(
        115deg,
        rgba(140,83,222,0.25),
        rgba(44,30,82,0.30)
    );

    box-shadow: 0 16px 60px #08051025;
}

.hero::after {

    content: "";

    position: absolute;

    right: -70px;

    top: -90px;

    width: 240px;

    height: 240px;

    border-radius: 50%;

    background: radial-gradient(
        circle,
        #A855F730,
        transparent 70%
    );

    pointer-events: none;
}

.eyebrow {

    font-size: 11px;

    letter-spacing: 2.5px;

    font-weight: 750;

    color: #C4A1FF;

    margin-bottom: 12px;
}

.hero h1 {

    font-size: clamp(
        28px,
        3.2vw,
        44px
    );

    margin: 0 0 12px;

    padding: 0;

    line-height: 1.15;
}

.hero p {

    font-size: 15px;

    margin: 0;

    color: #D1C5E8 !important;
}

/* BRANDING */

.brand {

    font-size: 29px;

    font-weight: 800;

    letter-spacing: -1.3px;

    color: #FFFFFF;
}

.brand-sub {

    font-size: 10px;

    letter-spacing: 2px;

    color: #BCA8E9;

    margin: 5px 0 16px;
}

.status {

    font-size: 12px;

    color: #86EFAC !important;

    font-weight: 600;

    letter-spacing: 0.5px;
}

/* KEYWORD CHIPS */

.chip {

    display: inline-block;

    padding: 7px 13px;

    margin: 4px;

    border-radius: 100px;

    border: 1px solid #AA84F35C;

    background: #A78BFA20;

    color: #EADCFF;

    font-size: 13px;
}

/* FOOTER */

.footer {

    text-align: center;

    color: #A99BC7;

    font-size: 12px;

    margin: 25px 0 5px;
}

a {
    color: #C4B5FD !important;
}

/* RESPONSIVE DESIGN */

@media(max-width: 750px) {

    .block-container {

        padding: 1.2rem 1rem;
    }

    .hero {
        padding: 20px;
    }

    .hero h1 {
        font-size: 28px;
    }

}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD PROJECT FILES
# ==========================================

@st.cache_resource
def load_models():

    classifier = joblib.load(
        MODELS / "intent_classifier.pkl"
    )

    category_map = joblib.load(
        MODELS / "intent_category_map.pkl"
    )

    keyword_model = joblib.load(
        MODELS / "keyword_vectorizer.pkl"
    )

    metadata = joblib.load(
        MODELS / "model_metadata.pkl"
    )

    return (
        classifier,
        category_map,
        keyword_model,
        metadata
    )


@st.cache_data
def load_dataset():

    return pd.read_csv(
        ROOT / "data" / "customer_support.csv"
    )


@st.cache_resource
def load_nlp():

    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_sentiment():

    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_emotion():

    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1
    )


try:

    classifier, category_map, keyword_model, metadata = (
        load_models()
    )

    df = load_dataset()

except Exception as error:

    st.error(
        f"Unable to load project files: {error}"
    )

    st.stop()


# ==========================================
# REUSABLE UI COMPONENTS
# ==========================================

# FIXED HTML RENDERING

def hero(kicker, title, subtitle):

    st.markdown(
        f'<div class="hero">'
        f'<div class="eyebrow">{html.escape(kicker)}</div>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p>'
        f'</div>',
        unsafe_allow_html=True
    )


def chart(fig, height=350):

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(

            color="#E8DFFF",

            size=13
        ),

        margin=dict(

            l=12,

            r=24,

            t=25,

            b=25
        ),

        height=height,

        legend=dict(

            orientation="h"
        )
    )

    fig.update_xaxes(

        gridcolor="rgba(192,169,255,0.12)",

        zerolinecolor="rgba(192,169,255,0.18)",

        automargin=True
    )

    fig.update_yaxes(

        gridcolor="rgba(192,169,255,0.12)",

        zerolinecolor="rgba(192,169,255,0.18)",

        automargin=True
    )

    st.plotly_chart(

        fig,

        use_container_width=True,

        config={

            "displaylogo": False,

            "responsive": True
        }
    )


def table(data):

    st.dataframe(

        data,

        hide_index=True,

        use_container_width=True
    )


def show_chips(values):

    content = "".join(

        f'<span class="chip">'
        f'{html.escape(str(value))}'
        f'</span>'

        for value in values
    )

    st.markdown(

        content or "No results found.",

        unsafe_allow_html=True
    )


# ==========================================
# NLP AND ML FUNCTIONS
# ==========================================

def predict(text):

    normalized = normalize_ml_text(text)

    if not normalized:

        return (

            "Unknown",

            "Unknown",

            0.0,

            pd.DataFrame(

                columns=[
                    "Intent",
                    "Probability"
                ]
            )
        )

    probabilities = classifier.predict_proba(
        [normalized]
    )[0]

    top_indices = np.argsort(
        -probabilities
    )[:3]

    intent = str(
        classifier.predict([normalized])[0]
    )

    top_predictions = pd.DataFrame({

        "Intent": [

            str(classifier.classes_[i])
            .replace("_", " ")
            .title()

            for i in top_indices
        ],

        "Probability": probabilities[top_indices]
    })

    return (

        category_map.get(intent, "Unknown"),

        intent,

        float(probabilities.max()),

        top_predictions
    )


def sentiment(text):

    scores = load_sentiment().polarity_scores(
        text
    )

    compound = scores["compound"]

    if compound >= 0.05:

        label = "Positive"

    elif compound <= -0.05:

        label = "Negative"

    else:

        label = "Neutral"

    return label, scores


def keywords(text, top_n=5):

    processed = preprocess_query(text)

    vector = keyword_model.transform(
        [processed]
    )

    names = keyword_model.get_feature_names_out()

    scores = sorted(

        zip(vector.indices, vector.data),

        key=lambda item: item[1],

        reverse=True

    )[:top_n]

    return pd.DataFrame(

        [

            (
                names[index],

                round(float(score), 4)
            )

            for index, score in scores
        ],

        columns=[

            "Keyword",

            "TF-IDF Score"
        ]
    )


def show_keywords(text):

    result = keywords(text)

    if result.empty:

        st.info("No matching keywords found.")

        return

    show_chips(
        result["Keyword"].tolist()
    )

    fig = px.bar(

        result.sort_values("TF-IDF Score"),

        x="TF-IDF Score",

        y="Keyword",

        orientation="h",

        color_discrete_sequence=[
            "#A78BFA"
        ],

        text_auto=".3f"
    )

    chart(fig, 260)


def summarize(text, n=2):

    sentences = tokenize_sentences(text)

    if len(sentences) <= n:

        return " ".join(sentences)

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            sentences
        )

        scores = np.asarray(
            matrix.sum(axis=1)
        ).ravel()

        selected = sorted(
            np.argsort(-scores)[:n]
        )

        return " ".join(

            sentences[i]

            for i in selected
        )

    except ValueError:

        return " ".join(sentences[:n])


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(

        '<div class="brand">✦ SupportIQ</div>'
        '<div class="brand-sub">'
        'CUSTOMER INTELLIGENCE'
        '</div>'
        '<div class="status">'
        '● MODEL LOADED'
        '</div>',

        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(

        "WORKSPACE",

        [

            "Query Intelligence",

            "NLP Analysis",

            "Dataset Insights",

            "Model Performance"
        ]
    )

    st.divider()

    st.caption("MODEL ARCHITECTURE")

    st.write(metadata["model_name"])

    st.caption(
        "TF-IDF · 27 intents · 11 categories"
    )

    st.divider()

    with st.expander("Project Technologies"):

        st.markdown("""

- Python
- Natural Language Processing
- Scikit-learn
- NLTK
- spaCy
- Transformers
- PyTorch
- Streamlit
- Plotly

        """)

    st.divider()

    st.caption("PORTFOLIO PROJECT")

    st.write("Dhruv Panchal")

    st.markdown(

        "[LinkedIn]"
        "(https://www.linkedin.com/in/dhruv-panchal-a14865258/)"
    )

    st.markdown(

        "[GitHub]"
        "(https://github.com/dhruv4AlgoForge)"
    )


# ==========================================
# PAGE 1 — QUERY INTELLIGENCE
# ==========================================

if page == "Query Intelligence":

    hero(

        "01 / INTELLIGENCE WORKSPACE",

        "Understand every conversation.",

        "Transform customer messages into structured "
        "intent, sentiment and linguistic insights."
    )

    a, b, c, d = st.columns(4)

    a.metric(

        "Source Queries",

        f"{metadata['total_records']:,}"
    )

    b.metric(

        "Test Accuracy",

        f"{metadata['test_accuracy']:.2%}"
    )

    c.metric(

        "Supported Intents",

        metadata["total_intents"]
    )

    d.metric(

        "Support Categories",

        metadata["total_categories"]
    )

    st.write("")

    with st.container(border=True):

        st.subheader("✦ Analyze a Conversation")

        st.caption(

            "Enter a customer query to discover "
            "its intent and language characteristics."
        )

        examples = [

            "Custom query",

            "I forgot my account password.",

            "Please cancel my order immediately.",

            "My payment failed but money was deducted.",

            "I want my money back.",

            "How can I speak with a human agent?",

            "I need to change my shipping address.",

            "I am extremely disappointed with your service."
        ]

        if "query_input" not in st.session_state:

            st.session_state.query_input = ""

        def choose_example():

            selected = st.session_state.query_example

            st.session_state.query_input = (

                ""

                if selected == "Custom query"

                else selected
            )

        st.selectbox(

            "Quick Examples",

            examples,

            key="query_example",

            on_change=choose_example
        )

        query = st.text_area(

            "Customer Message",

            key="query_input",

            height=125,

            placeholder=(
                "Enter a customer support request..."
            )
        )

        emotion_enabled = st.toggle(

            "Enable Transformer Emotion Analysis",

            value=False
        )

        if st.button(

            "✦ Analyze Conversation",

            use_container_width=True
        ):

            if query.strip():

                st.session_state.analysis_query = query

                st.session_state.analysis_emotion = (
                    emotion_enabled
                )

            else:

                st.session_state.pop(
                    "analysis_query",
                    None
                )

                st.warning(
                    "Please enter a customer query."
                )


    # ======================================
    # ANALYSIS RESULTS
    # ======================================

    if "analysis_query" in st.session_state:

        query = st.session_state.analysis_query

        emotion_enabled = (
            st.session_state.analysis_emotion
        )

        category, intent, confidence, top = predict(
            query
        )

        sentiment_label, scores = sentiment(
            query
        )

        st.write("")

        st.subheader("Prediction Overview")

        st.caption(
            "Analyzed message: " + query
        )

        a, b, c = st.columns(3)

        a.metric(

            "Predicted Category",

            category
        )

        b.metric(

            "Predicted Intent",

            intent.replace("_", " ").title()
        )

        c.metric(

            "Model Confidence",

            f"{confidence:.2%}"
        )

        if confidence >= 0.70:

            st.success(
                "High-confidence prediction."
            )

            confidence_color = "#34D399"

        elif confidence >= 0.40:

            st.warning(

                "Moderate confidence. "
                "Prediction requires review."
            )

            confidence_color = "#FBBF24"

        else:

            st.error(

                "Low confidence. Please clarify "
                "or rephrase the customer query."
            )

            confidence_color = "#FB7185"


        # CONFIDENCE AND SENTIMENT

        left, right = st.columns(2)

        with left:

            with st.container(border=True):

                st.subheader("Confidence Profile")

                gauge = go.Figure(

                    go.Indicator(

                        mode="gauge+number",

                        value=confidence * 100,

                        number={

                            "suffix": "%",

                            "valueformat": ".2f"
                        },

                        gauge={

                            "axis": {

                                "range": [0, 100]
                            },

                            "bar": {

                                "color": confidence_color
                            },

                            "bgcolor": "#31234E",

                            "borderwidth": 0,

                            "steps": [

                                {

                                    "range": [0, 40],

                                    "color": "#49243C"
                                },

                                {

                                    "range": [40, 70],

                                    "color": "#4C3B31"
                                },

                                {

                                    "range": [70, 100],

                                    "color": "#203B43"
                                }
                            ]
                        }
                    )
                )

                chart(gauge, 250)

                st.markdown(
                    "**Top 3 Predicted Intents**"
                )

                if not top.empty:

                    fig = px.bar(

                        top.sort_values("Probability"),

                        x="Probability",

                        y="Intent",

                        orientation="h",

                        text_auto=".1%",

                        color_discrete_sequence=[

                            "#A78BFA"
                        ]
                    )

                    fig.update_xaxes(

                        tickformat=".0%",

                        range=[0, 1.15]
                    )

                    chart(fig, 250)

        with right:

            with st.container(border=True):

                st.subheader("Sentiment Spectrum")

                st.metric(

                    "Predicted Sentiment",

                    sentiment_label
                )

                st.metric(

                    "Compound Score",

                    f"{scores['compound']:+.4f}"
                )

                sentiment_data = pd.DataFrame({

                    "Label": [

                        "Positive",

                        "Neutral",

                        "Negative"
                    ],

                    "Score": [

                        scores["pos"],

                        scores["neu"],

                        scores["neg"]
                    ]
                })

                if sentiment_data["Score"].sum() > 0:

                    fig = px.pie(

                        sentiment_data,

                        names="Label",

                        values="Score",

                        hole=0.72,

                        color="Label",

                        color_discrete_map={

                            "Positive": "#34D399",

                            "Neutral": "#A78BFA",

                            "Negative": "#FB7185"
                        }
                    )

                    fig.update_traces(

                        textposition="inside",

                        textinfo="percent"
                    )

                    chart(fig, 310)

                else:

                    st.info(
                        "No sentiment-bearing text detected."
                    )


        # KEYWORDS AND EMOTIONS

        left, right = st.columns(2)

        with left:

            with st.container(border=True):

                st.subheader("Key Phrases")

                show_keywords(query)

        with right:

            with st.container(border=True):

                st.subheader("Emotion Intelligence")

                if emotion_enabled:

                    try:

                        with st.spinner(

                            "Analyzing customer emotion..."
                        ):

                            result = load_emotion()(

                                query,

                                truncation=True,

                                max_length=128

                            )[0]

                        st.metric(

                            "Predicted Emotion",

                            result["label"].title()
                        )

                        st.metric(

                            "Emotion Confidence",

                            f"{result['score']:.2%}"
                        )

                        st.progress(

                            float(result["score"])
                        )

                        st.caption(

                            "Powered by pretrained DistilRoBERTa"
                        )

                    except Exception as error:

                        st.warning(

                            "Emotion analysis unavailable: "

                            + str(error)
                        )

                else:

                    st.info(

                        "Enable Transformer Emotion Analysis "
                        "to identify seven emotion categories."
                    )

                    show_chips([

                        "Anger",

                        "Sadness",

                        "Joy",

                        "Fear",

                        "Surprise",

                        "Disgust",

                        "Neutral"
                    ])


        # DOMAIN ENTITY EXTRACTION

        with st.container(border=True):

            st.subheader(
                "Structured Customer Entities"
            )

            entities = extract_domain_entities(
                query
            )

            if entities:

                table(

                    pd.DataFrame(

                        entities,

                        columns=[

                            "Entity",

                            "Entity Type"
                        ]
                    )
                )

            else:

                st.info(

                    "No structured identifiers detected."
                )

        st.caption(

            "Prediction confidence is not guaranteed "
            "correctness. Confidence thresholds are "
            "illustrative and not statistically calibrated."
        )


# ==========================================
# PAGE 2 — NLP ANALYSIS
# ==========================================

elif page == "NLP Analysis":

    hero(

        "02 / LANGUAGE LAB",

        "Explore language, layer by layer.",

        "Inspect lexical, grammatical and statistical "
        "Natural Language Processing transformations."
    )

    with st.container(border=True):

        text = st.text_area(

            "Text to Analyze",

            value=(

                "I cannot access my account. "

                "Please help me recover my password."
            ),

            height=140
        )

        if st.button(

            "✦ Run NLP Analysis",

            use_container_width=True
        ):

            if text.strip():

                st.session_state.lab_text = text

            else:

                st.session_state.pop(

                    "lab_text",

                    None
                )

                st.warning(
                    "Enter some text first."
                )

    if "lab_text" in st.session_state:

        text = st.session_state.lab_text

        st.caption(
            "Results for: " + text
        )

        cleaned = clean_text(text)

        tokens = tokenize_text(text)

        filtered = remove_stopwords(tokens)

        stems = apply_stemming(filtered)

        lemmas = remove_stopwords(

            apply_lemmatization(tokens)
        )

        negations = handle_negation(lemmas)

        a, b, c = st.columns(3)

        a.metric(

            "Word Tokens",

            len(tokens)
        )

        b.metric(

            "Filtered Tokens",

            len(filtered)
        )

        c.metric(

            "Removed Tokens",

            len(tokens) - len(filtered)
        )

        st.write("")

        tabs = st.tabs([

            "Preprocessing",

            "POS & NER",

            "Dependency Parsing",

            "Keywords",

            "Summarization"
        ])


        # TAB 1 — PREPROCESSING

        with tabs[0]:

            with st.container(border=True):

                st.subheader("Normalized Text")

                st.code(

                    cleaned,

                    language=None
                )

                preprocessing_results = [

                    (

                        "Word Tokenization",

                        tokens
                    ),

                    (

                        "Sentence Tokenization",

                        tokenize_sentences(text)
                    ),

                    (

                        "Stopword Removal",

                        filtered
                    ),

                    (

                        "Stemming",

                        stems
                    ),

                    (

                        "Lemmatization",

                        lemmas
                    ),

                    (

                        "Negation Handling",

                        negations
                    )
                ]

                for title, values in preprocessing_results:

                    st.markdown(
                        "**" + title + "**"
                    )

                    show_chips(values)

                    st.write("")


        # TAB 2 — POS AND NER

        with tabs[1]:

            st.subheader(
                "Part-of-Speech Tagging"
            )

            tags = pos_tag(
                word_tokenize(text)
            )

            table(

                pd.DataFrame(

                    tags,

                    columns=[

                        "Word",

                        "POS Tag"
                    ]
                )
            )

            doc = load_nlp()(text)

            st.subheader(
                "Named Entity Recognition"
            )

            if doc.ents:

                table(

                    pd.DataFrame(

                        [

                            (ent.text, ent.label_)

                            for ent in doc.ents
                        ],

                        columns=[

                            "Entity",

                            "Entity Type"
                        ]
                    )
                )

            else:

                st.info(

                    "No general-purpose named entities detected."
                )

            st.subheader(
                "Domain-Specific Entities"
            )

            domain_entities = extract_domain_entities(
                text
            )

            if domain_entities:

                table(

                    pd.DataFrame(

                        domain_entities,

                        columns=[
                            "Entity",
                            "Type"
                        ]
                    )
                )

            else:

                st.info(
                    "No structured entities detected."
                )

            st.subheader(
                "Noun Phrase Extraction"
            )

            show_chips([

                chunk.text

                for chunk in doc.noun_chunks
            ])


        # TAB 3 — DEPENDENCY PARSING

        with tabs[2]:

            st.subheader(
                "Syntactic Relationships"
            )

            doc = load_nlp()(text)

            dependency_data = [

                (

                    token.text,

                    token.pos_,

                    token.dep_,

                    token.head.text
                )

                for token in doc
            ]

            table(

                pd.DataFrame(

                    dependency_data,

                    columns=[

                        "Word",

                        "POS",

                        "Dependency",

                        "Head"
                    ]
                )
            )

            from spacy import displacy

            rendered = displacy.render(

                doc,

                style="dep",

                jupyter=False,

                options={

                    "compact": True,

                    "distance": 115,

                    "bg": "#19142E",

                    "color": "#E8DFFF",

                    "font": "Arial"
                }
            )

            components.html(

                '<div style="background:#19142E;'
                'padding:16px;overflow:auto">'

                + rendered +

                '</div>',

                height=350,

                scrolling=True
            )


        # TAB 4 — KEYWORDS

        with tabs[3]:

            st.subheader(
                "TF-IDF Keyword Extraction"
            )

            show_keywords(text)


        # TAB 5 — SUMMARIZATION

        with tabs[4]:

            st.subheader(
                "Extractive Text Summary"
            )

            summary = summarize(text)

            with st.container(border=True):

                st.write(summary)

            original = len(text.split())

            shortened = len(summary.split())

            reduction = (

                1 - shortened / max(original, 1)

            ) * 100

            a, b, c = st.columns(3)

            a.metric(

                "Original Words",

                original
            )

            b.metric(

                "Summary Words",

                shortened
            )

            c.metric(

                "Word Reduction",

                f"{reduction:.2f}%"
            )

            st.caption(

                "Compression is not a measure of "
                "summary quality. Text containing at most "
                "two sentences is retained."
            )


# ==========================================
# PAGE 3 — DATASET INSIGHTS
# ==========================================

elif page == "Dataset Insights":

    hero(

        "03 / DATA EXPLORER",

        "A closer look at customer demand.",

        "Explore the customer support dataset "
        "and its intent distribution."
    )

    a, b, c, d = st.columns(4)

    a.metric(

        "Source Records",

        f"{len(df):,}"
    )

    b.metric(

        "Categories",

        df["category"].nunique()
    )

    c.metric(

        "Intents",

        df["intent"].nunique()
    )

    d.metric(

        "Unique Raw Queries",

        f"{df['instruction'].nunique():,}"
    )

    st.write("")


    # CATEGORY DISTRIBUTION

    with st.container(border=True):

        st.subheader(
            "Support Category Distribution"
        )

        counts = (

            df["category"]

            .value_counts()

            .rename_axis("Category")

            .reset_index(name="Queries")
        )

        fig = px.bar(

            counts.sort_values("Queries"),

            x="Queries",

            y="Category",

            orientation="h",

            color="Queries",

            color_continuous_scale=[

                "#50327D",

                "#C4A1FF"
            ],

            text_auto=",d"
        )

        fig.update_layout(

            coloraxis_showscale=False
        )

        chart(fig, 470)


    # INTENT DISTRIBUTION

    with st.container(border=True):

        st.subheader(
            "Customer Intent Distribution"
        )

        counts = (

            df["intent"]

            .value_counts()

            .rename_axis("Intent")

            .reset_index(name="Queries")
        )

        fig = px.bar(

            counts.sort_values("Queries"),

            x="Queries",

            y="Intent",

            orientation="h",

            color="Queries",

            color_continuous_scale=[

                "#49347B",

                "#38BDF8"
            ],

            text_auto=",d"
        )

        fig.update_layout(

            coloraxis_showscale=False
        )

        chart(fig, 850)


    # DATASET EXPLORER

    with st.container(border=True):

        st.subheader(
            "Explore Source Records"
        )

        selected_category = st.selectbox(

            "Filter by Category",

            ["All"] + sorted(

                df["category"].unique().tolist()
            )
        )

        search_text = st.text_input(

            "Search Customer Queries",

            placeholder="Example: refund, order, password"
        )

        if selected_category == "All":

            view = df.copy()

        else:

            view = df[

                df["category"] == selected_category

            ].copy()

        if search_text.strip():

            view = view[

                view["instruction"].str.contains(

                    search_text,

                    case=False,

                    regex=False,

                    na=False
                )
            ]

        table(

            view[

                [

                    "instruction",

                    "category",

                    "intent"
                ]

            ].head(100)
        )

        st.caption(

            f"Showing up to 100 of "
            f"{len(view):,} matching records."
        )

        st.caption(

            "The source dataset contains synthetic "
            "customer queries and response templates."
        )


# ==========================================
# PAGE 4 — MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    hero(

        "04 / MODEL EVALUATION",

        "Performance, with perspective.",

        "Held-out evaluation of the selected TF-IDF "
        "and Logistic Regression pipeline."
    )

    a, b, c, d = st.columns(4)

    a.metric(

        "Test Accuracy",

        f"{metadata['test_accuracy']:.2%}"
    )

    b.metric(

        "Macro Precision",

        f"{metadata['test_precision']:.2%}"
    )

    c.metric(

        "Macro Recall",

        f"{metadata['test_recall']:.2%}"
    )

    d.metric(

        "Macro F1 Score",

        f"{metadata['test_f1']:.2%}"
    )

    a, b, c = st.columns(3)

    a.metric(

        "Training Records",

        f"{metadata['training_records']:,}"
    )

    b.metric(

        "Testing Records",

        f"{metadata['testing_records']:,}"
    )

    c.metric(

        "Category Accuracy",

        f"{metadata['category_accuracy']:.2%}"
    )

    st.write("")

    left, right = st.columns(2)


    # EVALUATION METRICS

    with left:

        with st.container(border=True):

            st.subheader(
                "Held-Out Evaluation Metrics"
            )

            metrics = pd.DataFrame({

                "Metric": [

                    "Accuracy",

                    "Macro Precision",

                    "Macro Recall",

                    "Macro F1"
                ],

                "Score": [

                    metadata["test_accuracy"],

                    metadata["test_precision"],

                    metadata["test_recall"],

                    metadata["test_f1"]
                ]
            })

            fig = px.bar(

                metrics,

                x="Metric",

                y="Score",

                color="Metric",

                color_discrete_sequence=PALETTE,

                text_auto=".2%"
            )

            fig.update_layout(

                showlegend=False
            )

            fig.update_yaxes(

                range=[0, 1.12],

                tickformat=".0%"
            )

            chart(fig, 390)


    # PREDICTION OUTCOMES

    with right:

        with st.container(border=True):

            st.subheader(
                "Prediction Outcomes"
            )

            results = pd.DataFrame({

                "Outcome": [

                    "Correct",

                    "Incorrect"
                ],

                "Count": [

                    metadata["correct_predictions"],

                    metadata["incorrect_predictions"]
                ]
            })

            fig = px.pie(

                results,

                names="Outcome",

                values="Count",

                hole=0.76,

                color="Outcome",

                color_discrete_map={

                    "Correct": "#34D399",

                    "Incorrect": "#FB7185"
                }
            )

            fig.add_annotation(

                text=f"{metadata['test_accuracy']:.2%}",

                x=0.5,

                y=0.5,

                showarrow=False,

                font=dict(

                    size=30,

                    color="#F5F1FF"
                )
            )

            chart(fig, 300)

            a, b = st.columns(2)

            a.metric(

                "Correct Predictions",

                f"{metadata['correct_predictions']:,}"
            )

            b.metric(

                "Incorrect Predictions",

                metadata["incorrect_predictions"]
            )


    # TRAINING DISTRIBUTION

    with st.container(border=True):

        st.subheader(
            "Training & Testing Distribution"
        )

        split_data = pd.DataFrame({

            "Dataset": [

                "Training",

                "Testing"
            ],

            "Records": [

                metadata["training_records"],

                metadata["testing_records"]
            ]
        })

        fig = px.pie(

            split_data,

            names="Dataset",

            values="Records",

            hole=0.65,

            color_discrete_sequence=[

                "#A78BFA",

                "#38BDF8"
            ]
        )

        chart(fig, 330)


    # METHODOLOGY

    with st.container(border=True):

        st.subheader(
            "Evaluation Methodology"
        )

        st.markdown("""

- Text normalization and duplicate removal
- Stratified 80:20 train-test split
- TF-IDF unigram and bigram features
- Five-fold stratified cross-validation
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
- Model selection using macro F1-score
- Held-out test evaluation
- Confusion matrix and misclassification analysis

        """)

        st.warning(

            "The model was evaluated on a synthetic "
            "customer support dataset. Real-world "
            "performance may differ due to spelling "
            "mistakes, ambiguous queries and unfamiliar intents."
        )


# ==========================================
# FOOTER — FIXED HTML RENDERING
# ==========================================

st.markdown(

    '<div class="footer">'
    '✦ SupportIQ &nbsp; / &nbsp; '
    'NLP & Machine Learning Portfolio'
    ' &nbsp; / &nbsp; v1.0'
    '</div>',

    unsafe_allow_html=True
)
