import asyncio
import string

import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from streamlit_app.utils.utils import fit_model


@st.cache_resource
def download_nltk_resources():
    """Download necessary NLTK resources."""
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")


download_nltk_resources()


async def process_page():
    """Main function for processing the page."""
    st.header("Model Training")

    display_file_requirements()

    uploaded_file = st.file_uploader(
        "Upload a file for training (CSV, json):",
        type=["csv", "jsonl", "json"]
    )
    if uploaded_file:
        df = validate_csv(uploaded_file)
        if df is not None:
            X = df["text"]
            y = df["label"].tolist()
            X = X.apply(
                lambda text: " ".join(
                    tokenize_and_clean_text(
                        " ".join(lemmatize(tokenize_and_clean_text(text)))
                    )
                )
            ).tolist()

            config = model_hyperparameters(X, y)
            if st.button("Train Model"):
                if config:
                    st.write("Training the model...")
                    try:
                        _, result = await fit_model(config)
                        st.success(result["message"])
                        train_sizes = result["train_sizes"]
                        train_scores_mean = result["train_scores_mean"]
                        train_scores_std = result["train_scores_std"]
                        test_scores_mean = result["test_scores_mean"]
                        test_scores_std = result["test_scores_std"]

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=train_sizes,
                                y=train_scores_mean,
                                mode="lines+markers",
                                name="Training Accuracy (Mean)",
                                line={"color": 'blue'},
                                error_y={
                                    "type": "data",
                                    "array": train_scores_std, "visible": True
                                },
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=train_sizes,
                                y=test_scores_mean,
                                mode="lines+markers",
                                name="Test Accuracy (Mean)",
                                line={"color": 'green'},
                                error_y={
                                    "type": "data",
                                    "array": test_scores_std, "visible": True
                                },
                            )
                        )
                        fig.update_layout(
                            title="Learning Curves",
                            xaxis_title="Training Set Size",
                            yaxis_title="Accuracy",
                            xaxis={"tickmode": "linear"},
                            yaxis={"range": [0, 1]},
                        )
                        st.plotly_chart(fig)

                    except ValueError as e:
                        st.error(f"Value error: {e}")

            display_eda(df['label'], df['text'])


def display_eda(labels, texts):
    """Display EDA."""

    st.subheader("Label Distribution")
    label_counts = labels.value_counts()
    fig_label_dist = px.bar(
        label_counts,
        x=label_counts.index,
        y=label_counts.values,
        labels={"x": "Label", "y": "Count"},
        title="Label Distribution",
    )
    st.plotly_chart(fig_label_dist)

    st.subheader("Text Length Distribution")
    text_lengths = texts.str.len()
    length_df = pd.DataFrame({"length": text_lengths, "label": labels})
    fig_text_len = px.histogram(
        length_df,
        x="length",
        color="label",
        nbins=50,
        labels={"length": "Text Length", "count": "Frequency", "label": "Label"},
        title="Text Length Distribution by Label",
    )
    st.plotly_chart(fig_text_len)

    st.subheader("Word Frequency Analysis")
    vectorizer = CountVectorizer(stop_words="english", max_features=50)
    word_counts = vectorizer.fit_transform(texts).toarray().sum(axis=0)
    words = vectorizer.get_feature_names_out()

    word_freq = pd.DataFrame({"word": words, "count": word_counts})
    fig_word_freq = px.bar(
        word_freq.sort_values("count", ascending=False),
        x="word",
        y="count",
        labels={"word": "Word", "count": "Frequency"},
        title="Word Frequency Analysis",
    )
    st.plotly_chart(fig_word_freq)


def model_hyperparameters(X, y):
    """Define and save model hyperparameters."""
    with st.sidebar:
        st.header("Model Hyperparameters")
        C = st.slider(
            "C (Regularization Strength)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
        )
        fit_intercept = st.checkbox("Fit Intercept", value=False)
        random_state = st.number_input("Random State",
                                       min_value=0, value=0, step=1)
        verbose = st.selectbox("Verbose", options=[0, 1, 2], index=0)

        model_type = st.selectbox("Model Type",
                                  options=["logistic", "svm"], index=0)

        st.header("Vectorization Parameters")
        vec_type = st.selectbox("Vectorization Type",
                                options=["bow", "tfidf"], index=0)
        max_features = st.number_input("Max Features",
                                       min_value=1, value=1, step=1)

        if st.button("Save Configuration"):
            st.session_state.config = {
                "X": X,
                "config": {
                    "id": "string1",
                    "hyperparams": {
                        "C": C,
                        "fit_intercept": fit_intercept,
                        "random_state": random_state,
                        "verbose": verbose,
                    },
                    "model_type": model_type,
                    "vec_type": vec_type,
                    "vec_params": {
                        "max_features": max_features,
                    },
                },
                "y": y,
            }
            st.success("Configuration saved!")

        return st.session_state.get("config", None)


def display_file_requirements():
    """Display required columns for the input file."""
    required_columns = pd.DataFrame(
        {
            "Column Name": ["text", "label"],
            "Description": [
                "Text for analysis",
                "Text category",
            ],
        }
    )
    st.markdown("### Required Columns in the CSV File:")
    st.table(required_columns)

    st.markdown(
        """
    - **Additional columns** can
     be added arbitrarily, e.g., `model`, `prompt`, etc.
    """
    )


def validate_csv(file):
    """Validate the uploaded CSV or JSONL file."""
    try:
        df = pd.DataFrame()
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
        required = {"text", "label"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            st.error(f"Error: Missing required columns: {', '.join(missing)}")
            return None
        st.success("File successfully loaded!")
        st.write("Data Preview:", df.head())
        return df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None


def tokenize_and_clean_text(text):
    """Tokenize and clean text by removing stop words and punctuation."""
    tokens = tokenize.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    punct_chars = (
        string.punctuation
        + r"'s"
        + r"'t"
        + r"n't"
        + r"'ll"
        + r"'re"
        + '""'
        + "..."
        + "'"
        + "``"
    )
    filtered_tokens = [
        w.lower()
        for w in tokens
        if w not in stop_words and w not in punct_chars and r"'" not in w
    ]
    return filtered_tokens


def lemmatize(tokens):
    """Lemmatize tokens."""
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


if __name__ == "__main__":
    asyncio.run(process_page())
