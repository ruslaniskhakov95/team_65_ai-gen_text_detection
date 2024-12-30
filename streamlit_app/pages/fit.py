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

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


async def process_page():
    st.header("Обучение модели")

    file_markdown()

    uploaded_file = st.file_uploader(
        "Загрузите файл для обучения(CSV, json):", type=["csv", "jsonl", "json"]
    )
    if uploaded_file:
        df = validate_csv(uploaded_file)
        if df is not None:
            X = df["text"]
            y = df["label"].tolist()
            # cleaned_texts = []
            X = X.apply(
                lambda text: " ".join(
                    tokenize_and_clean_text(
                        " ".join(lemmatize(tokenize_and_clean_text(text)))
                    )
                )
            ).tolist()

            config = model_hyperparameters(X, y)

            if st.button("Обучить модель"):
                if config:
                    st.write("Обучение модели...")
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
                                name="Точность на обучении (средняя)",
                                line=dict(color="blue"),
                                error_y=dict(
                                    type="data", array=train_scores_std, visible=True
                                ),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=train_sizes,
                                y=test_scores_mean,
                                mode="lines+markers",
                                name="Точность на тесте (средняя)",
                                line=dict(color="green"),
                                error_y=dict(
                                    type="data", array=test_scores_std, visible=True
                                ),
                            )
                        )
                        fig.update_layout(
                            title="Кривые обучения",
                            xaxis_title="Размер обучающей выборки",
                            yaxis_title="Точность",
                            xaxis=dict(tickmode="linear"),
                            yaxis=dict(range=[0, 1]),
                        )
                        st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Ошибка обучения: {str(e)}")

            labels = df["label"]
            texts = df["text"]

            st.subheader("Распределение меток")
            label_counts = labels.value_counts()
            fig_label_dist = px.bar(
                label_counts,
                x=label_counts.index,
                y=label_counts.values,
                labels={"x": "Метка", "y": "Количество"},
                title="Распределение меток",
            )
            st.plotly_chart(fig_label_dist)

            st.subheader("Распределение длины текста")
            text_lengths = texts.str.len()
            length_df = pd.DataFrame({"length": text_lengths, "label": labels})
            fig_text_len = px.histogram(
                length_df,
                x="length",
                color="label",
                nbins=50,
                labels={"length": "Длина текста", "count": "Частота", "label": "Метка"},
                title="Распределение длины текста с учетом метки",
            )
            st.plotly_chart(fig_text_len)

            st.subheader("Частотный анализ слов")
            vectorizer = CountVectorizer(stop_words="english", max_features=50)
            word_counts = vectorizer.fit_transform(texts).toarray().sum(axis=0)
            words = vectorizer.get_feature_names_out()

            word_freq = pd.DataFrame({"word": words, "count": word_counts})
            fig_word_freq = px.bar(
                word_freq.sort_values("count", ascending=False),
                x="word",
                y="count",
                labels={"word": "Слово", "count": "Частота"},
                title="Частотный анализ слов",
            )
            st.plotly_chart(fig_word_freq)


def model_hyperparameters(X, y):
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
        random_state = st.number_input("Random State", min_value=0, value=0, step=1)
        verbose = st.selectbox("Verbose", options=[0, 1, 2], index=0)

        model_type = st.selectbox("Model type", options=["logistic", "svm"], index=0)

        st.header("Vectorization parameters")
        vec_type = st.selectbox("Vectorization type", options=["bow", "tfidf"], index=0)
        max_features = st.number_input("Max Features", min_value=1, value=1, step=1)

        if st.button("Save configuration"):
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
            st.success("Конфигурация сохранена!")

        return st.session_state.get("config", None)


def file_markdown():
    required_columns = pd.DataFrame(
        {
            "Column Name": ["text", "label"],
            "Description": [
                "Текст для проверки",
                "Категория текста",
            ],
        }
    )
    st.markdown("### Обязательные столбцы CSV файла:")
    st.table(required_columns)

    st.markdown(
        """
    - **Дополнительные столбцы** могут быть добавлены произвольно, например: `model`, `prompt` и т.д.
    """
    )


def validate_csv(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
        required = {"text", "label"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            st.error(f"Ошибка: Отсутствуют обязательные столбцы: {', '.join(missing)}")
            return None
        st.success("Файл успешно загружен!")
        st.write("Предпросмотр данных:", df.head())
        return df
    except Exception as e:
        st.error(f"Не удалось загрузить файл: {e}")
        return None


def tokenize_and_clean_text(text):
    tokens = tokenize.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    punct_chars = (
        string.punctuation
        + r"'s"
        + r"'t"
        + r"\n't"
        + r"'ll"
        + r"'re"
        + '""'
        + "..."
        + "'"
        + "``"
    )
    filtered_tokens = [
        word.lower()
        for word in tokens
        if word not in stop_words and word not in punct_chars and r"'" not in word
    ]
    return filtered_tokens


def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


if __name__ == "__main__":
    asyncio.run(process_page())
