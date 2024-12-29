import asyncio

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from PIL import Image
import pandas as pd
import streamlit as st
from matplotlib.style.core import available

from utils import *

def predict_text_sync(payload):
    return asyncio.run(predict_text(payload))

def predict_corpus_sync(payload):
    return asyncio.run(predict_corpus(payload))


def get_model_list():
    return asyncio.run(get_list_of_models())

def load_model_sync(payload):
    return asyncio.run(load_model(payload))

def unload_model_sync():
    return asyncio.run(unload_model())

def fit_model_sync(payload):
    return asyncio.run(fit_model(payload))

def show_main_page():
    image = Image.open('streamlit/image/i_am_robot.jpeg')
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="AI Detection",
        page_icon=image,
    )

    st.title("Человек или машина?")
    st.image(image)
    file_markdown()

    st.header("Обучение модели")

    uploaded_file = st.file_uploader(
        "Загрузите файл для обучения(CSV):", type=["csv"]
    )
    if uploaded_file:
        df = validate_csv(uploaded_file)
        if df is not None:
            X, y = preprocess_df_sync(df)
            config = model_hyperparameters([X[0], X[-1]], [y[0], y[-1]])

            if st.button("Обучить модель"):
                if config:
                    st.write("Обучение модели...")
                    try:
                        result = fit_model_sync(config)
                        st.success("Модель успешно обучена!")
                        st.json(result)
                    except Exception as e:
                        st.error(f"Ошибка обучения: {str(e)}")

    st.header("Загрузка модели на инференс")

    selected_load_models = st.selectbox("Выберите модель для загрузки на инференс", get_model_list())
    if selected_load_models:
        if st.button(f"Загрузить модель {selected_load_models}"):
            load_model_sync(selected_load_models)
            st.success(f"Модель {selected_load_models} загружена")

    st.header("Выгрузка модели из инференса")

    if st.button(f"Выгрузить модели"):
        resp = unload_model_sync()
        st.success(resp)

    st.header("Предсказание текста")

    text_input = st.text_area("Введите текст для предсказания", height=200)

    if st.button("Предсказать"):
        if not text_input.strip():
            st.error("Введите текст для предсказания!")
        else:
            resp = predict_text_sync({"X":text_input})
            st.success(resp['prediction'])

    st.header("Предсказание на корпусе текстов")

    uploaded_file = st.file_uploader("Загрузите текстовый файл (.csv)", type=["csv"])
    if uploaded_file:
        texts = pd.read_csv(uploaded_file).tolist()
        st.write("Тексты для предсказания:")
        st.write(texts[:5])

        if st.button("Предсказать на корпусе текстов"):
            resp = predict_corpus_sync({"X": texts})
            st.success(resp)


def preprocess_df_sync(df):
    return asyncio.run(preprocess_df(df))


def model_hyperparameters(X, y):
    with st.sidebar:
        st.header("Model Hyperparameters")
        C = st.slider("C (Regularization Strength)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
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
    required_columns = pd.DataFrame({
        "Column Name": ["text", "label"],
        "Description": [
            "Текст для проверки",
            "Категория текста",
        ],
    })
    st.markdown("### Обязательные столбцы CSV файла:")
    st.table(required_columns)

    st.markdown("""
    - **Дополнительные столбцы** могут быть добавлены произвольно, например: `model`, `prompt` и т.д.
    """)


def validate_csv(file):
    try:
        df = pd.read_csv(file)
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


if __name__ == "__main__":
    show_main_page()
