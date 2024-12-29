import streamlit as st
from streamlit_app.utils import fit_model
import asyncio
import pandas as pd

async def process_page():
    st.header("Обучение модели")

    file_markdown()

    uploaded_file = st.file_uploader(
        "Загрузите файл для обучения(CSV):", type=["csv"]
    )
    if uploaded_file:
        df = validate_csv(uploaded_file)
        if df is not None:
            X, y = await preprocess_df(df)
            config = model_hyperparameters([X[0], X[-1]], [y[0], y[-1]])

            if st.button("Обучить модель"):
                if config:
                    st.write("Обучение модели...")
                    try:
                        # result = fit_model_sync(config)
                        result = await fit_model(config)
                        st.success("Модель успешно обучена!")
                        st.json(result)
                    except Exception as e:
                        st.error(f"Ошибка обучения: {str(e)}")

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
    asyncio.run(process_page())