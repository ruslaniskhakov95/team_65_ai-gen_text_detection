import asyncio

import pandas as pd
import streamlit as st

from streamlit_app.utils.utils import (predict_corpus, predict_response,
                                       predict_text)


async def process_page():
    st.header("Предсказание текста")

    text_input = st.text_area("Введите текст для предсказания", height=200)

    if st.button("Предсказать"):
        if not text_input.strip():
            st.error("Введите текст для предсказания!")
        else:
            # resp = predict_text_sync({"X":text_input})
            resp = await predict_text({"X": text_input})

            prediction_data = {}
            for key, value in predict_response.items():
                prediction_data.update({value: resp[1]["probability"][0][key]})
            print(prediction_data)
            prediction_df = pd.DataFrame(prediction_data, index=[0])

            st.write(prediction_df)

    st.header("Предсказание на корпусе текстов")

    st.markdown(
        """
            - Файл обязательно должен содержать столбец `text`
            """
    )

    uploaded_file = st.file_uploader(
        "Загрузите файл для предсказания (.csv, .jsonl)", type=["csv", "jsonl"]
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            texts = pd.read_csv(uploaded_file)["text"].tolist()
        elif uploaded_file.name.endswith(".jsonl"):
            texts = pd.read_json(uploaded_file, lines=True)["text"].tolist()
        st.write("Тексты для предсказания:")
        st.write(texts[:5])

        if st.button("Предсказать на корпусе текстов"):
            resp = await predict_corpus({"X": texts})
            print(resp)

            prediction_data = []
            for i in range(len(texts)):
                prediction_data.append({})
                for key, value in predict_response.items():
                    prediction_data[i].update({value: resp[1]["probability"][i][key]})

            prediction_df = pd.DataFrame(
                prediction_data, index=[i for i in range(len(texts))]
            )

            st.write(prediction_df)


if __name__ == "__main__":
    asyncio.run(process_page())
