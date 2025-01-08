import asyncio
import pandas as pd
import streamlit as st
from streamlit_app.utils.utils import (
    predict_corpus, predict_response, predict_text
)
from pages import logger


async def process_page():
    """
    Main function to render the Streamlit page and handle text prediction.
    Includes single-text prediction and batch text prediction on uploaded
    files.
    """
    logger.info('Loading the predict page')
    st.header("Text Prediction")

    text_input = st.text_area("Enter text for prediction", height=200)

    if st.button("Predict"):
        if not text_input.strip():
            st.error("Please enter text for prediction!")
        else:
            logger.info("Making a prediction with basic model.")
            resp = await predict_text({"X": text_input})

            prediction_data = {}
            for key, value in predict_response.items():
                prediction_data.update({value: resp[1]["probability"][0][key]})

            prediction_df = pd.DataFrame(prediction_data, index=[0])

            st.write(prediction_df)

    st.header("Batch Text Prediction")

    st.markdown(
        """
        - The file must contain a column named `text`.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a file for prediction (.csv, .jsonl)", type=["csv", "jsonl"]
    )
    if uploaded_file:
        logger.info("Making a basic prediction with a file.")
        texts = pd.DataFrame()
        if uploaded_file.name.endswith(".csv"):
            texts = pd.read_csv(uploaded_file)["text"].tolist()
        elif uploaded_file.name.endswith(".jsonl"):
            texts = pd.read_json(uploaded_file, lines=True)["text"].tolist()

        st.write("Texts for prediction:")
        st.write(texts[:5])  # Display a sample of the texts

        if st.button("Predict on Batch of Texts"):
            resp = await predict_corpus({"X": texts})

            pred_data = []
            for i in range(len(texts)):
                pred_data.append({})
                for k, v in predict_response.items():
                    pred_data[i].update({v: resp[1]["probability"][i][k]})

            prediction_df = pd.DataFrame(
                pred_data, index=list(range(len(texts)))
            )

            st.write(prediction_df)


if __name__ == "__main__":
    asyncio.run(process_page())
