import asyncio

import streamlit as st

from streamlit_app.utils.transfromer_utils import predict, get_list_of_loaded_models
import pandas as pd


async def process_page():
    """
    Main function to render the Streamlit page for batch text prediction.

    Features:
    - Allows users to upload a CSV file containing texts for prediction.
    - Displays a sample of uploaded texts.
    - Provides a dropdown to select the model type for prediction.
    - Performs batch predictions asynchronously and displays the results.
    """
    st.header("Batch Text Prediction")

    st.markdown(
        """
        - The file must contain a column named `text`.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a file for prediction (.csv)", type=["csv"]
    )
    if uploaded_file:
        texts = pd.read_csv(uploaded_file)["text"].tolist()
        st.write("Texts for prediction:")
        st.write(texts[:2])

        selected_model = st.selectbox(
            "Select model type for prediction",
            await get_list_of_loaded_models()
        )

        if st.button("Predict on Batch of Texts"):
            resp = await predict({"X": texts, "model_type": selected_model})
            st.success(list(resp['predictions']))


if __name__ == "__main__":
    asyncio.run(process_page())
