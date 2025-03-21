import asyncio

import streamlit as st
from pages import logger

from streamlit_app.utils.utils import (get_list_of_models, load_model,
                                       unload_model)


async def process_page():
    """
    Main function to handle model inference operations.

    This function provides a user interface to load and unload models
    for inference using Streamlit.
    """
    logger.info("Processing the inference page")
    st.header("Load Model for Inference")

    selected_load_model = st.selectbox(
        "Select a model to load for inference", await get_list_of_models()
    )

    if selected_load_model:
        logger.info("Loading model with id %s", selected_load_model)
        if st.button(f"Load Model: {selected_load_model}"):
            await load_model(selected_load_model)
            st.success(f"Model '{selected_load_model}' has been successfully loaded.")

    st.header("Unload Models from Inference")

    if st.button("Unload Models"):
        logger.info("Unloading active model")
        response = await unload_model()
        st.success(response)


if __name__ == "__main__":
    asyncio.run(process_page())
