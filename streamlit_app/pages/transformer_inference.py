import asyncio

import streamlit as st

from streamlit_app.utils.transfromer_utils import (
    get_list_of_loaded_models,
    get_list_of_models,
    load_model,
    unload_model
)
from pages import logger


async def process_page():
    """
    Main function to handle model inference operations.
    This function provides a user interface to load and unload
    transformer models for inference using Streamlit.
    """
    logger.info('Loading the transformer inference page')
    st.header("Load Transformer Model for Inference")

    selected_load_models = st.selectbox(
        "Select a model to load for inference", await get_list_of_models()
    )
    if selected_load_models:
        if st.button(f"Load model {selected_load_models}"):
            logger.info("Loading transformer model")
            await load_model(selected_load_models)
            st.success(f"Model {selected_load_models} has been loaded")

    st.header("Unload Transformer Model from Inference")

    selected_unload_models = st.selectbox(
        'Selecct a model to unload from inference',
        await get_list_of_loaded_models()
    )
    if st.button(f"Unload model {selected_unload_models}"):
        logger.info("Unloading transformer model.")
        resp = await unload_model(selected_unload_models["id"])
        st.success(resp)

if __name__ == "__main__":
    asyncio.run(process_page())
