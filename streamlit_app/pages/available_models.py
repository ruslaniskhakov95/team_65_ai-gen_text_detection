import asyncio

import streamlit as st
from pages import logger

from streamlit_app.utils.transfromer_utils import \
    get_list_of_models as get_list_of_transformer_models
from streamlit_app.utils.utils import get_list_of_models, get_model_info


async def process_page():
    logger.info("Processing the model info page")
    """
    Main function for processing the page.
    """
    st.header("Available Models and Hyperparameters")

    models = await get_list_of_models()

    if not models:
        st.warning("Not available models")

    table_data = []
    """Classic models"""
    for model in models:
        _, info = await get_model_info({"id": model["id"]})

        table_data.append(
            {
                "Model ID": model["id"],
                "Model Type": info["model_type"],
                "Model Hyperparams": info["hyperparams"],
                "Vectorizer Type": info["vec_type"],
                "Vectorizer Hyperparams": info["vec_params"],
            }
        )

    st.table(table_data)

    transformer_models = await get_list_of_transformer_models()
    transformer_table_data = []
    """Transformer models"""
    for model in transformer_models:
        transformer_table_data.append(
            {
                "Model Type": model,
            }
        )
    st.table(transformer_table_data)


if __name__ == "__main__":
    asyncio.run(process_page())
