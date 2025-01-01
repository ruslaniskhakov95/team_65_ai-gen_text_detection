import asyncio

import streamlit as st

from streamlit_app.utils.transfromer_utils import (get_list_of_loaded_models,
                                                   get_list_of_models,
                                                   load_model, unload_model)


async def process_page():
    st.header("Загрузка модели трансформера на инференс")

    selected_load_models = st.selectbox(
        "Выберите модель для загрузки на инференс", await get_list_of_models()
    )
    if selected_load_models:
        if st.button(f"Загрузить модель {selected_load_models}"):
            await load_model(selected_load_models["id"])
            st.success(f"Модель {selected_load_models['id']} загружена")

    st.header("Выгрузка модели трансформера из инференса")

    selected_unload_models = st.selectbox(
        "Выберите модель для выгрузки из инференса",
        await get_list_of_loaded_models()
    )
    if st.button(f"Выгрузить модели"):
        resp = await unload_model(selected_unload_models["id"])
        st.success(resp)


if __name__ == "__main__":
    asyncio.run(process_page())
