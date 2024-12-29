import streamlit as st
from streamlit_app.utils import *
import asyncio

async def process_page():
    st.header("Загрузка модели на инференс")

    selected_load_models = st.selectbox("Выберите модель для загрузки на инференс", await get_list_of_models())
    if selected_load_models:
        if st.button(f"Загрузить модель {selected_load_models}"):
            # load_model_sync(selected_load_models)
            await load_model(selected_load_models)
            st.success(f"Модель {selected_load_models} загружена")

    st.header("Выгрузка модели из инференса")

    if st.button(f"Выгрузить модели"):
        # resp = unload_model_sync()
        resp = await unload_model()
        st.success(resp)

if __name__ == "__main__":
    asyncio.run(process_page())