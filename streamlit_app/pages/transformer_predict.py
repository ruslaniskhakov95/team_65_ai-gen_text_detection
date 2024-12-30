import asyncio

import streamlit as st

from streamlit_app.utils.transfromer_utils import predict


async def process_page():

    st.header("Предсказание на корпусе текстов")

    st.markdown(
        """
            - Файл обязательно должен содержать столбец `text`
            """
    )

    uploaded_file = st.file_uploader(
        "Загрузите файл для предсказания (.csv)", type=["csv"]
    )
    if uploaded_file:
        texts = pd.read_csv(uploaded_file)["text"].tolist()
        st.write("Тексты для предсказания:")
        st.write(texts[:5])

        selected_model = st.selectbox(
            "Выберите тип модели для предсказания", await get_status()
        )

        if st.button("Предсказать на корпусе текстов"):
            resp = await predict({"X": texts, "model_type": selected_model})
            st.success([t for t in resp["predictions"]])


if __name__ == "__main__":
    asyncio.run(process_page())
