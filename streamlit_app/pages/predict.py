import streamlit as st
from streamlit_app.utils.utils import predict_corpus, predict_text
import asyncio

async def process_page():
    st.header("Предсказание текста")

    text_input = st.text_area("Введите текст для предсказания", height=200)

    if st.button("Предсказать"):
        if not text_input.strip():
            st.error("Введите текст для предсказания!")
        else:
            # resp = predict_text_sync({"X":text_input})
            resp = await predict_text({"X": text_input})
            st.success(predict_response[resp[1]['prediction'][0]])

    st.header("Предсказание на корпусе текстов")

    st.markdown("""
            - Файл обязательно должен содержать столбец `text`
            """)

    uploaded_file = st.file_uploader("Загрузите файл для предсказания (.csv)", type=["csv"])
    if uploaded_file:
        texts = pd.read_csv(uploaded_file)['text'].tolist()
        st.write("Тексты для предсказания:")
        st.write(texts[:5])

        if st.button("Предсказать на корпусе текстов"):
            # resp = predict_corpus_sync({"X": texts})
            resp = await predict_corpus({"X": texts})
            st.success([predict_response[t] for t in resp[1]['prediction']])

if __name__ == "__main__":
    asyncio.run(process_page())