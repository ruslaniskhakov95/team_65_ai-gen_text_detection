import asyncio

from PIL import Image
import streamlit as st

async def show_main_page():
    image = Image.open('streamlit_app/image/i_am_robot.jpeg')
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="AI Detection",
        page_icon=image,
    )

    st.title("Человек или машина?")
    st.image(image)


if __name__ == "__main__":
    asyncio.run(show_main_page())
