import asyncio
from pathlib import Path
from PIL import Image
import sys
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))


async def show_main_page():
    '''
    Show the main page of the app.
    '''
    image = Image.open("streamlit_app/image/i_am_robot.jpeg")
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="AI Detection",
        page_icon=image,
    )

    st.title("Human or AI?")
    st.image(image)


if __name__ == "__main__":
    asyncio.run(show_main_page())
