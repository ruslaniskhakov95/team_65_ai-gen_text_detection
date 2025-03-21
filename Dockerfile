FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
COPY ./streamlit_app /app/streamlit_app

RUN mkdir -p ./streamlit_app/logs_streamlit

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]