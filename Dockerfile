FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# (the default port Streamlit runs on)
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
