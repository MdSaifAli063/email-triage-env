FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server/ ./server/
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]