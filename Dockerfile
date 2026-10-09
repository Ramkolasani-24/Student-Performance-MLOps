FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY quality_gate.py .
COPY data/study_performance.csv data/study_performance.csv

RUN python src/train_model.py

EXPOSE 5000

CMD ["python", "app.py"]
