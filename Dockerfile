# ===================== Dockerfile =====================
# Containerizes the FastAPI serving app so it runs identically
# on any machine (dev laptop, cloud VM, Kubernetes pod).

FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + trained model artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
