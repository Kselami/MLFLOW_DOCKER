FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mlflow==2.9.2" \
    "psycopg2-binary==2.9.7" \
    "gunicorn==21.2.0"

WORKDIR /mlflow
EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["mlflow", "server", \
    "--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", \
    "--serve-artifacts", \
    "--artifacts-destination", "/mlflow/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--workers", "2"]
