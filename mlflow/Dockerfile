FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mlflow==2.9.2" \
    "psycopg2-binary==2.9.7" \
    "gunicorn==21.2.0"

WORKDIR /mlflow   #### hfj
EXPOSE 5000

CMD ["mlflow", "server", \
    "--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", \
    "--serve-artifacts", \
    "--artifacts-destination", "/mlflow/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--workers", "2"]
