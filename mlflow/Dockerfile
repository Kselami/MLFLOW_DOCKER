FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "mlflow>=2.8,<3"

RUN useradd -m app && mkdir -p /mlflow/artifacts && chown -R app:app /mlflow
USER app
WORKDIR /mlflow

ENV BACKEND_URI=sqlite:////mlflow/mlflow.db
ENV ARTIFACTS_DIR=/mlflow/artifacts
EXPOSE 5000

HEALTHCHECK CMD curl -fsS http://localhost:5000/ || exit 1

CMD ["mlflow","server",     "--backend-store-uri","sqlite:////mlflow/mlflow.db",     "--serve-artifacts","--artifacts-destination","/mlflow/artifacts",     "--host","0.0.0.0","--port","5000"]
