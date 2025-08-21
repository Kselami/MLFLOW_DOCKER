FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# MLflow + driver Postgres
RUN pip install --no-cache-dir "mlflow>=2.8,<3" psycopg2-binary

WORKDIR /mlflow
RUN mkdir -p /mlflow/artifacts
EXPOSE 5000

# La commande finale est fournie par docker-compose
CMD ["bash","-lc","mlflow --version && tail -f /dev/null"]
