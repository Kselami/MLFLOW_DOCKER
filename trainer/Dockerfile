FROM python:3.11-slim
WORKDIR /app
COPY trainer/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY trainer/entrypoint.py entrypoint.py
COPY trainer/src src
COPY trainer/tests tests
CMD ["python","entrypoint.py"]
