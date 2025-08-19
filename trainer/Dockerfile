FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY entrypoint.py entrypoint.py
COPY src src
COPY tests tests
CMD ["python","entrypoint.py"]

