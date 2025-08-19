import os
from typing import List

import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CancerClassifier API", version="1.0")

MODEL_NAME = os.getenv("MODEL_NAME", "CancerClassifier")
MODEL_STAGE = os.getenv("MODEL_STAGE")  # ex: "Staging" ou "Production"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

FEATURE_COUNT = 30
TARGET_NAMES = ["malignant", "benign"]
model = None

class PredictPayload(BaseModel):
    instances: List[List[float]]  # 30 features par échantillon

def resolve_model_uri() -> str:
    c = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    if MODEL_STAGE:
        latest = c.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not latest:
            raise RuntimeError(f"Aucune version en stage {MODEL_STAGE} pour {MODEL_NAME}")
        return f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    vers = sorted(c.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    if not vers:
        raise RuntimeError(f"Aucune version du modèle {MODEL_NAME} trouvée")
    return f"models:/{MODEL_NAME}/{vers[-1].version}"

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = resolve_model_uri()
    model = mlflow.pyfunc.load_model(uri)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.get("/")
def root():
    return {"message": "CancerClassifier FastAPI", "model": MODEL_NAME}

@app.post("/predict")
def predict(p: PredictPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if len(p.instances) == 0:
        raise HTTPException(status_code=400, detail="instances empty")
    for row in p.instances:
        if len(row) != FEATURE_COUNT:
            raise HTTPException(status_code=400, detail=f"each instance must have {FEATURE_COUNT} features")
    import numpy as np
    preds = model.predict(np.array(p.instances, dtype=float))
    labels = [TARGET_NAMES[int(i)] for i in preds]
    return {"predictions": labels}
