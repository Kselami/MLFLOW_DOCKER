from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os
from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model = None

class PredictionRequest(BaseModel):
    data: List[List[float]]

def load_model():
    global model
    try:
        model_name = os.getenv("MODEL_NAME", "CancerClassifier")
        logger.info(f"Loading model: {model_name}")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        logger.info(f"Successfully loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        raise

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_data = pd.DataFrame(request.data)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": os.getenv("MODEL_NAME", "CancerClassifier"),
        "model_type": str(type(model)),
        "status": "loaded"
    }

