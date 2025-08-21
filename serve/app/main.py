
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Global variable to store the model
model = None

class PredictionRequest(BaseModel):
    data: List[List[float]]

def load_model():
    global model
    try:
        model_name = os.getenv("MODEL_NAME", "CancerClassifier")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        print(f"Loaded production model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_data = pd.DataFrame(request.data)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
def model_info():
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_name": os.getenv("MODEL_NAME", "CancerClassifier"),
        "model_type": str(type(model))
    }

