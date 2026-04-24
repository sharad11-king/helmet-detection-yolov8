from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from typing import List, Union

# Initialize FastAPI app
app = FastAPI(title="PyTorch ML Model API", description="Model deployment on Render")

# Load model at startup
model = None

class PredictionInput(BaseModel):
    features: List[List[Union[float, int]]]  # Supports batch predictions

class PredictionOutput(BaseModel):
    predictions: List
    status: str

@app.on_event("startup")
async def load_model():
    global model
    model_path = "models/model.pt"  # Your PyTorch model file
    
    if os.path.exists(model_path):
        try:
            # Load PyTorch model
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()  # Set to evaluation mode
            print("PyTorch model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model not found at {model_path}")

@app.get("/")
async def root():
    return {"message": "PyTorch ML Model API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensor
        features = torch.tensor(input_data.features, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(features)
        
        # Convert tensor to list for JSON response
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        
        return {"predictions": predictions, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))