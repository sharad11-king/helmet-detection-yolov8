from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
    features: List[List[Union[float, int]]]

class PredictionOutput(BaseModel):
    predictions: List
    status: str

@app.on_event("startup")
async def load_model():
    global model
    model_path = "best.pt"  # Changed from models/model.pt to best.pt (your file is in root)
    
    if os.path.exists(model_path):
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
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

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_path = "dashboard.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Dashboard not found. Please add dashboard.html</h1>", status_code=404)

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = torch.tensor(input_data.features, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(features)
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        
        return {"predictions": predictions, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    # This is where you'll add YOLO detection
    # For now, returning placeholder
    return JSONResponse({
        "image": "",
        "detections": [],
        "count": 0
    })
