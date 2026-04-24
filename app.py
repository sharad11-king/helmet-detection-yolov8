from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import base64
import os
from PIL import Image
import io

app = FastAPI(title="Helmet Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = "best.pt"
    if os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error: {e}")
            model = None
    else:
        print(f"❌ Model not found at {model_path}")

@app.get("/")
async def root():
    return {"message": "Helmet Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_path = "dashboard.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard not found</h1>")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Run detection
        results = model(img_array, conf=0.5)
        
        # Get detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0])
                    })
        
        # Get annotated image
        annotated = results[0].plot()
        
        # Convert to base64
        annotated_pil = Image.fromarray(annotated)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image": img_base64
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
