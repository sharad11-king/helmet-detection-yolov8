from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from datetime import datetime

app = FastAPI(title="Helmet Detection API", description="YOLOv8 Helmet Detection")

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = "best.pt"
    if os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            print("✅ YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print(f"❌ Model not found at {model_path}")

@app.get("/")
async def root():
    return {"message": "Helmet Detection API is running", "endpoints": ["/dashboard", "/detect", "/health"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the HTML dashboard"""
    html_path = "dashboard.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Dashboard not found</h1>
                <p>Please upload dashboard.html file</p>
            </body>
        </html>
        """)

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    """Detect helmets in uploaded image"""
    
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Please try again in a few seconds."}
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO detection
        results = model(img, conf=0.5)  # 50% confidence threshold
        
        # Get detection results
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "class": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    })
        
        # Draw bounding boxes on image
        annotated_img = results[0].plot()
        
        # Convert to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image": img_base64,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
