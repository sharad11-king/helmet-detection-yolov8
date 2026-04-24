from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import base64
import os
from PIL import Image, ImageDraw
import io

app = FastAPI(title="Helmet Detection API")

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
            print("✅ YOLO model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print(f"❌ Model not found at {model_path}")

@app.get("/")
async def root():
    return {"message": "Helmet Detection API", "status": "running", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_path = "dashboard.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("""
    <html>
    <head>
        <title>Helmet Detection</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; }
            .upload-box { border: 2px dashed #ccc; padding: 40px; margin: 20px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            img { max-width: 100%; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>🪖 Helmet Detection System</h1>
        <div class="upload-box">
            <input type="file" id="file" accept="image/*">
            <br><br>
            <button onclick="detect()">Detect Helmets</button>
        </div>
        <div id="result"></div>
        <script>
            async function detect() {
                const file = document.getElementById('file').files[0];
                if (!file) return alert('Please select an image');
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('result').innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/detect', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('result').innerHTML = `
                            <h2>Found ${data.count} helmet(s)</h2>
                            <img src="data:image/jpeg;base64,${data.image}">
                        `;
                    } else {
                        document.getElementById('result').innerHTML = '<p>Error: ' + data.error + '</p>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = '<p>Error: ' + error.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """)

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Model not loaded. Please wait 30 seconds and try again."}
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run YOLO detection
        results = model(image, conf=0.5)
        
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
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    label = f"{model.names[int(box.cls[0])]}: {confidence:.2f}"
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1 - 10), label, fill="red")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image": img_base64
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
