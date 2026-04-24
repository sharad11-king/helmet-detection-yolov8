from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64
import os
from PIL import Image, ImageDraw
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists("best.pt"):
        try:
            model = YOLO("best.pt")
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print("❌ best.pt file not found")

@app.get("/")
async def root():
    return {"message": "Helmet Detection API", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Helmet Detection</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .card { background: white; border-radius: 20px; padding: 30px; max-width: 800px; margin: auto; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
            input { margin: 20px 0; }
            button { background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 25px; cursor: pointer; font-size: 16px; }
            button:hover { background: #764ba2; }
            img { max-width: 100%; margin-top: 20px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            .result { margin-top: 20px; }
            .stats { background: #f0f0f0; padding: 10px; border-radius: 10px; margin-top: 10px; }
            .loading { display: none; color: #667eea; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🪖 Helmet Detection System</h1>
            <p>Upload an image to detect helmets using YOLOv8</p>
            <input type="file" id="file" accept="image/*">
            <br>
            <button onclick="detect()">Detect Helmets</button>
            <div class="loading" id="loading">Processing image...</div>
            <div class="result" id="result"></div>
        </div>
        <script>
            async function detect() {
                const file = document.getElementById('file').files[0];
                if (!file) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';
                
                try {
                    const response = await fetch('/detect', { 
                        method: 'POST', 
                        body: formData 
                    });
                    
                    const data = await response.json();
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.success) {
                        let html = `<img src="data:image/jpeg;base64,${data.image}">`;
                        html += `<div class="stats"><strong>🎯 Found ${data.count} helmet(s)</strong>`;
                        
                        data.detections.forEach((det, i) => {
                            html += `<div>#${i+1}: ${det.class} - ${(det.confidence * 100).toFixed(1)}% confidence</div>`;
                        });
                        
                        html += `</div>`;
                        document.getElementById('result').innerHTML = html;
                    } else {
                        document.getElementById('result').innerHTML = `<div class="stats">❌ Error: ${data.error}</div>`;
                    }
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('result').innerHTML = `<div class="stats">❌ Error: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return {"success": False, "error": "Model not loaded. Please wait 30 seconds and try again."}
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run detection
        results = model(image, conf=0.5)
        
        # Get detections
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    detections.append({
                        "class": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0])
                    })
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    confidence = float(box.conf[0])
                    label = f"{confidence:.2f}"
                    draw.text((x1, y1-15), label, fill="red")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image": img_base64
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
