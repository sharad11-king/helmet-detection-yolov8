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

model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists("best.pt"):
        model = YOLO("best.pt")
        print("✅ Model loaded")
    else:
        print("❌ best.pt not found")

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
            .card { background: white; border-radius: 20px; padding: 30px; max-width: 800px; margin: auto; }
            input { margin: 20px 0; }
            button { background: #667eea; color: white; padding: 10px 30px; border: none; border-radius: 25px; cursor: pointer; }
            img { max-width: 100%; margin-top: 20px; border-radius: 10px; }
            .result { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🪖 Helmet Detection System</h1>
            <input type="file" id="file" accept="image/*">
            <br>
            <button onclick="detect()">Detect Helmets</button>
            <div class="result" id="result"></div>
        </div>
        <script>
            async function detect() {
                const file = document.getElementById('file').files[0];
                if (!file) return alert('Select an image');
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('result').innerHTML = '<p>Processing...</p>';
                
                const response = await fetch('/detect', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('result').innerHTML = `
                        <h2>Found ${data.count} helmet(s)</h2>
                        <img src="data:image/jpeg;base64,${data.image}">
                    `;
                } else {
                    document.getElementById('result').innerHTML = `<p>Error: ${data.error}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return {"success": False, "error": "Model not loaded"}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = model(image, conf=0.5)
        
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    detections.append({
                        "class": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0])
                    })
        
        draw = ImageDraw.Draw(image)
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image": img_base64
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
