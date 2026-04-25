from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64
from PIL import Image, ImageDraw
import io
import os
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None
if os.path.exists("best.pt"):
    model = YOLO("best.pt")
    print("✅ Model loaded")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Helmet Detection</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card { background: white; border-radius: 20px; padding: 40px; max-width: 800px; margin: auto; }
        button { background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 25px; cursor: pointer; }
        img { max-width: 100%; margin-top: 20px; border-radius: 10px; }
        .result-box { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px; }
        .loading { display: inline-block; width: 20px; height: 20px; border: 2px solid #f3f3f3; border-top: 2px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="card">
        <h1>🪖 Helmet Detection System</h1>
        <input type="file" id="fileInput" accept="image/*">
        <br>
        <button onclick="detect()">Detect Helmets</button>
        <div id="result"></div>
    </div>
    <script>
        async function detect() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) return alert('Select image');
            
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div class="loading"></div><p>Processing...</p>';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/detect', { 
                    method: 'POST', 
                    body: formData,
                    timeout: 30000
                });
                const data = await response.json();
                
                if (data.image) {
                    resultDiv.innerHTML = `
                        <div class="result-box">
                            <h3>✅ Found ${data.count} helmet(s)</h3>
                            <img src="data:image/jpeg;base64,${data.image}">
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="result-box">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result-box">Error: ${error.message}. The server may be overloaded. Try a smaller image.</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return {"message": "API running", "model_loaded": model is not None}

@app.get("/dashboard")
async def dashboard():
    return HTMLResponse(content=HTML_PAGE)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize image to smaller size for faster processing
        if image.size[0] > 640:
            ratio = 640 / image.size[0]
            new_size = (640, int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        count = 0
        if model:
            # Run inference with lower memory usage
            results = model(image, conf=0.5, verbose=False)
            
            draw = ImageDraw.Draw(image)
            for r in results:
                if r.boxes:
                    count = len(r.boxes)
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {"image": img_base64, "count": count}
        
    except Exception as e:
        return {"error": str(e)}
