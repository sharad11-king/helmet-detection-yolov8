from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64
from PIL import Image, ImageDraw
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = None
if os.path.exists("best.pt"):
    try:
        model = YOLO("best.pt")
        print("✅ Helmet model loaded")
    except:
        pass

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Helmet Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 800px;
            margin: auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        img {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 10px;
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>🪖 Helmet Detection System</h1>
        <p>Upload an image to detect helmets</p>
        
        <input type="file" id="fileInput" accept="image/*">
        <br>
        <button onclick="detectHelmets()">Detect Helmets</button>
        
        <div id="resultDiv"></div>
    </div>

    <script>
        async function detectHelmets() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) {
                alert('Please select an image');
                return;
            }
            
            const resultDiv = document.getElementById('resultDiv');
            resultDiv.innerHTML = '<p>Processing...</p>';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.image) {
                    let html = `<div class="result-box"><h3>✅ Found ${data.count} helmet(s)</h3>`;
                    html += `<img src="data:image/jpeg;base64,${data.image}">`;
                    if (data.detections && data.detections.length > 0) {
                        html += `<p>`;
                        data.detections.forEach(d => {
                            html += `${d.class}: ${(d.confidence * 100).toFixed(1)}%<br>`;
                        });
                        html += `</p>`;
                    }
                    html += `</div>`;
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = `<div class="result-box"><p>Error: ${data.error}</p></div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result-box"><p>Error: ${error.message}</p></div>`;
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
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        detections = []
        
        if model:
            results = model(image, conf=0.5)
            
            draw = ImageDraw.Draw(image)
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        detections.append({
                            "class": model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0])
                        })
        
        if image.size[0] > 800:
            ratio = 800 / image.size[0]
            new_size = (800, int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image": image_base64
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
