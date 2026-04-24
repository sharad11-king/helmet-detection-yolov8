from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image, ImageDraw
import io
import os
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load YOLO
model = None
try:
    from ultralytics import YOLO
    if os.path.exists("best.pt"):
        model = YOLO("best.pt")
        print("✅ Model loaded")
    else:
        print("best.pt not found")
except Exception as e:
    print(f"Error loading model: {e}")

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
            min-height: 100vh;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 800px;
            margin: auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
        }
        input {
            margin: 20px 0;
            padding: 10px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
        }
        img {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 10px;
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 10px;
        }
        .loading {
            display: none;
            margin: 20px;
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
        
        <div class="loading" id="loadingDiv">
            <p>Processing...</p>
        </div>
        
        <div id="resultDiv"></div>
    </div>

    <script>
        async function detectHelmets() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image');
                return;
            }
            
            const loadingDiv = document.getElementById('loadingDiv');
            const resultDiv = document.getElementById('resultDiv');
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                loadingDiv.style.display = 'none';
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="result-box">
                            <h3>✅ Found ${data.count} helmet(s)</h3>
                            <img src="data:image/jpeg;base64,${data.image}">
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result-box">
                            <p style="color: red;">Error: ${data.error}</p>
                        </div>
                    `;
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultDiv.innerHTML = `
                    <div class="result-box">
                        <p style="color: red;">Error: ${error.message}</p>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return {"message": "API running", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/dashboard")
async def dashboard():
    return HTMLResponse(content=HTML_PAGE)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Process with YOLO if available
        if model is not None:
            results = model(image, conf=0.5)
            
            # Draw boxes
            draw = ImageDraw.Draw(image)
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Count detections
        count = 0
        if model is not None:
            for r in results:
                if r.boxes:
                    count = len(r.boxes)
        
        return JSONResponse(content={
            "success": True,
            "count": count,
            "image": img_base64
        })
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })
