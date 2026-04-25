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

# Load your model
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
            const formData = new FormData();
            formData.append('file', file);
            document.getElementById('result').innerHTML = '<p>Processing...</p>';
            const res = await fetch('/detect', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('result').innerHTML = `
                <div class="result-box">
                    <h3>✅ Found ${data.count} helmet(s)</h3>
                    <img src="data:image/jpeg;base64,${data.image}">
                </div>
            `;
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
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    count = 0
    if model:
        results = model(image, conf=0.5)
        draw = ImageDraw.Draw(image)
        for r in results:
            if r.boxes:
                count = len(r.boxes)
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {"image": img_base64, "count": count}
