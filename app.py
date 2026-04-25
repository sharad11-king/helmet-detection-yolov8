from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                    resultDiv.innerHTML = `
                        <div class="result-box">
                            <h3>✅ Result</h3>
                            <img src="data:image/jpeg;base64,${data.image}">
                            <p><strong>Found ${data.count} helmet(s)</strong></p>
                        </div>
                    `;
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
    return {"message": "API is running"}

@app.get("/dashboard")
async def dashboard():
    return HTMLResponse(content=HTML_PAGE)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize if too large
        if image.size[0] > 800:
            ratio = 800 / image.size[0]
            new_size = (800, int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "count": 0,
            "image": image_base64
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
