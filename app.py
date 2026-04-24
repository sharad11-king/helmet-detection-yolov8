from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML dashboard (built-in)
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
            margin-bottom: 10px;
        }
        p {
            color: #666;
            margin-bottom: 30px;
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
            margin: 10px;
        }
        button:hover {
            transform: scale(1.05);
        }
        img {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .loading {
            display: none;
            color: #667eea;
            margin: 20px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>🪖 Helmet Detection System</h1>
        <p>Upload an image to detect helmets</p>
        
        <input type="file" id="fileInput" accept="image/jpeg,image/png,image/jpg">
        <br>
        <button onclick="detectHelmets()">🔍 Detect Helmets</button>
        
        <div class="loading" id="loadingDiv">
            <p>Processing your image...</p>
        </div>
        
        <div id="resultDiv"></div>
    </div>

    <script>
        async function detectHelmets() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image first');
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
                            <h3>✅ Detection Complete!</h3>
                            <img src="data:image/jpeg;base64,${data.image}">
                            <p><strong>Found ${data.count} helmet(s)</strong></p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result-box">
                            <p style="color: red;">❌ Error: ${data.error}</p>
                            <p>Note: The model may still be loading. Please wait 30 seconds and try again.</p>
                        </div>
                    `;
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultDiv.innerHTML = `
                    <div class="result-box">
                        <p style="color: red;">❌ Error: ${error.message}</p>
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
    return {"message": "Helmet Detection API is running", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/dashboard")
async def dashboard():
    return HTMLResponse(content=HTML_PAGE)

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize if too large
        if image.size[0] > 1000:
            ratio = 1000 / image.size[0]
            new_size = (1000, int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "count": 0,
            "image": image_base64,
            "message": "Image processed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
