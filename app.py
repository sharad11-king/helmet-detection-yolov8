from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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

@app.get("/")
async def root():
    return {"message": "Helmet Detection API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Helmet Detection</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .card { background: white; border-radius: 20px; padding: 30px; max-width: 600px; margin: auto; }
            input, button { margin: 10px; padding: 10px 20px; }
            button { background: #667eea; color: white; border: none; border-radius: 25px; cursor: pointer; }
            img { max-width: 100%; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🪖 Helmet Detection</h1>
            <input type="file" id="file" accept="image/*">
            <br>
            <button onclick="detect()">Detect</button>
            <div id="result"></div>
        </div>
        <script>
            async function detect() {
                const file = document.getElementById('file').files[0];
                if (!file) return alert('Select image');
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('result').innerHTML = '<p>Processing...</p>';
                
                try {
                    const res = await fetch('/detect', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    if (data.image) {
                        document.getElementById('result').innerHTML = '<img src="data:image/jpeg;base64,' + data.image + '">';
                    } else {
                        document.getElementById('result').innerHTML = '<p>Detection complete</p>';
                    }
                } catch(e) {
                    document.getElementById('result').innerHTML = '<p>Error: ' + e.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Just return the image (no model - for testing)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {"success": True, "image": img_base64, "count": 0}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
