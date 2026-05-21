from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
import time

from src.pipeline import OCRPipeline

# Setup direktori
ROOT_DIR = Path(__file__).parent.parent
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="OCR FinSight Premium API",
    description="Sistem pengenalan entitas struk belanja canggih dengan deep learning.",
    version="2.0.0"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pemasangan aset statis dan perenderan templat
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT_DIR / "web" / "templates"))

# Pipeline global (Singleton)
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("[API] Memuat OCR dan Deep Learning Model ke memori GPU...")
    pipeline = OCRPipeline()
    print("[API] Peladen siap menerima permintaan ekstraksi.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sajikan antarmuka web modern dengan efek kaca (Glassmorphism)."""
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/test", response_class=HTMLResponse)
async def test_upload(request: Request):
    """Test upload page yang lebih sederhana."""
    return templates.TemplateResponse(
        request=request,
        name="test_upload.html"
    )

@app.get("/simple", response_class=HTMLResponse)
async def simple_ui(request: Request):
    """Simple UI dengan hasil ekstraksi yang jelas."""
    return templates.TemplateResponse(
        request=request,
        name="simple_ui.html"
    )

@app.post("/api/extract")
async def extract_receipt(file: UploadFile = File(...)):
    """Proses gambar unggahan dan kembalikan JSON terstruktur kaya geometri."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tidak ada berkas yang diunggah.")
        
    timestamp = int(time.time() * 1000)
    # Hapus spasi dari nama berkas demi keamanan URL
    safe_name = f"{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = UPLOADS_DIR / safe_name
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"\n[API] Memproses gambar: {safe_name}")
        t0 = time.time()
        result = pipeline.process(file_path)
        elapsed = time.time() - t0
        print(f"[API] Sukses diekstrak dalam {elapsed:.2f} detik.\n")
        
        result['image_url'] = f"/api/uploads/{safe_name}"
        result['processing_time_sec'] = round(elapsed, 2)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"[API] GAGAL memproses: {e}")
        raise HTTPException(status_code=500, detail=f"Kegagalan internal: {str(e)}")

@app.get("/api/uploads/{filename}")
async def get_uploaded_image(filename: str):
    """Sajikan berkas citra untuk penumpukan penanda visual pada peramban."""
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Citra tidak ditemukan di peladen.")
    return FileResponse(file_path)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("OCR FinSight 2.0 - Starting Server")
    print("=" * 60)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
