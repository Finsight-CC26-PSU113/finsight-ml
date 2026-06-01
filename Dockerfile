# ============================================================
# OCR FinSight - Production Dockerfile
# Multi-stage build for smaller image size
# Runs: FastAPI (api_v2.py) on port 8000
# ============================================================

FROM python:3.11-slim AS builder

# System dependencies for opencv-headless and easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Step 1: NumPy first (must be < 2.0)
RUN pip install --no-cache-dir 'numpy>=1.26.0,<2.0'

# Step 2: PyTorch CPU (smaller than CUDA, fits most servers)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Everything else
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# Final stage
# ============================================================
FROM python:3.11-slim

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only production-needed code
COPY src/ ./src/
COPY web/ ./web/
COPY models/classifier_v2/ ./models/classifier_v2/
COPY models/finetuned_easyocr/best_model.pth ./models/finetuned_easyocr/best_model.pth

# Pre-download EasyOCR model weights (so first request is fast)
RUN python -c "import easyocr; easyocr.Reader(['en', 'id'], gpu=False, verbose=False)"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=-1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "web.api_v2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
