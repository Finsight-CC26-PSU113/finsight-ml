# ============================================================
# OCR FinSight - Production Dockerfile
# Multi-stage build for smaller image size
# ============================================================

# Use slim Python 3.11 (good balance between size and compatibility)
FROM python:3.11-slim AS builder

# Install system dependencies needed for opencv-headless, easyocr
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

# Copy and install Python dependencies (in correct order)
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

# Copy application code
COPY src/ ./src/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Pre-download EasyOCR model (so first request is fast)
RUN python -c "import easyocr; easyocr.Reader(['en', 'id'], gpu=False, verbose=False)"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health', timeout=5)" || exit 1

# Run the app
CMD ["python", "web/simple_app.py"]
