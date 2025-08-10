# Cancer Alpha - LightGBM SMOTE Production API
# ============================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY api/requirements_lightgbm.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements_lightgbm.txt

# Copy application files
COPY api/lightgbm_api.py .
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache

# Set environment variables
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check for LightGBM API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Non-root user for security
RUN groupadd -r cancer-alpha && useradd -r -g cancer-alpha cancer-alpha
RUN chown -R cancer-alpha:cancer-alpha /app
USER cancer-alpha

# Command to run the production API
CMD ["python", "-m", "uvicorn", "lightgbm_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
