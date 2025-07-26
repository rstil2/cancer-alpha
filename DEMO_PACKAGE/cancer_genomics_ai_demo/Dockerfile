# Cancer Genomics AI Classifier - Docker Configuration
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_streamlit.txt .

# Install Python dependencies
RUN pip3 install -r requirements_streamlit.txt

# Copy application files
COPY streamlit_app.py .
COPY test_models.py .
COPY .streamlit/ ./.streamlit/

# Create models directory (mount point for models)
RUN mkdir -p /app/models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
