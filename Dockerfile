# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code
COPY src/ ./src/
COPY conf/ ./conf/

# Create directories for local cache (models will be downloaded from GCS)
RUN mkdir -p /tmp/models /tmp/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV MODEL_BUCKET=timepiece-watch-models
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]