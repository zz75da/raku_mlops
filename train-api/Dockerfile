# train-api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building and data handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY train-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and necessary data
COPY train-api/app.py .
COPY data /app/data

EXPOSE 5002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]