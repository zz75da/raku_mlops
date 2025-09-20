FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full train-api code
COPY . .

# Set environment variables (default values, override in docker-compose)
ENV GATE_API_URL=http://gate-api:5000
ENV SECRET_KEY=default_secret

RUN python -m spacy download en_core_web_sm

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 5002

# Start FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]
