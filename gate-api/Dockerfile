# gate-api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY gate-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gate-api/app.py .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]