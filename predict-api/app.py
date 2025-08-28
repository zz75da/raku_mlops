# predict-api/app.py
import os
import numpy as np
import pickle
import joblib
import mlflow
import mlflow.pyfunc
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram
from starlette.middleware.wsgi import WSGIMiddleware

# --- Prometheus metrics ---
REQUEST_COUNT = Counter("model_api_requests_total", "Total HTTP requests to model API", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("model_api_request_latency_seconds", "Request latency in seconds", ["endpoint"])

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = "neural_network_model" # Name of the registered model
MLFLOW_MODEL_STAGE = "Production" # Or 'Staging', 'Archived', etc.

# Input schema
class PredictRequest(BaseModel):
    description: Optional[str] = None
    image_features: Optional[List[float]] = None

app = FastAPI(title="Prediction API", version="0.1")

# Global variables for models
text_vectorizer = None
label_encoder = None
mlflow_model = None

# ---- Load models from MLflow Model Registry ----
def load_models_from_mlflow():
    global mlflow_model, text_vectorizer, label_encoder
    
    print("Attempting to load models from MLflow...")
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE])
        if not model_versions:
            raise ValueError(f"No model found in stage '{MLFLOW_MODEL_STAGE}' for name '{MLFLOW_MODEL_NAME}'")
        
        model_uri = model_versions[0].source
        print(f"Loading model from URI: {model_uri}")
        
        # Load the MLflow model as a pyfunc model
        mlflow_model = mlflow.pyfunc.load_model(model_uri)
        
        # Download and load associated artifacts
        client.download_artifacts(run_id=model_versions[0].run_id, path="text_vectorizer.pkl", dst_path="/tmp")
        client.download_artifacts(run_id=model_versions[0].run_id, path="label_encoder.pkl", dst_path="/tmp")
        
        text_vectorizer = pickle.load(open("/tmp/text_vectorizer.pkl", "rb"))
        label_encoder = pickle.load(open("/tmp/label_encoder.pkl", "rb"))
        
        print("Models and artifacts loaded successfully from MLflow.")
        return True
    except Exception as e:
        print(f"Failed to load models from MLflow: {e}")
        return False

@app.on_event("startup")
def startup_event():
    load_models_from_mlflow()

@app.get("/")
def root():
    return {"status": "API up and running", "version": "0.1", "endpoints": ["/predict", "/health", "/metrics"]}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if mlflow_model and text_vectorizer and label_encoder else "unhealthy",
        "mlflow_model_loaded": mlflow_model is not None,
        "text_vectorizer_loaded": text_vectorizer is not None,
        "label_encoder_loaded": label_encoder is not None
    }

def _preprocess_text(text):
    if not text or text_vectorizer is None:
        return np.zeros((1, 5000), dtype=np.float32)
    vec = text_vectorizer.transform([text]).toarray().astype(np.float32)
    if vec.shape[1] < 5000:
        pad_width = 5000 - vec.shape[1]
        vec = np.hstack([vec, np.zeros((1, pad_width), dtype=np.float32)])
    return vec

def _preprocess_image_features(img_feats):
    if img_feats is None:
        return np.zeros((1, 300), dtype=np.float32)
    arr = np.array(img_feats, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] != 300:
        raise ValueError(f"Image features shape mismatch: expected 300, got {arr.shape[1]}")
    return arr

@app.post("/predict")
def predict(req: PredictRequest):
    with REQUEST_LATENCY.labels(endpoint="/predict").time():
        try:
            if mlflow_model is None or text_vectorizer is None or label_encoder is None:
                raise HTTPException(status_code=503, detail="Models not loaded on server")

            text_arr = _preprocess_text(req.description)
            img_arr = _preprocess_image_features(req.image_features)

            X = np.hstack([text_arr, img_arr]).astype(np.float32)
            
            # The mlflow_model handles prediction for us
            probs = mlflow_model.predict(X)
            pred = int(np.argmax(probs, axis=1)[0])
            label = str(label_encoder.inverse_transform([pred])[0]) if label_encoder else None

            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
            return {"pred_class": pred, "label": label, "probs": probs.tolist()}

        except HTTPException:
            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
            raise
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)