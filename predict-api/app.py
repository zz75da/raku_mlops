import os
import time
import numpy as np
import pickle
import mlflow
import mlflow.pyfunc
import jwt
import requests
from typing import List, Optional, Dict, Generator
from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram
import json
import tempfile

app = FastAPI(title="Prediction API", version="0.6")

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "model_api_requests_total",
    "Total HTTP requests to model API",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "model_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "neural_network_model")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
PREPROCESS_API_URL = os.getenv("PREPROCESS_API_URL", "http://preprocess-api:5001")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
ALGORITHM = "HS256"

# --- Input schemas ---
class TextRequest(BaseModel):
    description: str

class ImageRequest(BaseModel):
    image_data: str  # Base64 encoded image

class ImagePathRequest(BaseModel):
    image_path: str  # Path to image file

class MultimodalRequest(BaseModel):
    description: Optional[str] = None
    image_data: Optional[str] = None
    image_path: Optional[str] = None
    model_uri: Optional[str] = None  # dynamic model choice

class BatchRequest(BaseModel):
    items: List[MultimodalRequest]
    batch_size: Optional[int] = 32
    output_file: Optional[str] = "/app/predictions.jsonl"
    model_uri: Optional[str] = None  # batch-level model choice

# --- Global model cache ---
mlflow_model = None
label_encoder = None
model_cache: Dict[str, object] = {}

# ---- Auth dependency ----
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_auth_header(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ---- MLflow model loading ----
def load_models_from_mlflow():
    global mlflow_model, label_encoder
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE])
        if not versions:
            raise ValueError(f"No model found in stage '{MLFLOW_MODEL_STAGE}'")
        model_uri = versions[0].source
        mlflow_model = mlflow.pyfunc.load_model(model_uri)
        with tempfile.TemporaryDirectory() as tmpdir:
            client.download_artifacts(versions[0].run_id, "label_encoder.pkl", tmpdir)
            label_encoder = pickle.load(open(f"{tmpdir}/label_encoder.pkl", "rb"))
        return True
    except Exception as e:
        print(f"Failed to load MLflow model: {e}")
        return False

def get_model_and_encoder(model_uri: Optional[str]):
    """Return (model, encoder) for a given model_uri. Fallback to default Production model."""
    global mlflow_model, label_encoder
    if model_uri:
        if model_uri in model_cache:
            return model_cache[model_uri]
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            encoder = None
            if "runs:" in model_uri:
                run_id = model_uri.split("/")[-1]
                with tempfile.TemporaryDirectory() as tmpdir:
                    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
                    client.download_artifacts(run_id, "label_encoder.pkl", tmpdir)
                    encoder = pickle.load(open(f"{tmpdir}/label_encoder.pkl", "rb"))
            model_cache[model_uri] = (model, encoder)
            return model, encoder
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_uri}: {e}")
    else:
        if not mlflow_model or not label_encoder:
            raise HTTPException(status_code=503, detail="No default model loaded")
        return mlflow_model, label_encoder

# --- Preprocessing helpers ---
def extract_text_features(auth_token: str, description: str):
    response = requests.post(
        f"{PREPROCESS_API_URL}/extract-text-features",
        json={"descriptions": [description]},
        headers=get_auth_header(auth_token),
        timeout=30
    )
    if response.status_code != 200:
        raise Exception(f"Text feature extraction failed: {response.text}")
    return np.array(response.json()["text_features"][0])

def extract_image_features_from_data(auth_token: str, image_data: str):
    response = requests.post(
        f"{PREPROCESS_API_URL}/extract-image-features",
        json={"image_data": image_data},
        headers=get_auth_header(auth_token),
        timeout=30
    )
    if response.status_code != 200:
        raise Exception(f"Image feature extraction failed: {response.text}")
    return np.array(response.json()["image_features"])

def extract_image_features_from_path(auth_token: str, image_path: str):
    response = requests.post(
        f"{PREPROCESS_API_URL}/extract-image-features-batch",
        json={"image_paths": [image_path]},
        headers=get_auth_header(auth_token),
        timeout=30
    )
    if response.status_code != 200:
        raise Exception(f"Image feature extraction failed: {response.text}")
    return np.array(response.json()["features"][image_path])

# --- Startup event ---
@app.on_event("startup")
def startup_event():
    print(f"MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}, MODEL_NAME={MLFLOW_MODEL_NAME}, STAGE={MLFLOW_MODEL_STAGE}")
    retries = 3
    for i in range(retries):
        if load_models_from_mlflow():
            print("Default model loaded successfully")
            return
        print(f"Retrying MLflow load... ({i+1}/{retries})")
        time.sleep(5)
    print("WARNING: No default model loaded")

# --- Health and root ---
@app.get("/")
def root():
    return {"status": "API up", "version": "0.6"}

@app.get("/health")
def health():
    return {
        "status": "healthy" if mlflow_model and label_encoder else "degraded",
        "mlflow_connected": mlflow_model is not None,
        "model_loaded": mlflow_model is not None,
        "label_encoder_loaded": label_encoder is not None,
    }

@app.post("/reload-model")
def reload_model(authorization: str = Header(...)):
    verify_token(authorization)
    success = load_models_from_mlflow()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload models")
    return {"status": "reloaded"}

# --- Prediction endpoints ---
@app.post("/predict-text")
def predict_text(req: TextRequest, authorization: str = Header(...)):
    auth_token = authorization.split(" ")[1]
    model, encoder = get_model_and_encoder(None)
    text_features = extract_text_features(auth_token, req.description)
    batch_text = np.array([text_features])
    batch_image = np.zeros((1, model._model_impl.input_shape[1][1]))
    probs = model.predict([batch_text, batch_image])
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(encoder.inverse_transform([pred])[0])
    REQUEST_COUNT.labels("/predict-text", "POST", "200").inc()
    return {"pred_class": pred, "label": label, "probs": probs.tolist(), "input_mode": "text_only"}

@app.post("/predict-image")
def predict_image(req: ImageRequest, authorization: str = Header(...)):
    auth_token = authorization.split(" ")[1]
    model, encoder = get_model_and_encoder(None)
    image_features = extract_image_features_from_data(auth_token, req.image_data)
    batch_text = np.zeros((1, model._model_impl.input_shape[0][1]))
    batch_image = np.array([image_features])
    probs = model.predict([batch_text, batch_image])
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(encoder.inverse_transform([pred])[0])
    REQUEST_COUNT.labels("/predict-image", "POST", "200").inc()
    return {"pred_class": pred, "label": label, "probs": probs.tolist(), "input_mode": "image_only"}

@app.post("/predict-multimodal")
def predict_multimodal(req: MultimodalRequest, authorization: str = Header(...)):
    auth_token = authorization.split(" ")[1]
    model, encoder = get_model_and_encoder(req.model_uri)

    text_features = extract_text_features(auth_token, req.description) if req.description else None
    if req.image_data:
        image_features = extract_image_features_from_data(auth_token, req.image_data)
    elif req.image_path:
        image_features = extract_image_features_from_path(auth_token, req.image_path)
    else:
        image_features = None

    if text_features is not None and image_features is not None:
        batch_text = np.array([text_features])
        batch_image = np.array([image_features])
        input_mode = "multimodal"
    elif text_features is not None:
        batch_text = np.array([text_features])
        batch_image = np.zeros((1, model._model_impl.input_shape[1][1]))
        input_mode = "text_only"
    elif image_features is not None:
        batch_text = np.zeros((1, model._model_impl.input_shape[0][1]))
        batch_image = np.array([image_features])
        input_mode = "image_only"
    else:
        raise HTTPException(status_code=400, detail="Failed to extract features")

    probs = model.predict([batch_text, batch_image])
    pred = int(np.argmax(probs, axis=1)[0])
    label = str(encoder.inverse_transform([pred])[0])
    REQUEST_COUNT.labels("/predict-multimodal", "POST", "200").inc()
    return {"pred_class": pred, "label": label, "probs": probs.tolist(), "input_mode": input_mode, "model_uri": req.model_uri or "default"}

@app.post("/predict")
def predict_legacy(req: MultimodalRequest, authorization: str = Header(...)):
    return predict_multimodal(req, authorization)

@app.post("/predict-multimodal-batch-stream")
def predict_multimodal_batch_stream(batch_req: BatchRequest, authorization: str = Header(...)):
    auth_token = authorization.split(" ")[1]
    model, encoder = get_model_and_encoder(batch_req.model_uri)
    output_file = batch_req.output_file
    batch_size = batch_req.batch_size or 32
    total_items = len(batch_req.items)

    def batch_generator() -> Generator[str, None, None]:
        with open(output_file, "w") as f:
            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)
                batch = batch_req.items[start:end]
                for item in batch:
                    try:
                        text_features = extract_text_features(auth_token, item.description) if item.description else None
                        if item.image_data:
                            image_features = extract_image_features_from_data(auth_token, item.image_data)
                        elif item.image_path:
                            image_features = extract_image_features_from_path(auth_token, item.image_path)
                        else:
                            image_features = None

                        if text_features is not None and image_features is not None:
                            batch_text = np.array([text_features])
                            batch_image = np.array([image_features])
                        elif text_features is not None:
                            batch_text = np.array([text_features])
                            batch_image = np.zeros((1, model._model_impl.input_shape[1][1]))
                        elif image_features is not None:
                            batch_text = np.zeros((1, model._model_impl.input_shape[0][1]))
                            batch_image = np.array([image_features])
                        else:
                            continue

                        probs = model.predict([batch_text, batch_image])
                        pred = int(np.argmax(probs, axis=1)[0])
                        label = str(encoder.inverse_transform([pred])[0])
                        result = {"pred_class": pred, "label": label, "probs": probs.tolist(), "model_uri": batch_req.model_uri or "default"}
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        yield json.dumps(result)
                    except Exception as e:
                        error_result = {"error": str(e)}
                        f.write(json.dumps(error_result) + "\n")
                        f.flush()
                        yield json.dumps(error_result)

    return Response(content=batch_generator(), media_type="application/json")

# Mount Prometheus metrics
prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
