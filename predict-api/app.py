import os
import time
import numpy as np
import pickle
import mlflow
import mlflow.pyfunc
import jwt
import requests
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Depends, Header, Body
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram

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

# --- JWT Config ---
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

app = FastAPI(title="Prediction API", version="0.4")  # Updated version

# Global variables for models
mlflow_model = None
label_encoder = None

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
    """Generate authorization header for internal API calls"""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ---- Load models from MLflow ----
def load_models_from_mlflow():
    global mlflow_model, label_encoder

    print("Attempting to load models from MLflow...")
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_versions = client.get_latest_versions(
            MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE]
        )
        if not model_versions:
            raise ValueError(
                f"No model found in stage '{MLFLOW_MODEL_STAGE}' for name '{MLFLOW_MODEL_NAME}'"
            )

        model_uri = model_versions[0].source
        print(f"Loading model from URI: {model_uri}")
        mlflow_model = mlflow.pyfunc.load_model(model_uri)

        # Download and load label encoder
        client.download_artifacts(
            run_id=model_versions[0].run_id, path="label_encoder.pkl", dst_path="/tmp"
        )
        label_encoder = pickle.load(open("/tmp/label_encoder.pkl", "rb"))

        print("Models and artifacts loaded successfully from MLflow.")
        return True
    except Exception as e:
        print(f"Failed to load models from MLflow: {e}")
        return False

# --- Preprocessing helpers that call preprocess-api ---
def extract_text_features(auth_token: str, description: str):
    """Call preprocess-api to extract text features"""
    try:
        response = requests.post(
            f"{PREPROCESS_API_URL}/extract-text-features",
            json={"descriptions": [description]},
            headers=get_auth_header(auth_token),
            timeout=30
        )
        
        if response.status_code == 200:
            features_data = response.json()
            return np.array(features_data["text_features"][0])  # Get first (and only) result
        else:
            raise Exception(f"Text feature extraction failed: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error getting text features: {e}")

def extract_image_features_from_data(auth_token: str, image_data: str):
    """Call preprocess-api to extract features from base64 image"""
    try:
        response = requests.post(
            f"{PREPROCESS_API_URL}/extract-image-features",
            json={"image_data": image_data},
            headers=get_auth_header(auth_token),
            timeout=30
        )
        
        if response.status_code == 200:
            features_data = response.json()
            return np.array(features_data["image_features"])
        else:
            raise Exception(f"Image feature extraction failed: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error getting image features: {e}")

def extract_image_features_from_path(auth_token: str, image_path: str):
    """Call preprocess-api to extract features from image path"""
    try:
        response = requests.post(
            f"{PREPROCESS_API_URL}/extract-image-features-batch",
            json={"image_paths": [image_path]},
            headers=get_auth_header(auth_token),
            timeout=30
        )
        
        if response.status_code == 200:
            features_data = response.json()
            return np.array(features_data["features"][image_path])
        else:
            raise Exception(f"Image feature extraction failed: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error getting image features: {e}")

@app.on_event("startup")
def startup_event():
    print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    print(f"MLFLOW_MODEL_NAME: {MLFLOW_MODEL_NAME}")
    print(f"MLFLOW_MODEL_STAGE: {MLFLOW_MODEL_STAGE}")
    print(f"PREPROCESS_API_URL: {PREPROCESS_API_URL}")
    
    # Test MLflow connection first
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiments = client.search_experiments()
        print(f"Connected to MLflow. Found {len(experiments)} experiments")
        
        # Check if our model exists
        try:
            model_versions = client.get_latest_versions(
                MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE, "None"]
            )
            print(f"Found {len(model_versions)} versions of model '{MLFLOW_MODEL_NAME}'")
        except Exception as model_error:
            print(f"Model not found yet: {model_error}")
            
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        # Don't crash, but note the connection issue
        print("Will continue without MLflow connection - predictions will fail")
    
    # Load models with retries (this is what your original code does)
    retries = 3
    models_loaded = False
    
    for i in range(retries):
        if load_models_from_mlflow():
            print("Models loaded from MLflow")
            models_loaded = True
            break
        print(f"Retrying MLflow load... ({i+1}/{retries})")
        time.sleep(5)
    
    if not models_loaded:
        print("WARNING: No models found in MLflow. This is normal before training.")
        print("Predict-api will start but predictions will fail until a model is trained.")
        print("Run the training DAG to register a model in MLflow.")
        # Don't raise an error - let the service start


@app.get("/")
def root():
    return {
        "status": "API up and running",
        "version": "0.4",
        "endpoints": ["/predict", "/predict-text", "/predict-image", "/predict-multimodal", "/health", "/metrics", "/reload-model"],
    }

@app.get("/health")
def health_check():
    mlflow_connected = False
    model_exists = False
    
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.search_experiments()
        mlflow_connected = True
        
        # Check if model exists
        try:
            model_versions = client.get_latest_versions(
                MLFLOW_MODEL_NAME, stages=[MLFLOW_MODEL_STAGE, "None"]
            )
            model_exists = len(model_versions) > 0
        except:
            model_exists = False
            
    except:
        mlflow_connected = False
    
    # Determine overall status
    if mlflow_model and label_encoder:
        status = "healthy"
    elif mlflow_connected and model_exists:
        status = "degraded"  # Model exists but not loaded yet
    elif mlflow_connected:
        status = "degraded"  # Connected but no model
    else:
        status = "unhealthy"  # No MLflow connection
    
    return {
        "status": status,
        "mlflow_connected": mlflow_connected,
        "model_exists_in_mlflow": model_exists,
        "model_loaded": mlflow_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "service": "predict-api",
        "message": "Ready for predictions" if mlflow_model else "Run training to load model"
    }


@app.post("/reload-model")
def reload_model(authorization: str = Header(...)):
    auth_payload = verify_token(authorization)
    success = load_models_from_mlflow()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload models")
    return {"status": "reloaded"}

@app.post("/predict-text")
def predict_text(req: TextRequest, authorization: str = Header(...)):
    """Predict from text only"""
    with REQUEST_LATENCY.labels(endpoint="/predict-text").time():
        try:
            auth_payload = verify_token(authorization)
            auth_token = authorization.split(" ")[1]
            
            if not mlflow_model or not label_encoder:
                raise HTTPException(status_code=503, detail="Models not loaded")
            
            # Extract text features using preprocess-api
            text_features = extract_text_features(auth_token, req.description)
            
            # For text-only prediction, we need to provide zeros for image input
            # This depends on how your multimodal model is structured
            # If your model expects both inputs, we need to provide zeros for the missing one
            batch_text = np.array([text_features])
            batch_image = np.zeros((1, mlflow_model._model_impl.input_shape[1][1]))  # Image input shape
            
            # Make prediction
            probs = mlflow_model.predict([batch_text, batch_image])
            pred = int(np.argmax(probs, axis=1)[0])
            label = str(label_encoder.inverse_transform([pred])[0])
            
            REQUEST_COUNT.labels(endpoint="/predict-text", method="POST", status="200").inc()
            return {
                "pred_class": pred, 
                "label": label, 
                "probs": probs.tolist(),
                "input_mode": "text_only"
            }

        except HTTPException:
            REQUEST_COUNT.labels(endpoint="/predict-text", method="POST", status="500").inc()
            raise
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/predict-text", method="POST", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-image")
def predict_image(req: ImageRequest, authorization: str = Header(...)):
    """Predict from image only (base64 encoded)"""
    with REQUEST_LATENCY.labels(endpoint="/predict-image").time():
        try:
            auth_payload = verify_token(authorization)
            auth_token = authorization.split(" ")[1]
            
            if not mlflow_model or not label_encoder:
                raise HTTPException(status_code=503, detail="Models not loaded")
            
            # Extract image features using preprocess-api
            image_features = extract_image_features_from_data(auth_token, req.image_data)
            
            # For image-only prediction, provide zeros for text input
            batch_text = np.zeros((1, mlflow_model._model_impl.input_shape[0][1]))  # Text input shape
            batch_image = np.array([image_features])
            
            # Make prediction
            probs = mlflow_model.predict([batch_text, batch_image])
            pred = int(np.argmax(probs, axis=1)[0])
            label = str(label_encoder.inverse_transform([pred])[0])
            
            REQUEST_COUNT.labels(endpoint="/predict-image", method="POST", status="200").inc()
            return {
                "pred_class": pred, 
                "label": label, 
                "probs": probs.tolist(),
                "input_mode": "image_only"
            }

        except HTTPException:
            REQUEST_COUNT.labels(endpoint="/predict-image", method="POST", status="500").inc()
            raise
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/predict-image", method="POST", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-multimodal")
def predict_multimodal(req: MultimodalRequest, authorization: str = Header(...)):
    """Predict from both text and image"""
    with REQUEST_LATENCY.labels(endpoint="/predict-multimodal").time():
        try:
            auth_payload = verify_token(authorization)
            auth_token = authorization.split(" ")[1]
            
            if not mlflow_model or not label_encoder:
                raise HTTPException(status_code=503, detail="Models not loaded")
            
            # Validate input
            if not req.description and not req.image_data and not req.image_path:
                raise HTTPException(status_code=400, detail="No input provided")
            
            # Extract features based on what's provided
            text_features = None
            image_features = None
            
            if req.description:
                text_features = extract_text_features(auth_token, req.description)
            
            if req.image_data:
                image_features = extract_image_features_from_data(auth_token, req.image_data)
            elif req.image_path:
                image_features = extract_image_features_from_path(auth_token, req.image_path)
            
            # Prepare inputs for the model
            if text_features is not None and image_features is not None:
                batch_text = np.array([text_features])
                batch_image = np.array([image_features])
                input_mode = "multimodal"
            elif text_features is not None:
                batch_text = np.array([text_features])
                batch_image = np.zeros((1, mlflow_model._model_impl.input_shape[1][1]))
                input_mode = "text_only"
            elif image_features is not None:
                batch_text = np.zeros((1, mlflow_model._model_impl.input_shape[0][1]))
                batch_image = np.array([image_features])
                input_mode = "image_only"
            else:
                raise HTTPException(status_code=400, detail="Failed to extract features from inputs")
            
            # Make prediction
            probs = mlflow_model.predict([batch_text, batch_image])
            pred = int(np.argmax(probs, axis=1)[0])
            label = str(label_encoder.inverse_transform([pred])[0])
            
            REQUEST_COUNT.labels(endpoint="/predict-multimodal", method="POST", status="200").inc()
            return {
                "pred_class": pred, 
                "label": label, 
                "probs": probs.tolist(),
                "input_mode": input_mode
            }

        except HTTPException:
            REQUEST_COUNT.labels(endpoint="/predict-multimodal", method="POST", status="500").inc()
            raise
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/predict-multimodal", method="POST", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

# Backward compatibility endpoint
@app.post("/predict")
def predict_legacy(req: MultimodalRequest, authorization: str = Header(...)):
    """Legacy endpoint that routes to multimodal prediction"""
    return predict_multimodal(req, authorization)

# Mount Prometheus metrics
prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)