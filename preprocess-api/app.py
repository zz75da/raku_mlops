import os
import numpy as np
import pandas as pd
import spacy 
import spacy.cli
import jwt
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Body, Header, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import base64
import io
from PIL import Image
import concurrent.futures
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import uuid
import hashlib
import json
from pathlib import Path

# --- Config ---
IMAGE_SIZE = (224, 224)
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
ALGORITHM = "HS256"
TEXT_FEATURES_LIMIT = 5000

# --- Cache Configuration ---
CACHE_DIR = "/app/data/feature_cache"
CACHE_ENABLED = os.getenv("FEATURE_CACHE_ENABLED", "true").lower() == "true"

app = FastAPI(title="Preprocessing API", version="0.5")  # Updated version

# --- Globals for heavy models ---
nlp = None
resnet_model = None
text_vectorizer = None

# ---- Auth dependency ----
def get_user_role(payload: dict) -> str:
    """Extract and normalize user role from token payload"""
    if 'role' in payload:
        return payload['role'].lower()
    return 'user'

def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    
    token = authorization.split(" ")[1]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_role = get_user_role(payload)
        if user_role not in ['admin', 'user']:
            raise HTTPException(status_code=403, detail="Insufficient privileges")
        payload['role'] = user_role
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(user: dict = Depends(verify_token)):
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

def require_auth(user: dict = Depends(verify_token)):
    return user

# ---- Lazy loading of models ----
def load_models():
    global nlp, resnet_model, text_vectorizer
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    if resnet_model is None:
        resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    
    if text_vectorizer is None:
        text_vectorizer = CountVectorizer(max_features=TEXT_FEATURES_LIMIT)

# ---- Cache Functions ----
def _get_cache_key(input_data: any, prefix: str) -> str:
    """Generate a unique cache key from input data"""
    if isinstance(input_data, list):
        data_str = json.dumps(input_data, sort_keys=True)
    elif isinstance(input_data, dict):
        data_str = json.dumps(input_data, sort_keys=True)
    else:
        data_str = str(input_data)
    
    return f"{prefix}_{hashlib.md5(data_str.encode()).hexdigest()}"

def _get_cached_features(cache_key: str) -> Optional[any]:
    """Retrieve features from cache if they exist"""
    if not CACHE_ENABLED:
        return None
        
    cache_path = Path(CACHE_DIR) / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                print(f"Cache hit for {cache_key}")
                return cached_data
        except Exception as e:
            print(f"Cache read error: {e}")
    return None

def _save_features_to_cache(cache_key: str, features: any):
    """Save features to cache"""
    if not CACHE_ENABLED:
        return
        
    try:
        cache_path = Path(CACHE_DIR) / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(features, f)
        print(f"Features cached for {cache_key}")
    except Exception as e:
        print(f"Cache save error: {e}")

@app.on_event("startup")
async def startup_event():
    print("Loading models on startup...")
    try:
        load_models()
        # Create cache and feature directories
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path("/app/data/features").mkdir(parents=True, exist_ok=True)
        print(f"Feature cache enabled: {CACHE_ENABLED}")
        print("Models loaded successfully")
    except Exception as e:
        print(f"Failed to load models: {e}")
        # Don't exit, but set status to unhealthy
        global nlp, resnet_model, text_vectorizer
        nlp = None
        resnet_model = None
        text_vectorizer = None

# ---- Helpers ----
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def _process_single_image_path(image_path: str) -> Optional[List[float]]:
    try:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None

        image = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = resnet_model.predict(img_array, verbose=0)
        return features.flatten().tolist()

    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return None

# --- Request Models ---
class ImageFeatureRequest(BaseModel):
    image_data: str

class BatchImagePathRequest(BaseModel):
    image_paths: List[str]

class TextDataRequest(BaseModel):
    descriptions: List[str]

class MultimodalDataRequest(BaseModel):
    descriptions: Optional[List[str]] = None
    image_paths: Optional[List[str]] = None

# --- API Endpoints ---
@app.post("/extract-image-features")
def extract_image_features(request: ImageFeatureRequest, user=Depends(verify_token)):
    """Extract features from base64 image (single image)"""
    try:
        cache_key = _get_cache_key(request.image_data, "image_single")
        cached_features = _get_cached_features(cache_key)
        if cached_features:
            return cached_features

        image_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = resnet_model.predict(img_array, verbose=0)
        features_flat = features.flatten().tolist()
        
        result = {"image_features": features_flat, "dimensions": len(features_flat)}
        _save_features_to_cache(cache_key, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image feature extraction failed: {str(e)}")

@app.post("/extract-image-features-batch")
def extract_image_features_batch(request: BatchImagePathRequest, user=Depends(require_admin)):
    """Batch feature extraction with caching"""
    if not request.image_paths:
        raise HTTPException(status_code=400, detail="No image paths provided")

    cache_key = _get_cache_key(request.image_paths, "image_batch")
    cached_features = _get_cached_features(cache_key)
    if cached_features:
        return cached_features

    print(f"Processing batch of {len(request.image_paths)} images...")
    max_workers = min(8, len(request.image_paths), os.cpu_count() + 2)
    features_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_process_single_image_path, path): path 
            for path in request.image_paths
        }

        for future in concurrent.futures.as_completed(future_to_path):
            image_path = future_to_path[future]
            try:
                feature_vector = future.result()
                features_dict[image_path] = feature_vector
            except Exception as exc:
                print(f"Image {image_path} generated an exception: {exc}")
                features_dict[image_path] = None

    successful = sum(1 for v in features_dict.values() if v is not None)
    print(f"Batch processing complete. Success: {successful}/{len(features_dict)} images.")
    
    result = {"features": features_dict}
    _save_features_to_cache(cache_key, result)
    return result

@app.post("/extract-text-features")
def extract_text_features(request: TextDataRequest, user=Depends(verify_token)):
    """Extract text features with caching"""
    try:
        cache_key = _get_cache_key(request.descriptions, "text_batch")
        cached_features = _get_cached_features(cache_key)
        if cached_features:
            return cached_features

        load_models()
        
        if not request.descriptions:
            raise HTTPException(status_code=400, detail="No descriptions provided")
        
        processed_descriptions = [
            preprocess_text(doc.text) if doc else ""
            for doc in nlp.pipe(request.descriptions, n_process=os.cpu_count())
        ]
        
        text_features = text_vectorizer.fit_transform(processed_descriptions).toarray()
        
        result = {
            "text_features": text_features.tolist(),
            "processed_descriptions": processed_descriptions,
            "dimensions": text_features.shape[1]
        }
        _save_features_to_cache(cache_key, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text feature extraction failed: {str(e)}")

@app.post("/extract-multimodal-features")
def extract_multimodal_features(request: MultimodalDataRequest, user=Depends(require_admin)):
    """Multimodal feature extraction with caching"""
    try:
        cache_key = _get_cache_key({
            "descriptions": request.descriptions,
            "image_paths": request.image_paths
        }, "multimodal")
        
        cached_features = _get_cached_features(cache_key)
        if cached_features:
            return cached_features

        results = {}
        
        if request.descriptions:
            text_response = extract_text_features(
                TextDataRequest(descriptions=request.descriptions), 
                user
            )
            results["text_features"] = text_response["text_features"]
            results["processed_descriptions"] = text_response["processed_descriptions"]
        
        if request.image_paths:
            image_response = extract_image_features_batch(
                BatchImagePathRequest(image_paths=request.image_paths),
                user
            )
            results["image_features"] = image_response["features"]
        
        if not results:
            raise HTTPException(status_code=400, detail="No data provided for processing")
        
        results["input_mode"] = (
            "multimodal" if request.descriptions and request.image_paths else
            "text_only" if request.descriptions else "image_only"
        )
        
        _save_features_to_cache(cache_key, results)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal feature extraction failed: {str(e)}")

@app.post("/save-features")
def save_features(features: dict = Body(...), filename: str = "features", user=Depends(require_admin)):
    """Save extracted features to disk"""
    try:
        output_dir = "/app/data/features"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"{filename}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(features, f)
        
        return {"status": "success", "filepath": filepath}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save features: {str(e)}")

@app.post("/load-features")
def load_features(filename: str = "features", user=Depends(verify_token)):
    """Load previously saved features"""
    try:
        filepath = os.path.join("/app/data/features", f"{filename}.pkl")
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Features file not found")
        
        with open(filepath, "rb") as f:
            features = pickle.load(f)
        
        return features
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load features: {str(e)}")

@app.post("/admin/clear-cache")
def clear_cache(user=Depends(require_admin)):
    """Clear feature cache - admin only"""
    try:
        cache_path = Path(CACHE_DIR)
        if cache_path.exists():
            for file in cache_path.glob("*.pkl"):
                file.unlink()
            return {"status": "cache cleared", "deleted_files": True}
        return {"status": "cache already empty", "deleted_files": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.post("/admin/clear-model-cache")
def clear_model_cache(user=Depends(require_admin)):
    """Clear model cache - admin only"""
    global nlp, resnet_model, text_vectorizer
    nlp = None
    resnet_model = None
    text_vectorizer = None
    load_models()  # Reload models immediately
    return {"status": "model cache cleared", "requires_reload": False}

@app.get("/cache-info")
def cache_info(user=Depends(require_admin)):
    """Get cache statistics"""
    cache_path = Path(CACHE_DIR)
    if cache_path.exists():
        files = list(cache_path.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "enabled": CACHE_ENABLED,
            "cache_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }
    return {"enabled": CACHE_ENABLED, "cache_files": 0, "total_size_bytes": 0}

@app.get("/")
def root():
    return {
        "status": "API up and running",
        "version": "0.5",
        "endpoints": [
            "/health", "/extract-image-features", "/extract-image-features-batch",
            "/extract-text-features", "/extract-multimodal-features", "/save-features",
            "/load-features", "/cache-info", "/admin/clear-cache", "/admin/clear-model-cache"
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if nlp is not None and resnet_model is not None else "unhealthy",
        "nlp_loaded": nlp is not None,
        "resnet_loaded": resnet_model is not None,
        "text_vectorizer_loaded": text_vectorizer is not None,
        "cache_enabled": CACHE_ENABLED,
        "service": "preprocess-api",
        "cpu_count": os.cpu_count()
    }