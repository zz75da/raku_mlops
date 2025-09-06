import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import requests
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from prometheus_client import CollectorRegistry, push_to_gateway, Gauge
from fastapi import FastAPI, HTTPException, Depends, Header, Body
import jwt
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List
from pydantic import BaseModel

# --- CONFIGURATION ---
RANDOM_SEED = 42
PCA_COMPONENTS = 300
EPOCHS = 10

X_CSV = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
Y_CSV = os.getenv("TRAIN_CSV_Y_PATH", "/app/data/Y_train_CVw08PX.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "rakuten-multimodal")
PROM_GATEWAY = os.getenv("PROM_PUSHGATEWAY_URL", "http://pushgateway:9091")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
ALGORITHM = "HS256"
PREPROCESS_API_URL = os.getenv("PREPROCESS_API_URL", "http://preprocess-api:5001")

app = FastAPI(title="Training API", version="0.3")  # Updated version

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

# --- Prometheus Helpers ---
def push_metrics_to_prometheus(job_name: str, metrics: dict, labels: dict = None):
    try:
        reg = CollectorRegistry()
        for k, v in metrics.items():
            gauge = Gauge(f"ml_{k}", f"{k} from training job", 
                         labelnames=tuple(labels.keys()) if labels else (), 
                         registry=reg)
            if labels:
                gauge.labels(**labels).set(float(v))
            else:
                gauge.set(float(v))
        push_to_gateway(PROM_GATEWAY, job=job_name, registry=reg)
        print(f"[Prometheus] Pushed metrics to {PROM_GATEWAY}")
    except Exception as e:
        print(f"[Prometheus] Push failed: {e}")

# --- Request Models ---
class TrainRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.1
    model_name: str = "neural_network_model"
    experiment_name: str = "production_training"
    enable_mlflow_tracking: bool = True
    mlflow_tracking_uri: str = "http://mlflow:5000"
    log_parameters: Optional[Dict] = None

# --- Helper Functions ---
def load_and_preprocess_data(auth_token: str):
    """Load data and get features from preprocess-api"""
    # Load raw data
    X_train_raw = pd.read_csv(X_CSV)
    Y_train_raw = pd.read_csv(Y_CSV)
    
    # Merge data
    train_data = pd.merge(X_train_raw, Y_train_raw, on='Unnamed: 0')
    train_data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    train_data['description'].fillna(train_data['designation'], inplace=True)
    train_data.dropna(subset=['designation', 'description'], inplace=True)
    
    # Filter to selected categories (same as before)
    selected_prdtypecodes = [2905, 1920, 60, 1300, 1180, 2220, 1301, 2462, 1140, 1940, 40]
    train_data_filtered = train_data[train_data["prdtypecode"].isin(selected_prdtypecodes)].copy()
    train_data_filtered = train_data_filtered.reset_index(drop=True)
    
    return train_data_filtered

def get_image_features(auth_token: str, image_paths: List[str]):
    """Get image features from preprocess-api batch endpoint"""
    try:
        response = requests.post(
            f"{PREPROCESS_API_URL}/extract-image-features-batch",
            json={"image_paths": image_paths},
            headers=get_auth_header(auth_token),
            timeout=300
        )
        
        if response.status_code == 200:
            features_data = response.json()
            # Convert dict to array in the same order as input paths
            features_array = np.array([features_data["features"][path] for path in image_paths 
                                     if features_data["features"][path] is not None])
            return features_array
        else:
            raise Exception(f"Image feature extraction failed: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error getting image features: {e}")

def get_text_features(auth_token: str, descriptions: List[str]):
    """Get text features from preprocess-api"""
    try:
        response = requests.post(
            f"{PREPROCESS_API_URL}/extract-text-features",
            json={"descriptions": descriptions},
            headers=get_auth_header(auth_token),
            timeout=120
        )
        
        if response.status_code == 200:
            features_data = response.json()
            return np.array(features_data["text_features"])
        else:
            raise Exception(f"Text feature extraction failed: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error getting text features: {e}")

def build_multimodal_model(text_dim: int, image_dim: int, num_classes: int):
    """Build model that can handle text-only, image-only, or combined inputs"""
    # Text branch
    text_input = Input(shape=(text_dim,), name='text_input')
    text_branch = Dense(256, activation='relu')(text_input)
    text_branch = Dropout(0.3)(text_branch)
    
    # Image branch  
    image_input = Input(shape=(image_dim,), name='image_input')
    image_branch = Dense(256, activation='relu')(image_input)
    image_branch = Dropout(0.3)(image_branch)
    
    # Combined features
    combined = Concatenate()([text_branch, image_branch])
    combined = Dense(512, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    
    # Output
    output = Dense(num_classes, activation='softmax')(combined)
    
    # Create model with multiple inputs
    model = Model(inputs=[text_input, image_input], outputs=output)
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# --- API ENDPOINTS ---
@app.post("/train")
def train_model(request: TrainRequest, authorization: str = Header(...)):
    """Train multimodal model using features from preprocess-api"""
    try:
        auth_payload = verify_token(authorization)
        auth_token = authorization.split(" ")[1] if authorization.startswith("Bearer ") else authorization
        
        # Load and preprocess data
        train_data = load_and_preprocess_data(auth_token)
        
        # Prepare image paths for batch processing
        image_paths = [
            os.path.join("/app/data/images/image_train", f"image_{row['imageid']}_product_{row['productid']}.jpg")
            for _, row in train_data.iterrows()
        ]
        
        # Get features from preprocess-api
        print("Extracting image features...")
        image_features = get_image_features(auth_token, image_paths)
        
        print("Extracting text features...")
        text_features = get_text_features(auth_token, train_data['description'].tolist())
        
        # Apply PCA to image features (optional, can be done in preprocess-api too)
        pca = IncrementalPCA(n_components=PCA_COMPONENTS)
        image_features_reduced = pca.fit_transform(image_features)
        
        # Encode labels
        y = train_data['prdtypecode']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Train-test split
        (X_train_text, X_val_text, 
         X_train_img, X_val_img, 
         y_train, y_val) = train_test_split(
            text_features, image_features_reduced, y_encoded,
            test_size=request.test_size, random_state=request.random_state
        )
        
        # Build multimodal model
        model = build_multimodal_model(
            text_dim=text_features.shape[1],
            image_dim=image_features_reduced.shape[1],
            num_classes=len(label_encoder.classes_)
        )
        
        # MLflow setup
        mlflow.set_tracking_uri(request.mlflow_tracking_uri)
        mlflow.set_experiment(request.experiment_name)
        
        with mlflow.start_run(run_name=f"multimodal_train_{int(time())}"):
            # Log parameters
            mlflow.log_params({
                "text_features_dim": text_features.shape[1],
                "image_features_dim": image_features_reduced.shape[1],
                "pca_components": PCA_COMPONENTS,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "test_size": request.test_size,
                "samples_count": len(train_data),
                "model_type": "multimodal"
            })
            
            if request.log_parameters:
                mlflow.log_params(request.log_parameters)
            
            # Train model
            history = model.fit(
                [X_train_text, X_train_img], y_train,
                validation_data=([X_val_text, X_val_img], y_val),
                epochs=request.epochs,
                batch_size=request.batch_size,
                verbose=1
            )
            
            # Evaluate
            val_loss, val_accuracy = model.evaluate([X_val_text, X_val_img], y_val, verbose=0)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_accuracy)
            
            # Save artifacts
            artifacts = {
                "pca_model.pkl": pca,
                "label_encoder.pkl": label_encoder
            }
            
            for filename, artifact in artifacts.items():
                with open(f"/tmp/{filename}", "wb") as f:
                    pickle.dump(artifact, f)
                mlflow.log_artifact(f"/tmp/{filename}")
            
            # Log and register model
            mlflow.tensorflow.log_model(
                model, 
                "model", 
                registered_model_name=request.model_name
            )
            
            # Push metrics to Prometheus
            push_metrics_to_prometheus(
                job_name="multimodal_train_job",
                metrics={"val_loss": val_loss, "val_accuracy": val_accuracy},
                labels={"experiment": request.experiment_name, "model_type": "multimodal"}
            )
        
        # Transition to Production
        client = MlflowClient()
        latest_version = client.get_latest_versions(request.model_name, stages=["None"])[0]
        client.transition_model_version_stage(
            name=request.model_name,
            version=latest_version.version,
            stage="Production"
        )
        
        return {
            "message": "Multimodal training completed successfully",
            "accuracy": float(val_accuracy),
            "loss": float(val_loss),
            "model_name": request.model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/evaluate")
def evaluate_model(
    model_version: str = "latest",
    test_data_path: str = "/app/data/X_test.csv",
    push_to_prometheus: bool = True,
    authorization: str = Header(...)
):
    """Evaluate model performance"""
    try:
        # This would need implementation based on your test data
        # For now, return a placeholder response
        return {
            "status": "evaluation_not_implemented",
            "message": "Evaluation endpoint needs implementation with test data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "train-api", "version": "0.3"}

@app.get("/")
def root():
    return {
        "status": "Training API running", 
        "version": "0.3",
        "endpoints": ["/train", "/evaluate", "/health"]
    }