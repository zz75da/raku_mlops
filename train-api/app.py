from fastapi import FastAPI, HTTPException, Header, Depends
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
from services.data_loader import load_and_merge_data
from services.preprocess_text import extract_text_features
from services.preprocess_image import extract_image_features
from services.pca_reducer import reduce_features
from services.trainer import build_and_train_model, evaluate_model
from services.artifacts import save_artifacts
from utils.logger import get_logger
import requests
import os

logger = get_logger()
app = FastAPI(title="Train API", description="Preprocessing + training", version="1.0")

# --- Prometheus metrics ---
train_loss_gauge = Gauge("train_loss", "Training loss per epoch")
val_loss_gauge = Gauge("val_loss", "Validation loss per epoch")
train_acc_gauge = Gauge("train_accuracy", "Training accuracy per epoch")
val_acc_gauge = Gauge("val_accuracy", "Validation accuracy per epoch")
epochs_counter = Counter("epochs_completed_total", "Total epochs completed")

# Instrument FastAPI to expose /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Gate-API URL and auth
GATE_API_URL = os.getenv("GATE_API_URL", "http://gate-api:5000")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
ALGORITHM = "HS256"


async def verify_jwt_token(authorization: str = Header(...)):
    """Verify JWT token using gate-api /validate-token endpoint"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    try:
        resp = requests.post(
            f"{GATE_API_URL}/validate-token",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return resp.json()
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Cannot reach gate-api for token validation")


@app.post("/train")
def train_model(user: dict = Depends(verify_jwt_token), use_dev_images: bool = True):
    """
    Train model, log to MLflow, push to registry, update Prometheus, and return metadata.
    """
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admin can trigger training")

    mode = "DEV" if use_dev_images else "FULL TRAIN"
    logger.info(f"Starting training pipeline by {user['username']} (mode={mode})")

    # --- Load & merge data ---
    train_data = load_and_merge_data(use_dev_images=use_dev_images)
    logger.info(f"Loaded {len(train_data)} rows from dataset (mode={mode})")

    # --- Preprocess ---
    logger.info("Extracting text features...")
    text_features, text_vectorizer = extract_text_features(train_data)

    logger.info("Extracting image features...")
    image_features = extract_image_features(train_data)

    # --- PCA reduction ---
    X_reduced, pca_models = reduce_features(text_features, image_features)
    y = train_data["prdtypecode"]

    # --- Train + MLflow logging + registry ---
    model, label_encoder, history, model_path, run_id, model_version = build_and_train_model(
        X_reduced, y
    )

    # --- Update Prometheus metrics per epoch ---
    for epoch, (t_loss, v_loss, t_acc, v_acc) in enumerate(
        zip(history.history.get("loss", []),
            history.history.get("val_loss", []),
            history.history.get("accuracy", []),
            history.history.get("val_accuracy", [])), start=1
    ):
        train_loss_gauge.set(t_loss)
        val_loss_gauge.set(v_loss)
        train_acc_gauge.set(t_acc)
        val_acc_gauge.set(v_acc)
        epochs_counter.inc()
        logger.debug(f"Epoch {epoch}: loss={t_loss}, val_loss={v_loss}, acc={t_acc}, val_acc={v_acc}")

    # --- Save artifacts ---
    save_artifacts(model, text_vectorizer, pca_models, label_encoder)

    # --- Evaluate model ---
    metrics = evaluate_model(model, X_reduced, y, label_encoder)

    logger.info(f"Training pipeline completed successfully (mode={mode})")

    # --- Return structured response for Airflow ---
    return {
        "status": "success",
        "mode": mode,
        "model_path": model_path,
        "mlflow_run_id": run_id,
        "mlflow_model_version": model_version,
        "history": history.history,
        "final_metrics": metrics,
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "train-api",
        "version": "1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
