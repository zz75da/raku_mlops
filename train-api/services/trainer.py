import os
import pickle
import numpy as np
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Global config ---
ARTIFACTS_DIR = "/opt/airflow/data/artifacts"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "rakuten_training")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "rakuten_multimodal")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def build_and_train_model(
    X,
    y,
    epochs: int = 10,
    batch_size: int = 32,
    run_name: str = "training_run"
):
    """
    Build, train, evaluate and log a simple dense model with MLflow.
    Returns model, label_encoder, history, model_path, run_id, registered_version
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Define model
    inputs = Input(shape=(X.shape[1],))
    x = Dense(512, activation="relu")(inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(len(label_encoder.classes_), activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --- MLflow logging ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_dim", X.shape[1])
        mlflow.log_param("num_classes", len(label_encoder.classes_))

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Log metrics per epoch
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Save model locally
        model_path = os.path.join(ARTIFACTS_DIR, "keras_model.h5")
        model.save(model_path)

        # Save label encoder
        encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)

        # Log artifacts
        mlflow.log_artifact(model_path, artifact_path="artifacts")
        mlflow.log_artifact(encoder_path, artifact_path="artifacts")

        # Log model in MLflow format
        mlflow.keras.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    # --- Register model in MLflow Model Registry ---
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    registered_version = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )

    # Transition to "Staging" by default
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=registered_version.version,
        stage="Staging",
        archive_existing_versions=False
    )

    return model, label_encoder, history, model_path, run_id, registered_version.version


def evaluate_model(model, X, y, label_encoder):
    """
    Evaluate trained model and return accuracy + classification report.
    Also logs evaluation metrics to MLflow.
    """
    y_encoded = label_encoder.transform(y)
    y_pred = np.argmax(model.predict(X), axis=1)
    acc = accuracy_score(y_encoded, y_pred)
    report = classification_report(y_encoded, y_pred, output_dict=True)

    # Log final evaluation metrics
    mlflow.log_metric("final_accuracy", acc)
    for label, metrics in report.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            mlflow.log_metric(f"f1_{label}", metrics["f1-score"])

    return {"accuracy": acc, "report": report}
