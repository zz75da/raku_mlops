# train-api/app.py
import os
import pickle
import numpy as np
import pandas as pd
import mlflow
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from prometheus_client import CollectorRegistry, push_to_gateway
from fastapi import FastAPI, HTTPException

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_SEED = 42
TEXT_FEATURES_LIMIT = 5000
PCA_COMPONENTS = 300
PCA_BATCH_SIZE = 512
TOTAL_FEATURES_DIM = 5300
EPOCHS = 10

X_CSV = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
Y_CSV = os.getenv("TRAIN_CSV_Y_PATH", "/app/data/Y_train_CVw08PX.csv")
IMAGE_ROOT_TRAIN = os.getenv("IMAGE_ROOT_TRAIN", "/app/data/images/image_train")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "rakuten-text-image")
PROM_GATEWAY = os.getenv("PROM_PUSHGATEWAY_URL", "http://pushgateway:9091")

app = FastAPI(title="Training API", version="0.1")

# --- Prometheus Helpers ---
def push_metrics_to_prometheus(job_name: str, metrics: dict, labels: dict = None):
    try:
        reg = CollectorRegistry()
        for k, v in metrics.items():
            gauge = Gauge(k, f"{k} from training job", labelnames=tuple((labels or {}).keys()), registry=reg)
            if labels:
                gauge.labels(**labels).set(float(v))
            else:
                gauge.set(float(v))
        push_to_gateway(PROM_GATEWAY, job=job_name, registry=reg)
        print(f"[Prometheus] Pushed metrics to {PROM_GATEWAY}")
    except Exception as e:
        print(f"[Prometheus] Push failed: {e}")

# --- API ENDPOINTS ---
@app.post("/train")
def train_model():
    # Load data from the shared volume
    try:
        X_train_raw = pd.read_csv(X_CSV)
        Y_train_raw = pd.read_csv(Y_CSV)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {e}")

    # Your data preparation, feature extraction, and model training logic goes here.
    # Note: The feature extraction part from the original script is handled here for simplicity,
    # but it could be a separate microservice call if desired.
    
    # 1. Merge and filter data
    train_data = pd.merge(X_train_raw, Y_train_raw, on='Unnamed: 0')
    train_data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    train_data['description'].fillna(train_data['designation'], inplace=True)
    train_data.dropna(subset=['designation', 'description'], inplace=True)
    selected_prdtypecodes = [2905, 1920, 60, 1300, 1180, 2220, 1301, 2462, 1140, 1940, 40]
    train_data_filtered = train_data[train_data["prdtypecode"].isin(selected_prdtypecodes)].copy().reset_index(drop=True)
    train_data_filtered['image_path'] = [os.path.join(IMAGE_ROOT_TRAIN, f"image_{row['imageid']}_product_{row['productid']}.jpg") for _, row in train_data_filtered.iterrows()]

    # 2. Text Features
    # Note: Spacy loading is slow. In a production setting, you'd pre-load this
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
    processed_descriptions = [preprocess_text(doc.text) if doc else "" for doc in nlp.pipe(train_data_filtered['description'].tolist(), n_process=os.cpu_count())]
    train_data_filtered['processed_description'] = processed_descriptions
    text_vectorizer = CountVectorizer(max_features=TEXT_FEATURES_LIMIT).fit(train_data_filtered['processed_description'])
    text_features = text_vectorizer.transform(train_data_filtered['processed_description']).toarray().astype(np.float32)

    # 3. Image Features
    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_dataframe(
        dataframe=train_data_filtered, x_col='image_path', target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode=None, shuffle=False
    )
    image_features = base_model.predict(generator, verbose=0).astype(np.float32)

    # 4. PCA and Concatenation
    pca_image = IncrementalPCA(n_components=PCA_COMPONENTS).fit(image_features)
    image_features_reduced = pca_image.transform(image_features)
    text_pca_dim = TOTAL_FEATURES_DIM - image_features_reduced.shape[1]
    pca_text = IncrementalPCA(n_components=text_pca_dim).fit(text_features)
    text_features_reduced = pca_text.transform(text_features)
    X_reduced = np.hstack([text_features_reduced, image_features_reduced]).astype(np.float32)
    y = train_data_filtered['prdtypecode']
    label_encoder = LabelEncoder().fit(y)
    y_encoded = label_encoder.transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=RANDOM_SEED)

    # 5. Build and Train Model
    input_shape = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    input_layer = Input(shape=(input_shape,))
    x = Dense(512, activation='relu')(input_layer)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    with mlflow.start_run(run_name=f"train_{time():.0f}"):
        mlflow.log_params({
            "text_features_limit": TEXT_FEATURES_LIMIT,
            "pca_components": PCA_COMPONENTS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "total_features_dim": TOTAL_FEATURES_DIM,
        })
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
        # 6. Evaluate and log metrics
        loss, accuracy = model.evaluate(X_val, y_val)
        mlflow.log_metric("val_loss", loss)
        mlflow.log_metric("val_accuracy", accuracy)
        
        # 7. Log artifacts to MLflow
        mlflow.keras.log_model(model, "model")
        with open("text_vectorizer.pkl", "wb") as f:
            pickle.dump(text_vectorizer, f)
        mlflow.log_artifact("text_vectorizer.pkl")
        with open("pca_model.pkl", "wb") as f:
            pickle.dump(pca_image, f)
        mlflow.log_artifact("pca_model.pkl")
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact("label_encoder.pkl")
        
    return {"message": "Model training completed successfully", "accuracy": accuracy}

@app.get("/health")
def health_check():
    return {"status": "healthy"}