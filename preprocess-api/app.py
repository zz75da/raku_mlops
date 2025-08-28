# preprocess-api/app.py
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
import spacy
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

app = FastAPI(title="Preprocessing API", version="0.1")

# --- Configuration (from original train.py) ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
TEXT_FEATURES_LIMIT = 5000
PCA_COMPONENTS = 300
PCA_BATCH_SIZE = 512

# --- Spacy & ResNet50 ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

# --- Helper functions (from original train.py) ---
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# --- API Endpoints ---
@app.post("/extract-features")
def extract_features(data: list = Body(...)):
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    df = pd.DataFrame(data)
    
    # Text preprocessing
    processed_descriptions = [
        preprocess_text(doc.text) if doc else ""
        for doc in nlp.pipe(df['description'].tolist(), n_process=os.cpu_count())
    ]
    df['processed_description'] = processed_descriptions
    
    # Return processed data for the training service to handle
    return {"processed_data": df.to_dict('records')}

@app.get("/health")
def health_check():
    return {"status": "healthy"}