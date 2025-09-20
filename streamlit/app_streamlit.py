# streamlit/app_streamlit.py - SINGLE + BATCH STREAMING PREDICTIONS WITH MLflow PRODUCTION/STAGING MODEL SELECTION
import streamlit as st
import requests
import json
import os
import base64
from pathlib import Path
import pandas as pd
from mlflow.tracking import MlflowClient

# --- API URLs ---
GATE_API_URL = os.environ.get("GATE_API_URL", "http://gate-api:5000/login")
PREPROCESS_API_URL = os.environ.get("PREPROCESS_API_URL", "http://preprocess-api:5001/extract-image-features")
PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://predict-api:5003")

# --- MLflow setup ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "rakuten_model")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def get_model_uri(stage_or_version="Production"):
    """Fetch model URI from MLflow based on stage or version string."""
    try:
        if stage_or_version.isdigit():
            # fetch specific version
            version_info = client.get_model_version(name=MODEL_NAME, version=stage_or_version)
            return version_info.source
        else:
            # fetch latest version from stage
            versions = client.get_latest_versions(MODEL_NAME, stages=[stage_or_version])
            if versions:
                return versions[0].source
            else:
                st.warning(f"⚠️ No model found in MLflow for stage '{stage_or_version}'.")
                return None
    except Exception as e:
        st.error(f"Failed to fetch model from MLflow: {e}")
        return None

# --- Authentication ---
def authenticate_user(username, password):
    try:
        response = requests.post(GATE_API_URL, json={"username": username, "password": password})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to authentication service.")
        return None

# --- Image feature extraction ---
def extract_image_features(uploaded_image, token):
    try:
        image_bytes = uploaded_image.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = requests.post(
            PREPROCESS_API_URL,
            json={"image_data": image_b64},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["image_features"]
        else:
            st.error(f"Image processing failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

# --- Single prediction ---
def predict_single(description, uploaded_image, token, model_uri):
    image_features = None
    if uploaded_image:
        with st.spinner("🔄 Extracting image features..."):
            image_features = extract_image_features(uploaded_image, token)

    if uploaded_image and not image_features:
        st.error("❌ Failed to extract image features")
        st.stop()

    payload = {
        "description": description if description else None,
        "image_data": None,
        "model_uri": model_uri,  # ensure prediction uses selected MLflow model
    }

    if image_features:
        payload["image_data"] = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        with st.spinner("🔮 Making prediction..."):
            resp = requests.post(f"{PREDICT_API_URL}/predict-multimodal", json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"✅ Predicted Category: {result['label']} (class {result['pred_class']})")
            if "probs" in result:
                st.write("📊 Probabilities per class:", result["probs"])
        else:
            st.error(f"API Error: {resp.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# --- Batch prediction with streaming ---
def predict_batch_stream(batch_items, token, output_file_path, model_uri, chunk_size=100):
    """
    Batch prediction streamed to disk in chunks.
    batch_items: list of dicts {"description": str, "image_data": base64 str}
    token: auth token
    output_file_path: path to save predictions
    chunk_size: number of items per API request
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("w", encoding="utf-8") as f_out:
        total_items = len(batch_items)
        st.info(f"🚀 Streaming predictions for {total_items} items in chunks of {chunk_size}...")

        for i in range(0, total_items, chunk_size):
            chunk = batch_items[i:i+chunk_size]
            payload = {"items": chunk, "batch_size": chunk_size, "model_uri": model_uri}

            try:
                with requests.post(f"{PREDICT_API_URL}/predict-multimodal-batch-stream",
                                   json=payload, headers=headers, stream=True, timeout=600) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines():
                            if line:
                                decoded = line.decode("utf-8")
                                f_out.write(decoded + "\n")
                                f_out.flush()
                                st.text(f"✅ Chunk {i//chunk_size + 1}/{(total_items-1)//chunk_size + 1}: {decoded}")
                    else:
                        st.error(f"API error on chunk {i//chunk_size + 1}: {resp.text}")
            except Exception as e:
                st.error(f"Chunk {i//chunk_size + 1} failed: {e}")

    st.success(f"🎉 All batch predictions saved to {output_file_path}")

# --- Streamlit UI ---
st.title("🖼️📦 MLOps Streamlit Prediction Interface")

# User login
if "user_token" not in st.session_state:
    st.subheader("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        auth_resp = authenticate_user(username, password)
        if auth_resp:
            st.session_state["user_token"] = auth_resp["token"]
            st.success("✅ Login successful!")
        else:
            st.error("❌ Login failed")

# Main UI after login
if "user_token" in st.session_state:
    # --- Model version selection ---
    st.sidebar.subheader("🔧 Select MLflow Model Version")
    stage_or_version = st.sidebar.selectbox("Stage or Version", ["Production", "Staging", "1", "2", "3"], index=0)
    model_uri = get_model_uri(stage_or_version)
    if not model_uri:
        st.stop()
    st.sidebar.success(f"Using MLflow model: {model_uri}")

    # --- Single Prediction ---
    st.subheader("🔮 Single Prediction")
    description = st.text_input("Product description")
    uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if st.button("Predict Single"):
        predict_single(description, uploaded_image, st.session_state["user_token"], model_uri)

    # --- Batch Prediction ---
    st.subheader("📑 Batch Prediction (Streaming)")
    uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    descriptions_file = st.file_uploader("Optional: CSV with descriptions", type=["csv"])

    if st.button("Predict Batch"):
        if uploaded_files:
            batch_items = []
            descriptions = []
            if descriptions_file:
                df_desc = pd.read_csv(descriptions_file)
                if "description" in df_desc.columns:
                    descriptions = df_desc["description"].tolist()

            for idx, file in enumerate(uploaded_files):
                desc = descriptions[idx] if idx < len(descriptions) else ""
                batch_items.append({
                    "description": desc,
                    "image_path": None,
                    "image_data": base64.b64encode(file.getvalue()).decode("utf-8")
                })

            output_file = "./batch_predictions_streamed.jsonl"
            predict_batch_stream(batch_items, st.session_state["user_token"], output_file, model_uri)
        else:
            st.warning("⚠️ Please upload at least one image")
