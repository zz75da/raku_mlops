# streamlit/app_streamlit.py - UPDATED
import streamlit as st
import requests
import json
import os
import base64
from PIL import Image
import io

# --- API URLs ---
GATE_API_URL = os.environ.get("GATE_API_URL", "http://gate-api:5000/login")
PREPROCESS_API_URL = os.environ.get("PREPROCESS_API_URL", "http://preprocess-api:5001/extract-image-features")
PREDICT_API_URL = os.environ.get("PREDICT_API_URL", "http://predict-api:5003/predict")


# --- Function to authenticate against Gate API ---
def authenticate_user(username, password):
    try:
        response = requests.post(GATE_API_URL, json={"username": username, "password": password})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter au service d'authentification.")
        return None
    
    
# --- Function to extract image features ---
def extract_image_features(uploaded_image, token):
    """Call preprocess-api to extract 2048-dim features from image"""
    try:
        # Convert image to base64
        image_bytes = uploaded_image.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Call preprocess-api
        response = requests.post(
            PREPROCESS_API_URL,
            json={"image_data": image_b64},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["image_features"]
        else:
            st.error(f"Image processing failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

# --- In your prediction button section, REPLACE the dummy code:
if st.button("Prédire"):
    image_features = None
    if uploaded_image:
        with st.spinner("Extracting image features..."):
            image_features = extract_image_features(uploaded_image, st.session_state['user_token'])
    
    if uploaded_image and not image_features:
        st.error("Failed to extract image features")
        st.stop()

    # Prepare payload with REAL features
    payload = {
        "description": description if description else None,
        "image_features": image_features  # REAL features from preprocess-api
    }

    headers = {
        "Authorization": f"Bearer {st.session_state['user_token']}",
        "Content-Type": "application/json"
    }

    try:
        with st.spinner("Making prediction..."):
            resp = requests.post(PREDICT_API_URL, json=payload, headers=headers, timeout=30)

        if resp.status_code == 200:
            result = resp.json()
            st.success(f"✅ Catégorie prédite: {result['label']} (classe {result['pred_class']})")
            if 'probs' in result:
                st.write("Probabilités par classe:", result['probs'])
        else:
            st.error(f"Erreur API: {resp.text}")
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue: {e}")