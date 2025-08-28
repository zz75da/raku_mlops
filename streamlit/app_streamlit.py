import streamlit as st
import requests
import json
import os
from PIL import Image

# --- New: User Authentication Setup ---
GATE_API_URL = os.environ.get("GATE_API_URL", "http://gate-api:5000/login")
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

# --- Session state for user login ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_token' not in st.session_state:
    st.session_state['user_token'] = None
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

# --- Main App Logic ---
st.title("Rakuten MLOps Product Classifier")

if not st.session_state['logged_in']:
    st.header("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        auth_info = authenticate_user(username, password)
        if auth_info:
            st.session_state['logged_in'] = True
            st.session_state['user_token'] = auth_info['token']
            st.session_state['user_role'] = auth_info['role']
            st.success("Connexion réussie! Bienvenue, " + st.session_state['user_role'] + ".")
            st.rerun() # Rerun to show the main content
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
else:
    # --- Prediction Interface ---
    st.header("Interface de prédiction")
    description = st.text_area("Description produit")
    uploaded_image = st.file_uploader("Image produit", type=["jpg", "png", "jpeg"])

    if st.button("Prédire"):
        if not description and not uploaded_image:
            st.error("Veuillez fournir au moins une description ou une image.")
        else:
            # Prepare payload for the Predict API
            payload = {
                "description": description if description else None
            }
            files = None
            if uploaded_image:
                files = {'image_file': (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}

            # Headers with the authentication token
            headers = {
                "Authorization": f"Bearer {st.session_state['user_token']}"
            }

            try:
                # Send request to the Predict API
                resp = requests.post(PREDICT_API_URL, data=payload, files=files, headers=headers)
                
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"✅ Catégorie prédite: {result['label']} (classe {result['pred_class']})")
                elif resp.status_code == 401:
                    st.error("Autorisation refusée. Veuillez vous reconnecter.")
                else:
                    st.error(f"Erreur API: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API de prédiction.")
            except Exception as e:
                st.error(f"Une erreur inattendue est survenue: {e}")

    # --- Admin Panel (Only for admin users) ---
    if st.session_state['user_role'] == 'admin':
        st.markdown("---")
        st.header("Panneau d'administration")
        st.info("En tant qu'administrateur, vous avez un accès complet aux opérations MLOps.")
        
        st.subheader("Déclencher une nouvelle exécution Airflow")
        st.markdown("""
        Pour déclencher la pipeline d'entraînement via Airflow, vous devez utiliser l'API Airflow.
        Exemple de commande `curl` (depuis le bon terminal):
        
        ```bash
        curl -X POST 'http://localhost:8080/api/v1/dags/train_model_dag/dagRuns' \\
        --header 'Content-Type: application/json' \\
        --header 'Authorization: Basic YWRtaW46YWRtaW4=' \\
        --data-raw '{}'
        ```
        (Note: `YWRtaW46YWRtaW4=` est la version encodée en base64 de `admin:admin`)
        """)
        
        # Add more admin features here (e.g., list experiments, deploy a model)

    # --- Logout button ---
    if st.button("Se déconnecter"):
        st.session_state['logged_in'] = False
        st.session_state['user_token'] = None
        st.session_state['user_role'] = None
        st.info("Déconnexion réussie.")
        st.rerun()