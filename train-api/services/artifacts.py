import pickle
import os

ARTIFACTS_PATH = "/app/data/artifacts"

def save_artifacts(model, vectorizer, pca_models, label_encoder):
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    model.save(os.path.join(ARTIFACTS_PATH, "neural_network_model.h5"))

    pickle.dump(vectorizer, open(os.path.join(ARTIFACTS_PATH, "text_vectorizer.pkl"), "wb"))
    pickle.dump(pca_models["image"], open(os.path.join(ARTIFACTS_PATH, "pca_image.pkl"), "wb"))
    pickle.dump(pca_models["text"], open(os.path.join(ARTIFACTS_PATH, "pca_text.pkl"), "wb"))
    pickle.dump(label_encoder, open(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"), "wb"))
