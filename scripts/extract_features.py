import numpy as np
import pandas as pd
import spacy
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.text import CountVectorizer
import os

def extract_and_save_features():
    """Pre-extract features and save to files"""
    print("Loading data...")
    X_train = pd.read_csv('/app/data/X_train_update.csv')
    Y_train = pd.read_csv('/app/data/Y_train_CVw08PX.csv')
    
    # Merge and preprocess
    train_data = pd.merge(X_train, Y_train, on='Unnamed: 0')
    train_data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    train_data['description'].fillna(train_data['designation'], inplace=True)
    train_data.dropna(subset=['designation', 'description'], inplace=True)
    
    selected_prdtypecodes = [2905, 1920, 60, 1300, 1180, 2220, 1301, 2462, 1140, 1940, 40]
    train_data_filtered = train_data[train_data["prdtypecode"].isin(selected_prdtypecodes)].copy().reset_index(drop=True)
    
    # Text features
    print("Processing text features...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)
    
    processed_descriptions = [
        preprocess_text(doc.text) if doc else "" 
        for doc in nlp.pipe(train_data_filtered['description'].tolist(), n_process=4)
    ]
    train_data_filtered['processed_description'] = processed_descriptions
    
    text_vectorizer = CountVectorizer(max_features=5000).fit(train_data_filtered['processed_description'])
    text_features = text_vectorizer.transform(train_data_filtered['processed_description']).toarray().astype(np.float32)
    np.save('/app/data/text_features.npy', text_features)
    
    # Image features
    print("Processing image features...")
    train_data_filtered['image_path'] = [
        os.path.join('/app/data/images/image_train', f"image_{row['imageid']}_product_{row['productid']}.jpg") 
        for _, row in train_data_filtered.iterrows()
    ]
    
    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(224, 224, 3))
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_dataframe(
        dataframe=train_data_filtered, x_col='image_path', target_size=(224, 224),
        batch_size=32, class_mode=None, shuffle=False
    )
    
    image_features = base_model.predict(generator, verbose=1).astype(np.float32)
    np.save('/app/data/image_features.npy', image_features)
    
    print("✅ Features extracted and saved successfully!")
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")

if __name__ == "__main__":
    extract_and_save_features()