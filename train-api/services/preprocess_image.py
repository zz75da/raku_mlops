import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

IMAGE_SIZE = (224, 224)

def extract_image_features(data, batch_size=64):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_dataframe(
        dataframe=data,
        x_col="image_path",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    image_features = base_model.predict(generator, verbose=1)
    return image_features
