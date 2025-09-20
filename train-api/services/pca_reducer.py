import numpy as np
from sklearn.decomposition import IncrementalPCA

def reduce_features(text_features, image_features, n_components=300):
    pca_img = IncrementalPCA(n_components=n_components)
    img_reduced = pca_img.fit_transform(image_features)

    target_dim = 5300
    n_text_comp = target_dim - img_reduced.shape[1]

    pca_text = IncrementalPCA(n_components=n_text_comp)
    txt_reduced = pca_text.fit_transform(text_features)

    X_reduced = np.hstack([txt_reduced, img_reduced])
    return X_reduced, {"image": pca_img, "text": pca_text}
