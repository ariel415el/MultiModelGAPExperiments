import numpy as np
from sklearn.manifold import TSNE


def pca(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    U, S, Vt = np.linalg.svd(centered_data)
    principal_components = Vt.T
    eigenvalues = S**2 / (len(data) - 1)

    return principal_components, eigenvalues, mean


def get_pcas(text_features, image_features, pca_mode='multimodal'):
    if pca_mode == 'multimodal':
        data = np.concatenate((text_features, image_features))
    elif pca_mode == 'text':
        data = text_features
    else:
       data = image_features
    PCs, _, mean = pca(data)
    text_embeddings = np.dot(text_features - mean, PCs)
    image_embeddings = np.dot(image_features - mean, PCs)
    return text_embeddings, image_embeddings


def get_TSNE_embeddings(text_features, image_features):
    model = TSNE(n_components=2)
    n = len(text_features)
    X = model.fit_transform(np.concatenate((text_features, image_features)))
    return X[:n], X[n:]
