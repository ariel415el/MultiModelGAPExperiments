import os
import numpy as np
from matplotlib import pyplot as plt

from dimensionality_recuction import pca
from utils import get_dataset, get_clip_features


def plot_eiegen_spectrum(model, dataset_name, outputs_dir, device):
    dataset, label_map = get_dataset(dataset_name, model.preprocess)

    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))
    PCs, eigv, mean = pca(np.concatenate((text_features, image_features)))
    # text_embeddings = np.dot(text_features - mean, PCs)
    # image_embeddings = np.dot(image_features - mean, PCs)
    # text_embeddings, image_embeddings = experiment.get_pca_embeddings(text_features, image_features)

    variance_explained = eigv / np.sum(eigv)

    # Plot the variance explained
    plt.figure(figsize=(8, 5))
    plt.step(range(len(variance_explained)), variance_explained, alpha=0.7, label='individual explained variance')
    plt.yscale('log')

    plt.savefig(os.path.join(outputs_dir, "eigen_spectrum.png"))