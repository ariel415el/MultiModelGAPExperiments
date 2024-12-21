import os

import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt

from dimensionality_recuction import pca
from utils import get_dataset, get_clip_features


def plot_pca_embeddings(text_embeddings, image_embeddings, title, outputs_dir):
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(111)
    for name, embeddings, color, marker in [("text", text_embeddings, 'blue', '^'),
                                            ("image", image_embeddings, 'red', 'o')]:
        ax.scatter(embeddings[:, pc1], embeddings[:, pc2],
                   color=color, label=name, marker=marker, s=5, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(outputs_dir, f"{title}.png"))
    plt.clf()


def plot_per_class_embeddings(text_embeddings, image_embeddings, all_labels, class_names, title, outputs_dir):
    pc1 = 0
    pc2 = 1
    cmap = plt.get_cmap('tab10')
    labels = np.unique(all_labels)
    colors = [cmap(i) for i in range(len(labels))]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    for label in range(len(labels)):
        idx = all_labels == label
        ax.scatter(text_embeddings[idx, pc1], text_embeddings[idx, pc2],
                   color=colors[label], label=class_names[label], s=10, alpha=0.5, marker='x')
        ax.scatter(image_embeddings[idx, pc1], image_embeddings[idx, pc2],
                   color=colors[label], s=2, alpha=0.5, marker='o')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f"{title}.png"))
    plt.clf()


def plot_2d(model, dataset, dataset_name, outputs_dir, device, label_map=None):
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, _, mean = pca(np.concatenate((text_features, image_features)))
    text_embeddings = (text_features - mean) @ PCs
    image_embeddings = (image_features - mean) @ PCs
    # text_embeddings, image_embeddings = get_TSNE_embeddings(text_features, image_features)

    title = dataset_name
    if dataset_name == 'Flickr8k':
        plot_pca_embeddings(text_embeddings, image_embeddings, title, outputs_dir)
    else:
        plot_per_class_embeddings(text_embeddings, image_embeddings, labels, dataset.classes, title, outputs_dir)


if __name__ == '__main__':
    root = '/cs/labs/yweiss/ariel1/'
    root = '/mnt/storage_ssd/'
    data_root = f'{root}/data'
    cache_dir = f'{root}/big_files'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_dataset = 'laion2b_s34b_b79k'
    model_name = 'ViT-B-32'
    restrict_to_classes = ['cat', 'dog']
    # dataset_name = 'STL10'
    dataset_name = 'Flickr8k'

    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_dataset})")
    os.makedirs(outputs_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset,
                                                                 device=device, cache_dir=cache_dir)
    model.preprocess = preprocess
    model.eval()

    dataset = get_dataset(dataset_name, model.preprocess, data_root, restrict_to_classes)
    label_map = None
    if dataset_name == "STL10":
        label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"
    plot_2d(model, dataset, dataset_name, outputs_dir=outputs_dir, device=device, label_map=label_map)
