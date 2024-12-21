"""Code from https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L272"""
import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from dimensionality_recuction import pca
from utils import get_clip_features, get_dataset
import torch
import numpy as np


def compute_self_nearest_neighbors(feats, topk=1, pt=True):
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    if pt:
        with torch.no_grad():
            feats_pt = torch.from_numpy(feats).cuda()
            knn = (feats_pt @ feats_pt.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
            knn = knn.cpu().numpy()
    else:
        sim_mat = feats @ feats.T
        np.fill_diagonal(sim_mat, -1e8)
        knn = np.argsort(sim_mat, axis=1)[:, -topk:]
    return knn


def mutual_knn(feats_A, feats_B, topk):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A torch tensor of shape N x feat_dim
        feats_B: A torch tensor of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    knn_A = compute_self_nearest_neighbors(feats_A, topk)
    knn_B = compute_self_nearest_neighbors(feats_B, topk)

    acc = (knn_A == knn_B).sum(1) / topk
    return acc.mean().item()


def plot_text_to_image_alignment(model, dataset_name, outputs_dir, device):
    dataset, label_map = get_dataset(dataset_name, model.preprocess)

    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, eigv, mean = pca(np.concatenate((text_features, image_features)))
    text_embeddings = np.dot(text_features - mean, PCs)
    image_embeddings = np.dot(image_features - mean, PCs)

    lower_scores = []
    top_scores = []
    n = text_embeddings.shape[1]
    dropped_top_pcs = np.arange(0, n, n//10)
    for i in tqdm(dropped_top_pcs):
        score = mutual_knn(text_embeddings[:, i:], image_embeddings[:, i:], topk=10)
        lower_scores.append(score)
        score = mutual_knn(text_embeddings[:, :i], image_embeddings[:, :i], topk=10)
        top_scores.append(score)
    plt.plot(n - dropped_top_pcs, lower_scores, label="low x pcs", color='r')
    plt.plot(dropped_top_pcs, top_scores, label="top x pcs", color='b')
    plt.xlabel("# lower PCs")
    plt.ylabel("Aligmnent")
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, "pc_aligment.png"))
    plt.clf()


def measure_nn_alignment(model, dataset_name, outputs_dir, device):
    topk=10
    dataset, label_map = get_dataset(dataset_name, model.preprocess)

    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, eigv, mean = pca(np.concatenate((text_features, image_features)))
    text_embeddings = np.dot(text_features - mean, PCs)
    image_embeddings = np.dot(image_features - mean, PCs)
    all_embeddings = np.concatenate((text_embeddings, image_embeddings))
    h = len(text_embeddings)
    n = text_embeddings.shape[1]
    text_to_text_accs = []
    image_to_image_accs = []
    dropped_top_pcs = np.arange(0, n, n//10)
    n_lowest_pcs = []
    for i in tqdm(dropped_top_pcs):
        embs = all_embeddings[:, i:]
        nns = compute_self_nearest_neighbors(embs, topk=topk, pt=False)
        n_lowest_pcs.append(embs.shape[1])
        text_to_text_accs.append((nns[:h] < h).mean().item())
        image_to_image_accs.append((nns[h:] > h).mean().item())

    plt.plot(n_lowest_pcs, text_to_text_accs, label='text_to_text_accs', color='b')
    plt.plot(n_lowest_pcs, image_to_image_accs, label='image_to_image_accs', color='r')
    plt.xlabel("# lower PCs")
    plt.ylabel("intra domain alignment")
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, "domain_alignment.png"))
    plt.clf()