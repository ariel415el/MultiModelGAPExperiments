import os
from collections import defaultdict

import open_clip
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alignment import plot_text_to_image_alignment, measure_nn_alignment
from classify_images import classify_stl, classify_pcs
from plot_2d import plot_2d
from plot_eigenvalues import plot_eiegen_spectrum


def print_models():
    models = defaultdict(list)
    for model, dataset in open_clip.list_pretrained():
        models[model].append(dataset)
    for model in models:
        print(f"{model} : {models[model]}")

if __name__ == '__main__':
    print_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'ViT-B-32'
    pretrained_datset = 'laion2b_s34b_b79k'
    # model_name = 'RN50'
    # pretrained_datset = 'openai'
    # model_name = 'ViT-B-16'
    # pretrained_datset = 'laion400m_e31'

    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")
    os.makedirs(outputs_dir, exist_ok=True)

    # model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_datset,
    #                                                              device=device, cache_dir='/cs/labs/yweiss/ariel1/big_files')
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
    model.preprocess = preprocess

    common_params = {'outputs_dir':outputs_dir, 'device':device}
    plot_2d(model, dataset_name='Flickr8k', pca_mode='image', **common_params)
    # plot_2d(model, dataset_name='Flickr8k', pca_mode='multimodal', **common_params)
    # plot_2d(model, dataset_name='STL10', pca_mode='multimodal', **common_params)
    # plot_eiegen_spectrum(model, dataset_name='Flickr8k', **common_params)
    # plot_text_to_image_alignment(model, dataset_name='Flickr8k', **common_params)
    # measure_nn_alignment(model, dataset_name='Flickr8k', **common_params)
    # classify_stl(model, n_images=100, **common_params)
    # classify_pcs(model, outputs_dir, device)
