# cache_dir='/cs/labs/yweiss/ariel1/big_files'
# data_root = '/cs/labs/yweiss/ariel1/data'
import os

import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from torch import optim
from torchvision.utils import save_image

from adversarial_examples import plot
from dimensionality_recuction import pca
from plot_2d import plot_per_class_embeddings
from utils import get_dataset, get_clip_features


def optimize_pc(model, img, PCs, mean, reg_lambda=100000, lr=0.001, n_steps=100, verbose=True):
    img = img.clone().to(device).unsqueeze(0)

    with torch.no_grad():
        features = model.encode_image(img).float()
        features = features / features.norm(dim=-1, keepdim=True)
        target = (features - mean) @ PCs

    img += torch.randn_like(img) * 1e-4
    img.requires_grad = True
    optimizer = optim.Adam([img], lr=lr)
    for i in range(n_steps):

        features = model.encode_image(img).float()
        features = features / features.norm(dim=-1, keepdim=True)
        pcs = (features - mean) @ PCs

        pc_loss = -torch.nn.MSELoss()(pcs[:, 0], target[:, 0])
        pc_reg = torch.nn.MSELoss()(pcs[:, 1:], target[:, 1:])
        loss = pc_loss + reg_lambda * pc_reg
        loss.backward()
        optimizer.step()
        # img.grad.zero_()

        if verbose or i == n_steps - 1:
            print(f"iter: {i}, pc_loss: {pc_loss}, pc_reg {pc_reg}")

    return (img).detach()

def get_pcs(model, PCs, mean, img):
    with torch.no_grad():
        f = model.encode_image(img).float()
        f /= f.norm(dim=-1, keepdim=True)
    return (f - mean) @ PCs

def plot(model, PCs, mean, reference_features, image_features,
    dataset_labels, class_names, input, output, save_path):
    cmap = plt.get_cmap('tab10')
    n = len(class_names)
    colors = [cmap(i / n) for i in range(n)]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    pc_1 = 0
    pc_2 = 1
    # plot data and labels
    label_pcs = ((reference_features - mean) @ PCs).cpu().numpy()
    image_pcs = ((image_features - mean) @ PCs).cpu().numpy()
    for label in np.unique(dataset_labels):
        idx = dataset_labels == label
        ax.scatter(image_pcs[idx, pc_1], image_pcs[idx, pc_2],
                   color=colors[-label - 1], s=15, alpha=0.25, marker='o')
        ax.scatter(label_pcs[label, pc_1], label_pcs[label, pc_2],
                   color=colors[-label - 1], label=class_names[label],
                   s=100, alpha=1, marker='x', linewidths=[4], edgecolors=['k'])

    input_pcs = get_pcs(model, PCs, mean, input.unsqueeze(0).to(device)).cpu().numpy()
    adv_pcs = get_pcs(model, PCs, mean, output.to(device)).cpu().numpy()
    dx, dy = adv_pcs[0, pc_1] - input_pcs[0, pc_1] , adv_pcs[0, pc_2] - input_pcs[0, pc_2]
    plt.arrow(input_pcs[0, pc_1], input_pcs[0, pc_2], dx, dy, head_width=0.005, color=colors[-label - 1])

    plt.xlabel(f"pc {pc_1}")
    plt.ylabel(f"pc {pc_2}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.clf()



if __name__ == '__main__':
    cache_dir = '/mnt/storage_ssd/big_files'
    data_root = '/mnt/storage_ssd/datasets'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 3
    model_name = 'ViT-B-32'
    pretrained_datset = 'laion2b_s34b_b79k'
    # model_name = 'RN50'
    # pretrained_datset = 'openai'
    restrict_to_classes = ['cat', 'dog']

    dataset_name = "STL10"
    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")

    # Load_model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_datset,
                                                                 device=device, cache_dir=cache_dir)
    model.preprocess = preprocess

    dataset = get_dataset(dataset_name, model.preprocess, data_root, restrict_to_classes)
    label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"

    # Extract CLIP features
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    # PCA
    PCs, _, mean = pca(np.concatenate((text_features, image_features)))

    text_embeddings = (text_features - mean) @ PCs
    image_embeddings = (image_features - mean) @ PCs

    PCs = torch.from_numpy(PCs).to(device)
    mean = torch.from_numpy(mean).to(device)

    img = dataset[0][0]
    output = optimize_pc(model, img, PCs, mean, reg_lambda=100, lr=0.1, n_steps=100, verbose=True)

    # plot_per_class_embeddings(text_embeddings, image_embeddings, labels, dataset.classes, 'asd', outputs_dir)
    plot(model, PCs, mean, torch.from_numpy(text_features).to(device), torch.from_numpy(image_features).to(device),
         labels, restrict_to_classes, img, output, "asd.png")
    print(output.shape)

    save_image(img, "input.png", normalize=True)
    save_image(output, "output.png", normalize=True)