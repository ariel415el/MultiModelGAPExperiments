import os
import open_clip
import torch
import numpy as np
from matplotlib import pyplot as plt

from dimensionality_recuction import pca
from utils import get_dataset, get_clip_features
from classify_images import ClipClassifier

from torch import optim
import torchvision


def optimize_image(model, img, gt_label, classifier, norm_coeff=0.01, lr=0.1, n_steps=100, pc_slice=None, verbose=False):
    y_gt = torch.tensor([gt_label]).cuda()
    img = img.clone().cuda()
    residue = torch.randn_like(img) * 1e-4
    residue.requires_grad = True
    optimizer = optim.Adam([residue], lr=lr)
    best_img = None
    best_norm = np.inf
    for i in range(n_steps):
        features = model.encode_image(img + residue).float()
        features = features / features.norm(dim=-1, keepdim=True)
        if i == 0:
            org_pred = classifier.predict(features).item()

        logits = classifier.predict(features, return_probs=True, slice=pc_slice)

        residue_norm = torch.norm(residue.view(len(residue), -1), dim=1).mean()

        loss = -torch.nn.functional.nll_loss(logits, y_gt) + norm_coeff * residue_norm
        loss.backward()
        # residue.grad = residue.grad.sign()
        optimizer.step()
        residue.grad.zero_()
        
        # with torch.no_grad():
        #     residue = torch.clamp(residue, -0.1, 0.1)
        #     # residue = residue / residue.norm() * 5
        # residue.requires_grad = True
        with torch.no_grad():
            features = model.encode_image(img + residue).float()
            features = features / features.norm(dim=-1, keepdim=True)
            cur_pred = classifier.predict(features).item()
        if residue_norm.item() < best_norm and cur_pred != org_pred:
            best_img = (img + residue).detach()
            best_norm = residue_norm.item()


        if verbose or i  == n_steps - 1:
            print(f"iter: {i}, loss: {loss}, best norm: {best_norm}, norm: {residue_norm}, pred {org_pred}->{cur_pred}")

    return best_img, best_norm


def test_average_adversarial_margin(model, dataset, classifier, n_images=10):
    all_inputs = []
    all_advs = []
    all_norms = []
    gt_labels = []

    for i in range(n_images):
        input, gt_label = dataset[i]
        input = input.unsqueeze(0)
        # target_label = np.random.choice([x for x in range(len(dataset.classes)) if x != gt_label])
        adv, adv_residue_norm = optimize_image(model, input, gt_label, classifier,
                                               norm_coeff=0.01, lr=0.01, n_steps=500, pc_slice=None, verbose=False)
        # assert adv is not None
        if adv is not None:
            all_norms.append(adv_residue_norm)
            all_inputs.append(input)
            all_advs.append(adv)
            gt_labels.append(gt_label)
    return all_inputs, all_advs, all_norms, gt_labels


def get_pcs(model, PCs, mean, img):
    with torch.no_grad():
        f = model.encode_image(img.cuda()).float().cpu()
        f /= f.norm(dim=-1, keepdim=True)
    return (f - mean) @ PCs


def plot(model, PCs, mean, reference_features, image_features,
    dataset_labels, class_names, inputs, advs, gt_labels, save_path):
    cmap = plt.get_cmap('tab10')
    n = len(class_names)
    colors = [cmap(i / n) for i in range(n)]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    pc_1 = 0
    pc_2 = 1
    # plot data and labels
    label_pcs = (reference_features - mean) @ PCs
    image_pcs = (image_features - mean) @ PCs
    for label in np.unique(dataset_labels):
        idx = dataset_labels == label
        ax.scatter(image_pcs[idx, pc_1], image_pcs[idx, pc_2],
                   color=colors[-label - 1], s=15, alpha=0.25, marker='o')
        ax.scatter(label_pcs[label, pc_1], label_pcs[label, pc_2],
                   color=colors[-label - 1], label=class_names[label],
                   s=100, alpha=1, marker='x', linewidths=[4], edgecolors=['k'])

    for input, adv, gt_label in zip(inputs, advs, gt_labels):
        input_pcs = get_pcs(model, PCs, mean, input)
        adv_pcs = get_pcs(model, PCs, mean, adv)
        dx, dy = adv_pcs[0, pc_1] - input_pcs[0, pc_1] , adv_pcs[0, pc_2] - input_pcs[0, pc_2]
        plt.arrow(input_pcs[0, pc_1], input_pcs[0, pc_2], dx, dy, head_width=0.005, color=colors[-gt_label - 1])

    plt.xlabel(f"pc {pc_1}")
    plt.ylabel(f"pc {pc_2}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.clf()


def plot_embedding_residue(model, PCs, mean, inputs, advs, save_path):
    fig, axis = plt.subplots(len(inputs), 1, figsize=(8, 8))
    for i,(input, adv )in enumerate(zip(inputs, advs)):
        input_pcs = get_pcs(model, PCs, mean, input)
        adv_pcs = get_pcs(model, PCs, mean, adv)
        diff = (input_pcs - adv_pcs)[0]
        axis[i].bar(np.arange(len(diff)), diff.abs())
    plt.xlabel("PCs")
    plt.ylabel("Coeff")
    plt.savefig(save_path)
    plt.show()
    plt.clf()


def main():
    # cache_dir='/cs/labs/yweiss/ariel1/big_files'
    # data_root = '/cs/labs/yweiss/ariel1/data'
    cache_dir='/mnt/storage_ssd/big_files'
    data_root = '/mnt/storage_ssd/datasets'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 3
    model_name = 'ViT-B-32'
    pretrained_datset = 'laion2b_s34b_b79k'
    # model_name = 'RN50'
    # pretrained_datset = 'openai'

    dataset_name = "STL10"
    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")

    # Load_model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_datset,
                                                                 device=device, cache_dir=cache_dir)
    model.preprocess = preprocess

    dataset = get_dataset(dataset_name, model.preprocess, data_root, restrict_to_classes=['cat', 'dog'])
    label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"

    # Extract CLIP features
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    # PCA
    PCs, _, mean = pca(np.concatenate((text_features, image_features)))
    PCs, mean = torch.from_numpy(PCs), torch.from_numpy(mean)
    image_features = torch.from_numpy(image_features)

    # Shift text labels into class image means
    classifier = ClipClassifier(model, [label_map(dataset.classes.index(label)) for label in dataset.classes], device)

    inputs, advs, all_norms, gt_labels = test_average_adversarial_margin(model, dataset, classifier, n_images=n_samples)
    plot_embedding_residue(model, PCs, mean, inputs, advs, "residue_with_gap.png")
    torchvision.utils.save_image(torch.cat(inputs), "inputs.png", normalize=True)
    torchvision.utils.save_image(torch.cat(advs), "advs_with_gap.png", normalize=True)
    plot(model, PCs, mean, classifier.reference_features.cpu(), image_features,
         labels, dataset.classes, inputs, advs, gt_labels, "With_gap.png")
    n_with_gap = len(inputs)
    gap_mean, gap_std = np.mean(all_norms), np.std(all_norms)

    for i in range(len(dataset.classes)):
        classifier.reference_features[i] = image_features[labels==i].mean(0)

    inputs, advs, all_norms, gt_labels = test_average_adversarial_margin(model, dataset, classifier, n_images=n_samples)
    plot_embedding_residue(model, PCs, mean, inputs, advs, "residue_no_gap.png")
    torchvision.utils.save_image(torch.cat(advs), "advs_no_gap.png", normalize=True)
    plot(model, PCs, mean, classifier.reference_features.cpu(), image_features,
         labels, dataset.classes, inputs, advs, gt_labels,"no_gap.png")


    no_gap_mean, no_gap_std = np.mean(all_norms), np.std(all_norms)
    n_no_gap = len(inputs)
    print(f"With gap: #sucess {n_no_gap}/{n_samples} avg adversarial norm {gap_mean:.2f}+-{gap_std:.2f}")
    print(f"No gap: #sucess {n_no_gap}/{n_samples} avg adversarial norm {no_gap_mean:.2f}+-{no_gap_std:.2f}")

if __name__ == '__main__':
    main()
