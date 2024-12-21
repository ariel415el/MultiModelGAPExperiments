import os

import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import clip
from tqdm import tqdm

from dimensionality_recuction import pca, get_pcas
from plot_2d import plot_per_class_embeddings
from text_synthesis.gpt import templates
from utils import get_dataset, get_clip_features


class ClipClassifier:
    def __init__(self, model, reference_texts, device):
        self.device = device
        text_tokens = clip.tokenize(reference_texts).to(self.device)
        with torch.no_grad():
            self.reference_features = model.encode_text(text_tokens).float()
            self.reference_features /= self.reference_features.norm(dim=-1, keepdim=True)

    def predict(self, features, return_probs=False, **kwargs):
        logits = (features.to(self.device) @ self.reference_features.T).softmax(-1)
        if return_probs:
            return logits
        return logits.argmax(1)


def plot_classification(model, n_images=100, outputs_dir='outputs', device=torch.device('cpu')):
    dataset = get_dataset('STL10', model.preprocess, data_root, restrict_to_classes=restrict_to_classes)
    classifier = ClipClassifier(model, [f"This is a photo of a {label}" for label in dataset.classes], device)

    images, labels = next(iter(DataLoader(dataset, batch_size=n_images, shuffle=True)))
    with torch.no_grad():
        image_features = model.encode_image(images.to(device)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    text_probs = 100.0 * classifier.predict(image_features, return_probs=True)

    top_probs, top_labels = text_probs.cpu().topk(min(5,text_probs.size(-1)), dim=-1)
    accuracy = (text_probs.argmax(-1).cpu() == labels).float().mean()

    print(f"Accuracy: {(accuracy*100).int()}%")

    plt.figure(figsize=(16, 16))
    plt.title(f"Accuracy: {(accuracy*100).int()}%")
    for i in range(8):
        plt.subplot(4, 4, 2 * i + 1)

        img = images[i]
        img = img - img.min()
        img = img / img.max()
        img = img.permute(1,2,0).cpu().numpy()

        plt.imshow(img)
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [dataset.classes[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(os.path.join(outputs_dir, "classify_STL.png"))
    plt.clf()


def compare_prompts(n_images):
    for template in templates[:15]:
        classifier = ClipClassifier(model, [template.replace('{object}', label) for label in dataset.classes], device)

        images, labels = next(iter(DataLoader(dataset, batch_size=n_images, shuffle=True)))

        with torch.no_grad():
            image_features = model.encode_image(images.to(device)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = classifier.predict(image_features).cpu()
        accuracy = (text_probs == labels).float().mean()

        print(f"{template}: Accuracy: {(accuracy*100).int()}%")


def plot_classifier(classifier, class_names, image_features, all_labels, PCs, mean, title):
    new_reference_embs = (classifier.reference_features.cpu().numpy() - mean) @ PCs
    image_embeddings = (image_features - mean) @ PCs
    cmap = plt.get_cmap('tab10')
    labels = range(len(classifier.reference_features))
    colors = [cmap(i) for i in labels]
    for label in labels:
        idx = all_labels == label
        plt.scatter(new_reference_embs[label, 0], new_reference_embs[label, 1],
                   color=colors[label], label=class_names[label], s=100, alpha=0.5, marker='x')
        plt.scatter(image_embeddings[idx, 0], image_embeddings[idx, 1],
                   color=colors[label], s=50, alpha=0.5, marker='o')

    # plot linear separator
    a,b = new_reference_embs[0, :2]
    c,d = new_reference_embs[1, :2]
    plt.plot([a,c], [b,d])
    y = lambda x: (b+d)/2 - (x-(a+c)/2)*(c-a)/(d-b)
    minv = min(image_embeddings[:, 0].min(), new_reference_embs[:, 0].min())
    maxv = max(image_embeddings[:, 0].max(), new_reference_embs[:, 0].max())
    xs =np.array([minv, maxv])
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot(xs, y(xs), linestyle='--', color='black', label='classifier')
    # Keep xticks and yticks of same size to see perpeendicular lines
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f"classifier_vis.png"))
    plt.show()
    plt.clf()


def l2_similarity(X, Y):
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return -dist


def test_reference_shift():
    pc_shift = 0.15
    dataset_name = 'STL10'
    dataset = get_dataset(dataset_name, model.preprocess, data_root, restrict_to_classes=['cat', 'dog'])
    label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"
    text_features, image_features, all_labels = get_clip_features(model, dataset, label_map, device,
                                                                  os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, _, mean = pca(np.concatenate((text_features, image_features)))

    classifier = ClipClassifier(model, dataset.classes, device)

    # Use different text templates to cause problem
    # classifier = ClipClassifier(model, [f'This is a photo if a {dataset.classes[0]}', f'Presenting a {dataset.classes[1]}'], device)

    # Move one of the reference vectors manualy
    classifier.reference_features[1, :] += torch.from_numpy(PCs[:, 0] * pc_shift).to(device)

    # Clssify using l2 because reference vectors are no longer with 1 norm
    logits = l2_similarity(image_features, classifier.reference_features.cpu().numpy())

    # Classify using first two PCs only:
    # logits = l2_similarity((image_features - mean) @ PCs[:, :2], (classifier.reference_features.cpu().numpy() - mean) @ PCs[:, :2])

    acc = (logits.argmax(-1) == all_labels)

    plot_classifier(classifier, dataset.classes, image_features, all_labels, PCs, mean, f"shift pc0: ({pc_shift}) Acc: {acc.sum()}/{len(acc)}")
    print(f"Accuracy: {acc.sum()}/{len(acc)}")


def classify_pcs(model, outputs_dir='outputs', device=torch.device('cpu')):
    dataset_name = "STL10"
    dataset, label_map = get_dataset(dataset_name, model.preprocess)
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, eigv, mean = pca(np.concatenate((text_features, image_features)))
    text_embeddings = np.dot(text_features - mean, PCs)
    image_embeddings = np.dot(image_features - mean, PCs)
    classifier = ClipClassifier(model, dataset)
    label_embeddings = np.dot(classifier.labe_features.cpu().numpy() - mean, PCs)

    n = text_embeddings.shape[1]
    dropped_top_pcs = np.arange(0, n)
    accuracies_lower = []
    accuracies_top = []
    for i in tqdm(dropped_top_pcs):
        # text_probs = np.(100.0 * image_embeddings[:, i:] @ text_embeddings[:, i:].T).softmax(dim=-1)
        text_probs = image_embeddings[:, i:] @ label_embeddings[:, i:].T
        acc = (text_probs.argmax(-1) == labels).mean()
        accuracies_lower.append(acc)

        text_probs = image_embeddings[:, :i] @ label_embeddings[:, :i].T
        acc = (text_probs.argmax(-1) == labels).mean()
        accuracies_top.append(acc)

    print(f"accuracies_lower: {accuracies_lower}")
    print(f"accuracies_top: {accuracies_top}")
    plt.plot(n - dropped_top_pcs, accuracies_lower, label="low x pcs", color='r')
    plt.plot(dropped_top_pcs, accuracies_top, label="top x pcs", color='b')
    plt.legend()
    plt.xlabel("# PCs")
    plt.ylabel("Aligmnent")
    plt.savefig(os.path.join(outputs_dir, "classify_pcs.png"))
    plt.clf()


if __name__ == '__main__':
    root = '/cs/labs/yweiss/ariel1/'
    root = '/mnt/storage_ssd/'
    data_root = f'{root}/data'
    cache_dir = f'{root}/big_files'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_dataset = 'laion2b_s34b_b79k'
    model_name = 'ViT-B-32'
    restrict_to_classes = ['cat', 'dog']
    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_dataset})")
    os.makedirs(outputs_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset,
                                                                 device=device, cache_dir=cache_dir)
    model.preprocess = preprocess
    model.eval()

    # plot_classification(model, n_images=100, outputs_dir=outputs_dir, device=device)

    # compare_prompts(n_images=1000)

    test_reference_shift()

    # classify_pcs(model, outputs_dir, device, outputs_dir=outputs_dir, device=device)