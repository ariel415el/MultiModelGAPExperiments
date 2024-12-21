import os

import numpy as np
import open_clip
import torch
import clip
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dimensionality_recuction import pca
from utils import get_dataset, get_clip_features


def get_multimodal_pcs():
    dataset_name = 'STL10'
    dataset = get_dataset(dataset_name, model.preprocess, data_root)
    label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"

    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, _, mean = pca(np.concatenate((text_features, image_features)))

    return PCs, mean


class ClipClassifier:
    def __init__(self, model, reference_images, device):
        self.device = device
        with torch.no_grad():
            self.reference_features = model.encode_image(reference_images).float()
            self.reference_features /= self.reference_features.norm(dim=-1, keepdim=True)

    def predict(self, features, return_probs=False, **kwargs):
        logits = (features.to(self.device) @ self.reference_features.T).softmax(-1)
        if return_probs:
            return logits
        return logits.argmax(1)


def classify_stl():
    classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    datasets = {cls: get_dataset('STL10', model.preprocess, data_root, restrict_to_classes=[cls]) for cls in classes}

    ref_images = torch.stack([datasets[cls][0][0] for cls in classes]).to(device)
    for i in range(len(ref_images)):
        save_image(ref_images[i], f"{outputs_dir}/{i}.png", normalize=True)
    classifier = ClipClassifier(model, ref_images, device=device)

    text_tokens = clip.tokenize(['A photo of a cat', 'cat', 'dog', 'airplane']).to(device)
    with torch.no_grad():
        f = model.encode_text(text_tokens).float()
        f /= f.norm(dim=-1, keepdim=True)

    output = classifier.predict(f)

    print([classes[i] for i in output])


def classify_flickr():
    dataset = get_dataset('Flickr8k', model.preprocess, data_root)
    # dataset.return_all_texts = True
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    image_features = []
    text_features = []
    for images, texts in tqdm(dataloader):
        # text_batches = np.array(texts).T
        # text_tokens = clip.tokenize(texts.tolist()).to(device)
        text_tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            f = model.encode_text(text_tokens).float()
            f /= f.norm(dim=-1, keepdim=True)
            text_features += [f]

            f = model.encode_image(images.to(device)).float()
            f /= f.norm(dim=-1, keepdim=True)
            image_features += [f]

    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    image_features += text_features.mean(0) - image_features.mean(0)

    logits = (text_features @ image_features.T).softmax(-1)
    acc = (logits.argmax(-1) == torch.arange(len(image_features)).to(device)).float().mean()
    print(acc)

    i = 1
    img, gt_caption = dataset[i]
    save_image(img, "GT.png", normalize=True)
    print(f"GT {gt_caption}")
    for j in range(5):
        idx = logits[i].topk(5).indices[j].item()
        save_image(dataset[idx][0], f"{i}.png", normalize=True)
        print(f'{j}: {dataset[idx][1]}')



if __name__ == '__main__':
    root = '/cs/labs/yweiss/ariel1/'
    root = '/mnt/storage_ssd/'
    data_root = f'{root}/data'
    cache_dir = f'{root}/big_files'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'ViT-L-14'
    pretrained_datset = 'openai'

    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")
    os.makedirs(outputs_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_datset,
                                                                 device=device, cache_dir=cache_dir)
    model.preprocess = preprocess
    model.eval()

    classify_flickr()
