import os

import open_clip
import torch
import torchvision
import clip
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from dimensionality_recuction import *


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,images_dir, captions_file, transform=None):
        self.images_dir = images_dir
        self.data = defaultdict(list)
        for line in open(captions_file).readlines():
            line = line.strip().replace(" .", ".")
            i = line.find("#")
            img_name, text = line[:i], line[i:]
            if os.path.exists(os.path.join(self.images_dir, img_name)):
                text = text[2:].strip()
                self.data[img_name].append(text)
        self.data = list(self.data.items())
        self.transform = transform
        self.return_all_texts = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, texts = self.data[idx]
        img = Image.open(os.path.join(self.images_dir, img_name))
        img = self.transform(img)
        if self.return_all_texts:
            return img, texts
        return img, texts[0]


def extract_clip_embedding_from_dataset(model, dataset, label_map=None, device=torch.device('cpu')):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    all_text_features = []
    all_image_features = []
    all_labels = []
    pbar = tqdm(dataloader)
    pbar.set_description("Extracting features from data")
    for images, labels in pbar:
        if label_map is not None:
            text_descriptions = map(label_map, labels)
        else:
            text_descriptions = list(labels)
        text_tokens = clip.tokenize(text_descriptions).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            image_features = model.encode_image(images.to(device)).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        all_text_features += [text_features.cpu().numpy()]
        all_image_features += [image_features.cpu().numpy()]
        all_labels += [labels]

    all_text_features = np.concatenate(all_text_features)
    all_image_features = np.concatenate(all_image_features)
    all_labels = np.concatenate(all_labels)

    return all_text_features, all_image_features, all_labels


def get_clip_features(model, dataset, label_map, device, output_dir=None):
    text_features, image_features, labels = extract_clip_embedding_from_dataset(model, dataset, label_map, device)

    # if output_dir is not None and os.path.exists(os.path.join(output_dir, 'text_features.npy')):
    #     text_features = np.load(os.path.join(output_dir, 'text_features.npy'))
    #     image_features = np.load(os.path.join(output_dir, 'image_features.npy'))
    #     labels = np.load(os.path.join(output_dir, 'labels.npy'))
    # else:
    #     text_features, image_features, labels = extract_clip_embedding_from_dataset(model, dataset, label_map, device)
    #     if output_dir is not None:
    #         os.makedirs(output_dir, exist_ok=True)
    #         np.save(os.path.join(output_dir, 'text_features.npy'), text_features)
    #         np.save(os.path.join(output_dir, 'image_features.npy'), image_features)
    #         np.save(os.path.join(output_dir, 'labels.npy'), labels)
    return text_features, image_features, labels


def get_dataset(dataset_name, preprocess, data_root, restrict_to_classes=None):
    if dataset_name == 'Flickr8k':
        dataset = FlickrDataset(os.path.join(data_root, "Flickr8k/images"),
                                captions_file=os.path.join(data_root, 'Flickr8k/Flickr8k_text/Flickr8k.token.txt'),
                                transform=preprocess)
    elif dataset_name == 'STL10':
        dataset = torchvision.datasets.STL10(os.path.join(data_root, "STL10"), transform=preprocess, download=True, split='train')
        # test_dataset = torchvision.datasets.STL10(os.path.join(data_root, "STL10"), transform=preprocess, download=True, split='test')
        # dataset.data = np.concatenate((dataset.data, test_dataset.data))
        # dataset.labels = np.concatenate((dataset.labels, test_dataset.labels))
        if restrict_to_classes is not None:
            subset_idx = [dataset.classes.index(x) for x in restrict_to_classes]
            perm = np.argsort(subset_idx)
            subset_idx = [subset_idx[i] for i in perm]
            dataset.classes = [restrict_to_classes[i] for i in perm]
            filter = np.any([dataset.labels == x for x in subset_idx], axis=0)
            dataset.data = dataset.data[filter]
            dataset.labels = dataset.labels[filter]
            for new_idx, old_idx in enumerate(subset_idx):
                dataset.labels[dataset.labels == old_idx] = new_idx
    else:
        raise ValueError(f"No such dataset: {dataset_name}")

    return dataset

def print_models():
    models = defaultdict(list)
    for model, dataset in open_clip.list_pretrained():
        models[model].append(dataset)
    for model in models:
        print(f"{model} : {models[model]}")


if __name__ == '__main__':
    print_models()