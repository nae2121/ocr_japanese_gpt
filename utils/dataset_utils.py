import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABELS = [
    "玄関",
    "バルコニー",
    "浴室",
    "トイレ",
    "収納",
    "洋室",
    "クローゼット",
    "廊下",
    "ホール",
    "和室",
]
label_to_idx = {label: i for i, label in enumerate(LABELS)}
idx_to_label = {i: label for label, i in label_to_idx.items()}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, transform=None, with_label=True):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )
        )
        self.with_label = with_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)

        if self.with_label:
            filename = os.path.basename(path)
            try:
                label_name = filename.split("_")[0]
                label = label_to_idx[label_name]
            except (IndexError, KeyError):
                raise ValueError(f"ファイル名 '{filename}' からラベルが抽出できませんでした。")
            return img, label
        else:
            return img, os.path.basename(path)
