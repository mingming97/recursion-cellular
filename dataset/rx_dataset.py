import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

import os


class RxDataset(data.Dataset):
    def __init__(self, img_dir, datalist, transform=None):
        super(RxDataset, self).__init__()
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
        ]) if transform is None else transform

        # datalist: a list like [('img_name', 'label'), ...]
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name, label = self.datalist[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        img = self.transform(img)
        if label is not None:
            label = torch.tensor(label)
        return img, label