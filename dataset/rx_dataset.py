import torch
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import os


class RxDataset(data.Dataset):
    def __init__(self, img_dir, datalist, transform=None, data_mode='rgb'):
        super(RxDataset, self).__init__()
        assert data_mode in ('rgb', 'six_channels')
        self.img_dir = img_dir

        if data_mode == 'rgb':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ]) if transform is None else transform
        else:
            self.transform = transform

        # datalist: a list like [('img_name', 'label'), ...]
        self.datalist = datalist
        self.data_mode = data_mode

        self._cal_num_dict()

    def _cal_num_dict(self):
        self.num_dict = {k: 0 for k in range(1108)}
        for _, label in self.datalist:
            num = self.num_dict.get(label, 0)
            self.num_dict[label] = num + 1

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if self.data_mode == 'rgb':
            img_name, label = self.datalist[idx]
            img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            img = self.transform(img)
            if label is not None:
                label = torch.tensor(label)
        else:
            img_path, label = self.datalist[idx]
            imgs = []
            for i in range(1, 7):
                img_name = img_path + 'w{}.png'.format(i)
                img = Image.open(os.path.join(self.img_dir, img_name))
                imgs.append(img)
            
            if label is not None:
                label = torch.tensor(label)
            img = np.stack(imgs, axis=-1)
            img = self.transform(img)
        return img, label