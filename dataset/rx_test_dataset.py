import torch
from torch.utils import data
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

import os


class RxTestDataset(data.Dataset):
    def __init__(self, img_dir, datalist, transform=None, data_mode='rgb', normalize=None, resize=None):
        super(RxTestDataset, self).__init__()
        assert data_mode in ('rgb', 'six_channels')
        assert normalize is None or isinstance(normalize, dict)

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
        assert len(datalist) % 2 == 0
        self.datalist_s1 = datalist[:len(datalist) // 2]
        self.datalist_s2 = datalist[len(datalist) // 2:]
        self.data_mode = data_mode

        self.normalize = normalize
        self.resize = resize
        self._cal_num_dict()

    def _cal_num_dict(self):
        self.num_dict = {k: 0 for k in range(1108)}
        for _, label in self.datalist_s1:
            num = self.num_dict.get(label, 0)
            self.num_dict[label] = num + 1

    def __len__(self):
        return len(self.datalist_s1)

    def _get_one_pic(self, site, idx):
        if site == 1:
            img_name, _ = self.datalist_s1[idx]
        else:
            img_name, _ = self.datalist_s2[idx]
        if self.data_mode == 'rgb':
            imgs = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            imgs = self.transform(imgs)
        else:
            imgs = []
            for i in range(1, 7):
                img_full_name = img_name + 'w{}.png'.format(i)
                img = Image.open(os.path.join(self.img_dir, img_full_name))
                img = self._single_channel_transform(img, i-1)
                imgs.append(img)
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def __getitem__(self, idx):
        imgs_s1 = self._get_one_pic(1, idx)
        imgs_s2 = self._get_one_pic(2, idx)
        return imgs_s1, imgs_s2

    def _single_channel_transform(self, img, channels_index):
        if self.resize is not None:
            img = F.resize(img, self.resize)
        img = F.to_tensor(img).squeeze()
        if self.normalize is not None:
            img.sub_(self.normalize['mean'][channels_index]).div_(self.normalize['std'][channels_index])
        return img