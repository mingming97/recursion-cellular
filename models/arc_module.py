import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=65, m=0.5, num_classes=1108):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s

        if self.training:
            labels_flip = labels +  self.num_classes // 2
            labels_flip = torch.remainder(labels_flip, self.num_classes)
            if labels_flip.dim() == 1:
                labels_flip = labels_flip.unsqueeze(-1)
            onehot = torch.zeros(outputs.size()).cuda()
            onehot.scatter_(1, labels_flip, 1)
            onehot_invert = (onehot == 0).float()
            assert onehot_invert.size() == outputs.size()
            outputs = outputs * onehot_invert - onehot_invert
        return outputs