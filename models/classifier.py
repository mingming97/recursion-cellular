import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, extractor, feat_dim, num_classes=1108):
        super(Classifier, self).__init__()
        self.extractor = extractor
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # self.use_bn_first = self.extractor.in_channel == 6
        self.use_bn_first = False
        self.classifier = nn.Linear(feat_dim, num_classes)
        if self.use_bn_first:
            self.bn = nn.BatchNorm2d(6)

    def forward(self, x):
        if self.use_bn_first:
            x = self.bn(x)
        feat = self.extractor(x)
        pred = self.classifier(feat)
        return pred


class AMSoftmaxClassifier(nn.Module):
    def __init__(self, extractor, feat_dim, num_classes=1108):
        super(AMSoftmaxClassifier, self).__init__()
        self.extractor = extractor
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(feat_dim, feat_dim)
        self.weight = nn.Parameter(torch.Tensor(self.num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def forward(self, x):
        feat = self.extractor(x)
        feat = self.classifier(feat)
        cos_theta = self.cosine_sim(feat, self.weight)
        return cos_theta