import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, extractor, feat_dim, num_classes=1108, embedding=False):
        super(Classifier, self).__init__()
        self.extractor = extractor
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.embedding = embedding
        # self.use_bn_first = self.extractor.in_channel == 6
        self.use_bn_first = False
        if not self.embedding:
            self.classifier = nn.Linear(feat_dim, num_classes)
        if self.use_bn_first:
            self.bn = nn.BatchNorm2d(6)

    def forward(self, x):
        if self.use_bn_first:
            x = self.bn(x)
        feat = self.extractor(x)
        if self.embedding:
            return feat
        pred = self.classifier(feat)
        return pred


class AMSoftmaxClassifier(nn.Module):
    def __init__(self, extractor, feat_dim, num_classes=1108):
        super(AMSoftmaxClassifier, self).__init__()
        self.extractor = extractor
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.Tensor(self.num_classes, feat_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        feat = self.extractor(x)

        feat_norm = F.normalize(feat, p=2, dim=1, eps=1e-10)
        kernel_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-10)
        cos_theta = feat_norm.matmul(kernel_norm.t())
        return cos_theta
