import torch
import torch.nn as nn
import numpy as np


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
        self.kernel = self.classifier.weight

    def forward(self, x):
        if self.use_bn_first:
            x = self.bn(x)
        feat = self.extractor(x)
        if self.embedding:
            return feat
        pred = self.classifier(feat)
        return pred