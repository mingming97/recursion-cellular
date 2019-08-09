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
        if not self.embedding:
            self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.extractor(x)
        if self.embedding:
            return feat
        pred = self.classifier(feat)
        return pred

