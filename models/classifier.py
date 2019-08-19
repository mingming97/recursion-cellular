import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class Classifier(nn.Module):

    def __init__(self, extractor, feat_dim, extra_module=None, pre_layers=None, num_classes=1108):
        super(Classifier, self).__init__()
        self.pre_layers = pre_layers
        self.extractor = extractor
        self.extra_module = extra_module
        out_feat_dim = self.extra_module.in_features if self.extra_module is not None else num_classes
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(feat_dim, out_feat_dim)

    def forward(self, x, label=None):
        if self.pre_layers is not None:
            x = self.pre_layers(x)
        feat = self.extractor(x)
        feat = self.classifier(feat)
        if self.extra_module is not None:
            feat = F.normalize(feat)
            if label is not None:
                return self.extra_module(feat, label)
        return feat

    def forward_test(self, x, center_feat=None):
        with torch.no_grad():
            if self.pre_layers is not None:
                x = self.pre_layers(x)
            feat = self.extractor(x)
            feat = self.classifier(feat)
            if self.extra_module is not None:
                feat = F.normalize(feat)
                if center_feat is None:
                    return feat
                feat = feat.cpu().numpy()
                similarity = cosine_similarity(feat, center_feat)
                return torch.from_numpy(similarity)
        return feat