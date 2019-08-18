import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class Classifier(nn.Module):

    def __init__(self, extractor, feat_dim, extra_module=None, num_classes=1108):
        super(Classifier, self).__init__()
        self.extractor = extractor
        self.extra_module = extra_module
        out_feat_dim = self.extra_module.in_features if self.extra_module is not None else num_classes
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # self.use_bn_first = self.extractor.in_channel == 6
        self.use_bn_first = False
        self.classifier = nn.Linear(feat_dim, out_feat_dim)
        if self.use_bn_first:
            self.bn = nn.BatchNorm2d(6)

    def forward(self, x, label=None):
        if self.use_bn_first:
            x = self.bn(x)
        feat = self.extractor(x)
        feat = self.classifier(feat)
        if self.extra_module is not None:
            feat = F.normalize(feat)
            if label is not None:
                return self.extra_module(feat, label)
        return feat

    def forward_test(self, x, center_feat=None):
        with torch.no_grad():
            if self.use_bn_first:
                x = self.bn(x)
            feat = self.extractor(x)
            feat = self.classifier(feat)
            if self.extra_module is not None:
                feat = F.normalize(feat)
                if center_feat is None:
                    return feat
                feat = feat.cpu().numpy()
                similarity = cosine_similarity(feat, center_feat)
                return similarity
        return feat
