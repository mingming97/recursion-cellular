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


class AMSoftmaxClassifier(nn.Module):
    def __init__(self, extractor, feat_dim, num_classes=1108):
        super(AMSoftmaxClassifier, self).__init__()
        self.extractor = extractor
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(feat_dim, 512)
        self.weight = nn.Parameter(torch.Tensor(self.num_classes, 512))
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

    def forward_center(self, x):
        return self.classifier(self.extractor(x))

    def forward_test(self, x, center_feat):
        with torch.no_grad():
            feat = self.extractor(x)
            feat = self.classifier(feat)
        feat = feat.cpu().numpy()
        similarity = cosine_similarity(feat, center_feat)
        return similarity