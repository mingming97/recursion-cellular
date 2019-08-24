import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class Classifier(nn.Module):

    def __init__(self, extractor, metric_fcs, pre_layers=None):
        super(Classifier, self).__init__()
        assert len(metric_fcs) == 1 or isinstance(metric_fcs[0], nn.Linear)
        self.pre_layers = pre_layers
        self.extractor = extractor
        self.metric_fcs = metric_fcs

    def forward(self, x, label):
        if self.pre_layers is not None:
            x = self.pre_layers(x)
        feat = self.extractor(x)
        outputs = []
        for i, metric_fc in enumerate(self.metric_fcs):
            if i == 0:
                output = metric_fc(feat)
            else:
                output = metric_fc(feat, label)
            outputs.append(output)
        return outputs

    def loss(self, outputs, label, criterion):
        losses = [criterion(output, label) for output in outputs]
        return sum(losses) / len(losses)

    def forward_test(self, x):
        if self.pre_layers is not None:
            x = self.pre_layers(x)
        feat = self.extractor(x)
        return self.metric_fcs[0](feat)
