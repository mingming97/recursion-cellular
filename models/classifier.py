import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, extractor, metric_fc, pre_layers=None):
        super(Classifier, self).__init__()
        self.pre_layers = pre_layers
        self.extractor = extractor
        self.metric_fc = metric_fc

    def forward(self, x):
        if self.pre_layers is not None:
            x = self.pre_layers(x)
        feat = self.extractor(x)
        output = self.metric_fc(feat)
        return output

    def losses(self, output, label, criterion):
        return criterion(output, label)

    def forward_test(self, x):
        if self.pre_layers is not None:
            x = self.pre_layers(x)
        feat = self.extractor(x)
        output = self.metric_fc(feat)
        return output