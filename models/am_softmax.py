import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AMSoftmaxLoss(nn.Module):
    def __init__(self, n_classes=1108, m=0.35, s=30):
        super(AMSoftmaxLoss, self).__init__()
        self.n_classes = n_classes
        self.m = m
        self.s = s
        self.CE = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        onehot = torch.zeros_like(pred).cuda()
        onehot.scatter_(1, target.view(-1, 1), 1.0)
        logits = self.s * (pred - onehot * self.m)
        loss = self.CE(logits, target)
        return loss