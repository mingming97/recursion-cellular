import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AMSoftmaxLoss(nn.Module):
    def __init__(self, feat_dim, n_classes=1108, m=0.35, s=30):
        super(AMSoftmaxLoss, self).__init__()
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.m = m
        self.s = s
        self.kernel = nn.Parameter(torch.Tensor(n_classes, feat_dim))
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, target):
        feat_norm = F.normalize(feat, p=2, dim=1, eps=1e-10)
        kernel_norm = F.normalize(self.kernel, p=2, dim=1, eps=1e-10)
        cos_theta = feat_norm.matmul(kernel_norm.t())
        cos_theta = torch.clamp(cos_theta, -1, 1)
        onehot = torch.zeros(feat.size(0), self.n_classes).float().cuda()
        onehot.scatter_(1, target.view(-1, 1), self.m)
        logits = self.s * (cos_theta - onehot)
        loss = self.CE(logits, target)
        return loss