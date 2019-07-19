import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseConfusion(nn.Module):

    def __init__(self):
        super(PairwiseConfusion, self).__init__()

    def forward(self, logits, target):
        batch_size = logits.size(0)
        if batch_size % 2 != 0:
            raise Exception('Incorrect batch size provided.')
        prob = F.softmax(logits, dim=1)
        prob_left = prob[:int(batch_size / 2)]
        prob_right = prob[int(batch_size / 2):]

        target_left = target[:int(batch_size / 2)]
        target_right =target[int(batch_size / 2):]

        target_mask = torch.eq(target_left, target_right)
        target_mask = 1 - target_mask
        target_mask = target_mask.type_as(prob)
        num = target_mask.sum()

        loss = torch.norm((prob_left - prob_right).abs(), 2, 1) * target_mask
        loss = loss.sum() / num
        
        return loss


class CrossEntropyWithPC(nn.Module):

    def __init__(self, weight, ce_weight=None):
        super(CrossEntropyWithPC, self).__init__()
        self.pc = PairwiseConfusion()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.weight = weight

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pc_loss = self.pc(pred, target)
        return ce_loss + self.weight * pc_loss
