import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import *

import os


class Tester:

    def __init__(self, cfg, dataloader):
        self.dataloader = dataloader

        backbone = self.build_backbone(cfg['backbone'].copy())
        metric_fcs = self.build_metric_fcs(cfg['metric_fcs'].copy(), backbone.out_feat_dim)
        self.model = Classifier(backbone, metric_fcs).cuda()

        checkpoint = cfg['checkpoint']
        assert os.path.exists(checkpoint)
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict['model_params'])
        self.model.eval()

        self.results = []

    def build_backbone(self, backbone_cfg):
        backbone_type = backbone_cfg.pop('type')
        if backbone_type == 'ResNet':
            backbone = ResNet(**backbone_cfg)
        elif backbone_type == 'ResNeXt':
            backbone = ResNeXt(**backbone_cfg)
        elif backbone_type == 'DenseNet':
            backbone = DenseNet(**backbone_cfg)
        else:
            raise ValueError('Illegal backbone: {}'.format(backbone_type))
        return backbone

    def build_metric_fcs(self, metric_cfg, out_feat_dim):
        metric_types = metric_cfg.pop('type')
        metric_fcs = nn.ModuleList()
        for metric_type in metric_types:
            if metric_type == 'add_margin':
                metric_fc = AddMarginProduct(out_feat_dim, **metric_cfg)
            elif metric_type == 'arc_margin':
                metric_fc = ArcMarginProduct(out_feat_dim, **metric_cfg)
            elif metric_type == 'sphere':
                metric_fc = SphereProduct(out_feat_dim, **metric_cfg)
            elif metric_type == 'linear':
                metric_fc = nn.Linear(out_feat_dim, metric_cfg['out_features'])
            else:
                raise ValueError('Illegal metric_type: {}'.format(metric_type))
            metric_fcs.append(metric_fc)
        return metric_fcs

    def val_on_dataloader(self):
        total_sample, total_correct = 0, 0
        correct_dict = {k: 0 for k in range(1108)}
        with torch.no_grad():
            for data, label in tqdm(self.dataloader):
                data = data.cuda()

                output = self.model.forward_test(data)
                pred = output.argmax(dim=1).cpu()
                correct = pred == label

                total_sample += label.size(0)
                total_correct += correct.sum().item()

                correct_label = label[correct].numpy()
                for cl in correct_label:
                    num = correct_dict.get(cl, 0)
                    correct_dict[cl] = num + 1

        for k, v in correct_dict.items():
            print('class{} : {}/{}'.format(k, v, self.dataloader.dataset.num_dict[k]))

        return total_correct / total_sample


    def test_on_dataloader(self, datalist_path, outfile):
        submission = pd.read_csv(datalist_path)
        submission = submission[:len(submission)//2]

        with torch.no_grad():
            preds = np.empty(0)
            for data_s1, data_s2 in tqdm(self.dataloader):
                data_s1 = data_s1.cuda()
                data_s2 = data_s2.cuda()
                output_s1 = F.softmax(self.model.forward_test(data_s1), dim=1)
                output_s2 = F.softmax(self.model.forward_test(data_s2), dim=1)
                output = (output_s1 + output_s2) / 2
                idx = output.argmax(dim=1).cpu().numpy()
                preds = np.append(preds, idx, axis=0)

        submission['sirna'] = preds.astype(int)
        submission.to_csv(outfile, index=False, columns=['id_code', 'sirna'])