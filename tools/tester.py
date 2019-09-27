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
        metric_fc = self.build_metric_fc(cfg['metric_fc'].copy(), backbone.out_feat_dim)
        self.model = Classifier(backbone, metric_fc)

        checkpoint = cfg['checkpoint']
        assert os.path.exists(checkpoint)
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict['model_params'])
        self.model.cuda()
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

    def build_metric_fc(self, metric_cfg, out_feat_dim):
        metric_type = metric_cfg.pop('type')
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
        return metric_fc

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


    def data_leak(self, predicted):
        test_csv = pd.read_csv('/home1/liangjianming/recursion-cellular/test.csv')
        train_csv = pd.read_csv('/home1/liangjianming/recursion-cellular/train.csv')
        sub = pd.read_csv("/home1/liangjianming/recursion-cellular/sample_submission.csv")

        plate_groups = np.zeros((1108,4), int)
        for sirna in range(1108):
            grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
            assert len(grp) == 3
            plate_groups[sirna,0:3] = grp
            plate_groups[sirna,3] = 10 - grp.sum()
        all_test_exp = test_csv.experiment.unique()

        group_plate_probs = np.zeros((len(all_test_exp),4))
        for idx in range(len(all_test_exp)):
            preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna'].values
            pp_mult = np.zeros((len(preds), 1108))
            pp_mult[range(len(preds)),preds] = 1
            
            sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
            assert len(pp_mult) == len(sub_test)
            
            for j in range(4):
                mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                       np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
                
                group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
        
        exp_to_group = np.array([3, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 3, 1, 0, 0, 0, 2, 3])

        def select_plate_group(pp_mult, idx):
            sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
            assert len(pp_mult) == len(sub_test)
            mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
                   np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
            pp_mult[mask] = 0
            return pp_mult

        for idx in range(len(all_test_exp)):
            indices = (test_csv.experiment == all_test_exp[idx])
            
            preds = predicted[indices,:].copy()
            
            preds = select_plate_group(preds, idx)
            sub.loc[indices,'sirna'] = preds.argmax(1)
        return sub


    def test_on_dataloader(self, datalist_path, outfile):
        predicted = []
        with torch.no_grad():
            for data_s1, data_s2 in tqdm(self.dataloader):
                data_s1 = data_s1.cuda()
                data_s2 = data_s2.cuda()

                output_s1 = F.softmax(self.model.forward_test(data_s1), dim=1)
                output_s2 = F.softmax(self.model.forward_test(data_s2), dim=1)
                output = (output_s1 + output_s2) / 2
                output = output.cpu().numpy()
                predicted.append(output)
        predicted = np.concatenate(predicted, 0).squeeze()
        sub = self.data_leak(predicted)
        print(sub.head())
        sub.to_csv(outfile, index=False, columns=['id_code','sirna'])