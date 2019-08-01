import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import ResNet, ResNeXt, DenseNet, Classifier

import os


class Ensembler:

    def __init__(self, net_cfgs, dataloader):
        self.dataloader = dataloader
        self.net_list = []
        for cfg in net_cfgs:
            backbone_cfg = cfg.copy()
            backbone_type = backbone_cfg.pop('type')
            checkpoint = backbone_cfg.pop('checkpoint')

            if backbone_type == 'ResNet':
                backbone = ResNet(**backbone_cfg)
            elif backbone_type == 'ResNeXt':
                backbone = ResNeXt(**backbone_cfg)
            elif backbone_type == 'DenseNet':
                backbone = DenseNet(**backbone_cfg)
            classifier = Classifier(backbone, backbone.out_feat_dim).cuda()

            assert os.path.exists(checkpoint)
            state_dict = torch.load(checkpoint)
            classifier.load_state_dict(state_dict['model_params'])
            classifier.eval()
            self.net_list.append(classifier)
        self.results = []


    def inference(self, imgs):
        preds = []
        with torch.no_grad():
            for net in self.net_list:
                pred = net(imgs)
                pred = F.softmax(pred.cpu(), dim=1)
                preds.append(pred)
        res = sum(preds) / len(preds)
        return res


    def val_on_dataloader(self):
        total_sample, total_correct = 0, 0
        correct_dict = {k: 0 for k in range(1108)}
        with torch.no_grad():
            for data, label in tqdm(self.dataloader):
                data = data.cuda()

                output = self.inference(data)
                pred = output.argmax(dim=1)
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
                output_s1 = self.inference(data_s1)
                output_s2 = self.inference(data_s2)
                output = (output_s1 + output_s2) / 2
                idx = output.argmax(dim=1).numpy()
                preds = np.append(preds, idx, axis=0)

        submission['sirna'] = preds.astype(int)
        submission.to_csv(outfile, index=False, columns=['id_code', 'sirna'])

