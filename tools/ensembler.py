import os
import torch
from models import ResNet, ResNeXt, DenseNet, Classifier
import torch.nn.functional as F


class Ensembler:

    def __init__(self, net_cfgs, test_dataloader):
        self.test_dataloader = test_dataloader
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


    def inference(self, imgs):
        preds = []
        with torch.no_grad():
            for net in self.net_list:
                pred = net(imgs)
                pred = F.softmax(pred.cpu(), dim=1)
                preds.append(pred)
        res = sum(preds) / len(preds)
        return res


    def test_on_dataloader(self):
        total_sample, total_correct = 0, 0
        correct_dict = {k: 0 for k in range(1108)}
        with torch.no_grad():
            for data, label in self.test_dataloader:
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
            print('class{} : {}/{}'.format(k, v, self.test_dataloader.dataset.num_dict[k]))

        return total_correct / total_sample