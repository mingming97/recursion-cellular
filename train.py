import torch
from torch.utils import data
import torch.nn as nn

from dataset import RxDataset
from dataset.utils.datalist import datalist_from_file
from models import ResNet, ResNeXt, DenseNet, CrossEntropyWithPC, Classifier
from tools import Trainer
from utils import cfg_from_file

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Recursion Cellular Classification ArgumentParser')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = cfg_from_file(args.config)
    print('using config: {}'.format(args.config))

    data_cfg = cfg['data']
    datalist = datalist_from_file(data_cfg['datalist_path'], data_mode=data_cfg.get('data_mode', 'rgb'))
    num_train_files = len(datalist) // 10 * 9
    train_dataset = RxDataset(data_cfg['dataset_path'],
                              datalist[:num_train_files],
                              transform=data_cfg.get('train_transform', None),
                              data_mode=data_cfg.get('data_mode', 'rgb'))
    test_dataset = RxDataset(data_cfg['dataset_path'],
                             datalist[num_train_files:],
                             transform=data_cfg.get('test_transform', None),
                             data_mode=data_cfg.get('data_mode', 'rgb'))
    train_dataloader = data.DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=data_cfg['batch_size'])

    # from pprint import pprint
    # print('train num dict: ')
    # pprint(train_dataset.num_dict)
    # print('test num dict: ')
    # pprint(test_dataset.num_dict)

    backbone_cfg = cfg['backbone'].copy()
    backbone_type = backbone_cfg.pop('type')
    if backbone_type == 'ResNet':
        backbone = ResNet(**backbone_cfg)
    elif backbone_type == 'ResNeXt':
        backbone = ResNeXt(**backbone_cfg)
    elif backbone_type == 'DenseNet':
        backbone = DenseNet(**backbone_cfg)
    classifier = Classifier(backbone, backbone.out_feat_dim).cuda()

    train_cfg, log_cfg = cfg['train'], cfg['log']
    with_pairwise_confusion = train_cfg.get('pairwise_confusion', False)
    if with_pairwise_confusion:
        print('using pairwise_confusion')
        criterion = CrossEntropyWithPC(train_cfg['loss_weight'])
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=train_cfg['lr'],
                                weight_decay=train_cfg['weight_decay'], momentum=train_cfg['momentum'])
    trainer = Trainer(
        model=classifier, 
        train_dataloader=train_dataloader, 
        val_dataloader=test_dataloader,
        criterion=criterion, 
        optimizer=optimizer,
        train_cfg=train_cfg,
        log_cfg=log_cfg)
    trainer.train()


if __name__ == '__main__':
    main()
