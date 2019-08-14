import torch
from torch.utils import data
import torch.nn as nn

from dataset import RxDataset
from dataset.utils.datalist import datalist_from_file
from models import *
from tools import Trainer
from utils import cfg_from_file

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Recursion Cellular Classification ArgumentParser')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # init configs
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

    # init backbone
    backbone_cfg = cfg['backbone'].copy()
    backbone_type = backbone_cfg.pop('type')
    if backbone_type == 'ResNet':
        backbone = ResNet(**backbone_cfg)
    elif backbone_type == 'ResNeXt':
        backbone = ResNeXt(**backbone_cfg)
    elif backbone_type == 'DenseNet':
        backbone = DenseNet(**backbone_cfg)

    # init loss criterion
    train_cfg, log_cfg = cfg['train'], cfg['log']
    loss_cfg = train_cfg['loss_cfg'].copy()
    loss_type = loss_cfg.pop('type')
    if loss_type == 'pairwise_confusion':
        print('using pairwise_confusion')
        classifier = Classifier(backbone, backbone.out_feat_dim, embedding=False)
        criterion = CrossEntropyWithPC(loss_cfg['loss_weight'])
    elif loss_type == 'cross_entropy':
        classifier = Classifier(backbone, backbone.out_feat_dim, embedding=False)
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'AM_softmax':
        classifier = AMSoftmaxClassifier(backbone, backbone.out_feat_dim)
        criterion = AMSoftmaxLoss(**loss_cfg)
    else:
        raise ValueError('Illegal loss_type: {}'.format(loss_type))
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    # init optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), **train_cfg['optimizer_cfg'])
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