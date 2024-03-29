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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Recursion Cellular Classification ArgumentParser')
    parser.add_argument('--config', type=str, default=None, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # init configs
    cfg = cfg_from_file(args.config)
    print('using config: {}'.format(args.config))

    data_cfg = cfg['data']
    train_datalist = datalist_from_file(data_cfg['train_datalist_path'], data_mode=data_cfg.get('data_mode', 'rgb'))
    val_datalist = datalist_from_file(data_cfg['val_datalist_path'], data_mode=data_cfg.get('data_mode', 'rgb'))
    print('train data number:{}'.format(len(train_datalist)))
    print('val data number:{}'.format(len(val_datalist)))


    train_dataset = RxDataset(data_cfg['dataset_path'],
                              train_datalist,
                              transform=data_cfg.get('train_transform', None),
                              data_mode=data_cfg.get('data_mode', 'rgb'))
    test_dataset = RxDataset(data_cfg['dataset_path'],
                             val_datalist,
                             transform=data_cfg.get('test_transform', None),
                             data_mode=data_cfg.get('data_mode', 'rgb'))
    train_dataloader = data.DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=data_cfg['batch_size'])
    pre_layers = data_cfg.get('pre_layers', None)

    # init backbone
    backbone_cfg = cfg['backbone'].copy()
    backbone_type = backbone_cfg.pop('type')
    if backbone_type == 'ResNet':
        backbone = ResNet(**backbone_cfg)
    elif backbone_type == 'ResNeXt':
        backbone = ResNeXt(**backbone_cfg)
    elif backbone_type == 'DenseNet':
        backbone = DenseNet(**backbone_cfg)
    else:
        raise ValueError('Illegal backbone_type: {}'.format(backbone_type))

    # init metric_fc
    train_cfg, log_cfg = cfg['train'], cfg['log']
    metric_cfg = train_cfg['metric_cfg'].copy()
    metric_type = metric_cfg.pop('type')
    if metric_type == 'add_margin':
        metric_fc = AddMarginProduct(backbone.out_feat_dim, **metric_cfg)
    elif metric_type == 'arc_margin':
        metric_fc = ArcMarginProduct(backbone.out_feat_dim, **metric_cfg)
    elif metric_type == 'sphere':
        metric_fc = SphereProduct(backbone.out_feat_dim, **metric_cfg)
    elif metric_type == 'linear':
        metric_fc = nn.Linear(backbone.out_feat_dim, metric_cfg['out_features'])
    else:
        raise ValueError('Illegal metric_type: {}'.format(metric_type))
    classifier = Classifier(backbone, metric_fc, pre_layers)

    # init loss criterion
    loss_cfg = train_cfg['loss_cfg'].copy()
    loss_type = loss_cfg.pop('type')
    if loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'focal_loss':
        criterion = FocalLoss()
    else:
        raise ValueError('Illegal loss_type: {}'.format(loss_type))

    # init optimizer
    optimizer_cfg = train_cfg['optimizer_cfg'].copy()
    optimizer_type = optimizer_cfg.pop('type')
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), **optimizer_cfg)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), **optimizer_cfg)
    else:
        raise ValueError('Illegal optimizer_type:{}'.format(optimizer_type))

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