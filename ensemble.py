import torch
from torch.utils import data

from dataset import RxDataset
from dataset.utils.datalist import datalist_from_file
from tools import Ensembler
from utils import cfg_from_file

import argparse
import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Rx ArgumentParser')
    parser.add_argument('--config', default='config/ensemble_test.py', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = cfg_from_file(args.config)
    print('using config: {}'.format(args.config))

    data_cfg = cfg['data']
    datalist = datalist_from_file(data_cfg['datalist_path'])
    random.shuffle(datalist)
    num_train_files = len(datalist) // 5 * 4
    test_dataset = RxDataset(data_cfg['dataset_path'],
                               datalist[num_train_files:],
                               transform=data_cfg['test_transform'])
    test_dataloader = data.DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    ensembler = Ensembler(cfg['net_cfgs'], test_dataloader)
    score = ensembler.test_on_dataloader()
    print('validation score: {}'.format(score))

if __name__ == '__main__':
    main()