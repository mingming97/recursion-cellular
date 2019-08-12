import torch
from torch.utils import data

from dataset import RxDataset, RxTestDataset
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
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = cfg_from_file(args.config)
    print('using config: {}'.format(args.config))

    data_cfg = cfg['data']
    datalist = datalist_from_file(data_cfg['datalist_path'], data_mode=data_cfg.get('data_mode', 'rgb'))

    if args.mode == 'val':
        num_train_files = len(datalist) // 10 * 9
        dataset = RxDataset(data_cfg['dataset_path'],
                            datalist[num_train_files:],
                            transform=data_cfg['test_transform'],
                            data_mode=data_cfg.get('data_mode', 'rgb'))
    else:
        dataset = RxTestDataset(data_cfg['dataset_path'],
                                datalist,
                                transform=data_cfg['test_transform'],
                                data_mode=data_cfg.get('data_mode', 'rgb'))

    dataloader = data.DataLoader(dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    ensembler = Ensembler(cfg['net_cfgs'], dataloader)
    if args.mode == 'val':
        score = ensembler.val_on_dataloader()
        print('validation score: {}'.format(score))
    else:
        outdir = 'csv_output'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outfile = os.path.join(outdir, cfg.get('outfile', 'submission.csv'))
        ensembler.test_on_dataloader(datalist_path=data_cfg['datalist_path'], outfile=outfile)

if __name__ == '__main__':
    main()