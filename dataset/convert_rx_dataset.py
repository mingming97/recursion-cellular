import os
import sys
import zipfile

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

sys.path.append('rxrx1-utils')
import rxrx.io as rio


ORI_DIR = '/home1/liangjianming/recursion-cellular'
BASE_DIR = '/home1/liangjianming/rgb-recursion-cellular'


def convert_to_rgb(df, split, resize=True, new_size=224, extension='jpeg'):
    N = df.shape[0]

    for i in tqdm(range(N)):
        code = df['id_code'][i]
        experiment = df['experiment'][i]
        plate = df['plate'][i]
        well = df['well'][i]

        for site in [1, 2]:
            save_path = os.path.join(BASE_DIR, '{}/{}_s{}.{}'.format(
                split, code, site, extension))

            im = rio.load_site_as_rgb(
                split, experiment, plate, well, site, 
                base_path=ORI_DIR
            )
            im = im.astype(np.uint8)
            im = Image.fromarray(im)
            
            if resize:
                im = im.resize((new_size, new_size), resample=Image.BILINEAR)
            
            im.save(save_path)


def build_new_df(df, extension='jpeg'):
    new_df = pd.concat([df, df])
    new_df['filename'] = pd.concat([
        df['id_code'].apply(lambda string: string + '_s1.{}'.format(extension)),
        df['id_code'].apply(lambda string: string + '_s2.{}'.format(extension))
    ])
    return new_df


if __name__ == '__main__':

    for folder in ['train', 'test']:
        dir_name = os.path.join(BASE_DIR, folder)
        if not os.path.exists(dir_name):
            os.makedirs(os.path.join(BASE_DIR, folder))

    train_df = pd.read_csv(os.path.join(ORI_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(ORI_DIR, 'test.csv'))

    new_train = build_new_df(train_df)
    new_test = build_new_df(test_df)

    new_train.to_csv(os.path.join(BASE_DIR, 'train.csv'), index=False)
    new_test.to_csv(os.path.join(BASE_DIR, 'test.csv'), index=False)

    convert_to_rgb(train_df, 'train', resize=False)
    convert_to_rgb(test_df, 'test', resize=False)




