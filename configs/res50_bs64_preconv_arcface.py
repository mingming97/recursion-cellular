from torchvision import transforms
from torch import nn

backbone = dict(
    type='ResNet',
    depth=50,
    in_channel=3,
    context_block_cfg=None,
    pretrained=True,)

data = dict(
    dataset_path=r'F:\xuhuanqiang\kaggle_recursion_cellular\recursion-cellular-dataset\train',
    datalist_path=r'F:\xuhuanqiang\kaggle_recursion_cellular\rgb-recursion-cellular-dataset\train.csv',
    data_mode='six_channels',
    batch_size=16,
    resize=(384, 384),
    # normalize=([0.5 for _ in range(6)], [1 for _ in range(6)]),
    pre_layers=nn.Sequential(
        nn.Conv2d(6, 3, 3, 1, 1),
        nn.BatchNorm2d(3)
    ), )

train = dict(
    epoch=60,
    loss_cfg=dict(
        type='Arc_Face',
        in_features=512,
        out_features=1108,
        m=0.5,
        s=60),
    optimizer_cfg=dict(
        lr=0.005,
        weight_decay=0.0001,
        momentum=0.9),
    lr_cfg=dict(
        warmup='linear',
        warmup_iters=500,
        gamma=0.1,
        step=[30, 50]),
    accumulate_batch_size=-1,
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/res50/res50_bs64_preconv_arcface',
    log_file='logs.log',
    val_frequency=5,
    save_frequency=10,
    print_frequency=50, )

