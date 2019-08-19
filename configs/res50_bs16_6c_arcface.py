from torchvision import transforms

backbone = dict(
    type='ResNet',
    depth=50,
    in_channel=6,
    context_block_cfg=None,
    pretrained=True,)

data = dict(
    dataset_path='/home1/liangjianming/recursion-cellular/train',
    datalist_path='/home1/liangjianming/rgb-recursion-cellular/train.csv',
    data_mode='six_channels',
    batch_size=16,
    resize=(384, 384),
    normalize=dict(
        mean=[0.02645871, 0.05782903, 0.04122592, 0.04099488, 0.02156705, 0.03849198],
        std=[0.03121084, 0.04773749, 0.02298717, 0.0307236, 0.01843595, 0.02129923]),)

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
    log_dir='./work_dir/res50/res50_bs16_arcface',
    log_file='logs.log',
    val_frequency=5,
    save_frequency=10,
    print_frequency=50)