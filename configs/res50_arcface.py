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
    batch_size=32,
    resize=(384, 384),
    normalize=dict(
        mean=[0.02645406, 0.05782261, 0.04123408, 0.04099084, 0.02156311, 0.03849946],
        std=[0.05697599, 0.05549077, 0.04151200, 0.05318175, 0.05224787, 0.03929300]))

train = dict(
    epoch=60,
    metric_cfg=dict(
        type=['linear', 'arc_margin'],
        out_features=1108,
        s=10,
        m=0.5),
    loss_cfg=dict(
        type=['cross_entropy', 'cross_entropy'],
        loss_weight=[1.0, 0.1]),
    optimizer_cfg=dict(
        type='SGD',
        lr=0.01,
        weight_decay=0.0001,
        momentum=0.9),
    lr_cfg=dict(
        warmup='linear',
        warmup_iters=500,
        gamma=0.1,
        step=[30, 50]),
    accumulate_batch_size=-1,
    load_from=None,
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/res50/res50_arcface',
    log_file='logs.log',
    val_frequency=1,
    save_frequency=10,
    print_frequency=50)