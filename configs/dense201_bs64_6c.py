from torchvision import transforms

backbone = dict(
    type='DenseNet',
    depth=201,
    in_channel=6,
    context_block_cfg=None,
    pretrained=True,)

data = dict(
    dataset_path=r'F:\xuhuanqiang\kaggle_recursion_cellular\recursion-cellular-dataset\train',
    datalist_path=r'F:\xuhuanqiang\kaggle_recursion_cellular\rgb-recursion-cellular-dataset\train.csv',
    data_mode='six_channels',
    batch_size=8,
    train_transform=transforms.ToTensor(),
    test_transform=transforms.ToTensor(),
)

train = dict(
    epoch=100,
    lr=0.01,
    weight_decay=0.0001,
    momentum=0.9,
    lr_cfg=dict(
        gamma=0.1,
        step=[60, 80]),
    accumulate_batch_size=64,
    mixup=False,
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/dense201/dense201_bs64',
    log_file='logs.log',
    print_frequency=50,)

