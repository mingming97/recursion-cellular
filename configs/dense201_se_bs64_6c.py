from torchvision import transforms

backbone = dict(
    type='DenseNet',
    depth=201,
    in_channel=6,
    context_block_cfg=dict(
        ratio=1./16, 
        context_modeling_type='avg',
        fusion_type='mul'),
    pretrained=True,)

data = dict(
    dataset_path='/home1/liangjianming/recursion-cellular/train',
    datalist_path='/home1/liangjianming/rgb-recursion-cellular/train.csv',
    data_mode='six_channels',
    batch_size=4,
    train_transform=transforms.ToTensor(),
    test_transform=transforms.ToTensor(),)

train = dict(
    epoch=60,
    lr=0.02,
    weight_decay=0.0001,
    momentum=0.9,
    lr_cfg=dict(
        gamma=0.1,
        step=[30, 50]),
    accumulate_batch_size=64,
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/dense201/dense201_se_bs64',
    log_file='logs.log',
    print_frequency=50,
    save_frequency=10)

