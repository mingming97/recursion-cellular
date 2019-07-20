from torchvision import transforms

backbone = dict(
    type='ResNet',
    depth=50,
    stage_with_context_block=[False, True, True, True],
    context_block_cfg=dict(ratio=1./4),
    pretrained=True,)

data = dict(
    dataset_path='/home1/liangjianming/rgb-recursion-cellular/train',
    datalist_path='/home1/liangjianming/rgb-recursion-cellular/train.csv',
    batch_size=64,
    train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ]),
    test_transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ]),)

train = dict(
    epoch=100,
    lr=0.01,
    loss_weight=10,
    pairwise_confusion=True,
    weight_decay=0.0001,
    momentum=0.9,
    lr_cfg=dict(
        gamma=0.1,
        step=[60, 80]),
    accumulate_batch_size=256,
    mix_up=False,
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/res50/res50_gc_bs256_pcloss',
    log_file='logs.log',
    print_frequency=50,)

