from torchvision import transforms

backbone = dict(
    type='DenseNet',
    depth=121,
    context_block_cfg=dict(ratio=1./4),
    pretrained=True,)

data = dict(
    dataset_path='/home1/liangjianming/rgb-recursion-cellular/train',
    datalist_path='/home1/liangjianming/rgb-recursion-cellular/train.csv',
    batch_size=32,
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
    loss_cfg=dict(
        loss_type='cross_entropy'),
    optimizer_cfg=dict(
        lr=0.01,
        weight_decay=0.0001,
        momentum=0.9),
    lr_cfg=dict(
        gamma=0.1,
        step=[60, 80]),
    accumulate_batch_size=256,
    mixup=False,
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/dense121/dense121_gc_bs256',
    log_file='logs.log',
    print_frequency=50,)

