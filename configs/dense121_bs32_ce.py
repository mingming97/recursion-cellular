from torchvision import transforms
# U2OS RPE HUVEC HEPG2
backbone = dict(
    type='DenseNet',
    depth=121,
    in_channel=6,
    context_block_cfg=None,
    pretrained=True,)

data = dict(
    dataset_path='/home1/liangjianming/recursion-cellular/train',
    train_datalist_path='./dataset/csv_files/HUVEC_split_train.csv',
    val_datalist_path='./dataset/csv_files/HUVEC_split_val.csv',
    data_mode='six_channels',
    batch_size=16,
    resize=(384, 384),
    normalize=dict(
        mean=[0.02645406, 0.05782261, 0.04123408, 0.04099084, 0.02156311, 0.03849946],
        std=[0.05697599, 0.05549077, 0.04151200, 0.05318175, 0.05224787, 0.03929300]),)

train = dict(
    epoch=40,
    metric_cfg=dict(
        type=['linear'],
        out_features=1108),
    loss_cfg=dict(
        type=['cross_entropy']),
    optimizer_cfg=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,),
    lr_cfg=dict(
        warmup='linear',
        warmup_iters=500,
        gamma=0.1,
        step=[25, 35]),
    accumulate_batch_size=32,
    load_from='./work_dir/dense121/dense121_bs32_ce/epoch_20.pth',
    checkpoint=None,)


log = dict(
    log_dir='./work_dir/dense121/dense121_bs32_ce_HUVEC',
    log_file='logs.log',
    val_frequency=1,
    save_frequency=10,
    print_frequency=50)