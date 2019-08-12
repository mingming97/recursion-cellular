from torchvision import transforms

net_cfgs = [
    dict(
        type='DenseNet',
        depth=201,
        in_channel=6,
        context_block_cfg=None,
        pretrained=False,
        checkpoint='work_dir/dense201/dense201_bs64_6c/epoch_40.pth'),
]

data = dict(
    dataset_path='/home1/liangjianming/recursion-cellular/test',
    datalist_path='/home1/liangjianming/rgb-recursion-cellular/test.csv',
    data_mode='six_channels',
    batch_size=32,
    test_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.02645871, 0.05782903, 0.04122592, 0.04099488, 0.02156705, 0.03849198],
                                     std=[0.03121084, 0.04773749, 0.02298717, 0.0307236 , 0.01843595, 0.02129923])
                ]),)

outfile = 'dense201_6c_epoch_40.csv'