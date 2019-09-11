from torchvision import transforms

net_cfg = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channel=6,
        context_block_cfg=None,
        pretrained=False,),
    metric_fc=dict(
        type='linear',
        out_features=1108),
    checkpoint='./work_dir/res50/res50_ce/best_model.pth')

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

outfile = 'res50.csv'