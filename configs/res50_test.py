from torchvision import transforms

net_cfg = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channel=6,
        context_block_cfg=None,
        pretrained=False,),
    metric_fcs=dict(
        type=['linear', 'arc_margin'],
        out_features=1108,
        s=10,
        m=0.5),
    checkpoint='./best_model.pth')

data = dict(
    dataset_path='/home1/liangjianming/recursion-cellular/test',
    datalist_path='/home1/liangjianming/rgb-recursion-cellular/test.csv',
    data_mode='six_channels',
    batch_size=64,
    resize=(384, 384),
    normalize=dict(
        mean=[0.02645406, 0.05782261, 0.04123408, 0.04099084, 0.02156311, 0.03849946],
        std=[0.05697599, 0.05549077, 0.04151200, 0.05318175, 0.05224787, 0.03929300]))

outfile = 'res50.csv'