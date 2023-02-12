from microsnoop.models.net_blocks import *
from scellseg.net_utils import downsample as CNN_downsample
from scellseg.net_utils import upsample as CNN_upsample


class CNN_encoder(nn.Module):
    def __init__(self, in_chans=1, depths=[32, 64, 128, 256], sz=3, residual_on=True):
        super().__init__()
        nbase = [in_chans] + depths
        self.downsample = CNN_downsample(nbase, sz=sz, residual_on=residual_on)

    def forward(self, x, mask_ratio=None):
        embeddings = self.downsample(x)
        mask = None  # 为了和其他模型格式统一
        return embeddings, mask

class CNN_decoder(nn.Module):
    def __init__(self, depths=[32, 64, 128, 256], out_chans=1, sz=3, residual_on=True, concatenation=False):
        super().__init__()
        self.depths = depths
        nbase = depths + [depths[-1]]
        self.upsample = CNN_upsample(nbase, sz=sz, residual_on=residual_on, concatenation=concatenation,
                                     style_channels=[depths[-1], depths[-1], depths[-1], depths[-1]])
        self.make_style = makeStyle()
        self.base_bn = nn.BatchNorm2d(depths[0], eps=1e-5)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_conv = nn.Conv2d(depths[0], out_chans, 1, padding=1 // 2)

    def forward(self, embeddings):
        styles = [self.make_style(embeddings[-1]) for _ in range(len(self.depths))]
        y = self.upsample(styles, embeddings)
        y = self.base_conv(self.base_relu(self.base_bn(y)))
        return y
