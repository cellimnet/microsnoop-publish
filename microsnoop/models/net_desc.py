"""
设计模型计算图，包括
    __init__: 计算图的构建
    forward: 计算图流动
    load_model: 读取模型
    save_model: 保存模型
    加上其他附加功能的函数
"""
from microsnoop.models.net_combine import *


class CNNNet(nn.Module):
    """
    References:
        MAE: https://github.com/facebookresearch/mae
        Uformer: https://github.com/ZhendongWang6/Uformer
    """

    def __init__(self, in_chans=1, out_chans=1, depths=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = CNN_encoder(in_chans=in_chans, depths=depths)
        self.decoder = CNN_decoder(depths=depths, out_chans=out_chans)
        # self.apply(self._init_weights)

    def forward(self, imgs, mask_ratio=None):
        latent, mask = self.encoder(imgs, mask_ratio)  # mask_ratio影响的就是采样率，可以设置为0，不采样
        pred = self.decoder(latent)  # [N, L, p*p*3]
        return pred, mask

