from microsnoop.models.net_combine import *


class CNNNet(nn.Module):
    """
    refer to https://github.com/MouseLand/cellpose and https://github.com/cellimnet/scellseg-publish
    """

    def __init__(self, in_chans=1, out_chans=1, depths=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = CNN_encoder(in_chans=in_chans, depths=depths)
        self.decoder = CNN_decoder(depths=depths, out_chans=out_chans)

    def forward(self, imgs, mask_ratio=None):
        latent, mask = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent)  # [N, L, p*p*3]
        return pred, mask

