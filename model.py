
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pathlib import Path
from typing import List

from tqdm.notebook import tqdm


# -----------------------------------------------
#               Basic blocks
# -----------------------------------------------
class REBNCONV(nn.Module):
    """ReLU + BN + 3x3 Conv (used throughout U²-Net)."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# -----------------------------------------------
#          Residual U-blocks (RSU)
# -----------------------------------------------

def _upsample_like(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Bilinear upsample `src` to the spatial size of `tgt`."""
    return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=False)


class _RSU_Base(nn.Module):
    """Base class for RSU blocks with variable depth."""
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.height = height
        # initial conv
        self.rebn_in = REBNCONV(in_ch, out_ch)

        # encoder convs
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        for _ in range(height - 1):
            self.enc.append(REBNCONV(out_ch if _ == 0 else mid_ch, mid_ch))
            self.pool.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # bottom conv (dilated)
        self.btm = REBNCONV(mid_ch, mid_ch, dilation=2)

        # decoder convs
        self.dec = nn.ModuleList()
        for _ in range(height - 1):
            self.dec.append(REBNCONV(mid_ch * 2, mid_ch))

        # final conv
        self.rebn_out = REBNCONV(mid_ch + out_ch, out_ch)

    def forward(self, x):
        x_in = self.rebn_in(x)

        # encoder
        enc_feats = []
        h = x_in
        for enc, pool in zip(self.enc, self.pool):
            h = enc(h)
            enc_feats.append(h)
            h = pool(h)

        # bottom
        h = self.btm(h)

        # decoder
        for idx, dec in enumerate(reversed(self.dec)):
            h = _upsample_like(h, enc_feats[-(idx + 1)])
            h = dec(torch.cat([h, enc_feats[-(idx + 1)]], dim=1))

        # output
        h = self.rebn_out(torch.cat([h, x_in], dim=1))
        return h + x_in  # residual


class RSU7(_RSU_Base):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__(height=7, in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch)


class RSU6(_RSU_Base):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__(height=6, in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch)


class RSU5(_RSU_Base):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__(height=5, in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch)


class RSU4(_RSU_Base):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__(height=4, in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch)


class RSU4F(nn.Module):
    """RSU4F: RSU block without pooling (all convolutions with dilation)."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebn_in = REBNCONV(in_ch, out_ch)

        self.enc1 = REBNCONV(out_ch, mid_ch)
        self.enc2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.enc3 = REBNCONV(mid_ch, mid_ch, dilation=4)

        self.dec1 = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch, dilation=1)

        self.rebn_out = REBNCONV(mid_ch + out_ch, out_ch)

    def forward(self, x):
        x_in = self.rebn_in(x)

        h1 = self.enc1(x_in)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)

        d1 = self.dec1(torch.cat([h3, h2], dim=1))
        d2 = self.dec2(torch.cat([d1, h1], dim=1))

        h = self.rebn_out(torch.cat([d2, x_in], dim=1))

        return h + x_in


# -----------------------------------------------
#                U²-Net Model
# -----------------------------------------------
class U2Net_Hierarchical(nn.Module):
    """
    U²-Net for hierarchical semantic segmentation.
    Args:
        num_classes (int): Number of output masks (channels) to predict.
    Input:
        RGB image tensor of shape (B, 3, H, W)
    Output:
        Tensor of shape (B, num_classes, H, W) -- hierarchical masks
    """
    def __init__(self, num_classes: int = 9, base_ch: int = 64):
        super().__init__()
        self.stage1 = RSU7(3, base_ch, base_ch)          # 3 -> 64
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(base_ch, base_ch, base_ch)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(base_ch, base_ch, base_ch)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(base_ch, base_ch, base_ch)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(base_ch, base_ch, base_ch)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(base_ch, base_ch, base_ch)

        # decoder
        self.stage5d = RSU4F(base_ch * 2, base_ch, base_ch)
        self.stage4d = RSU4(base_ch * 2, base_ch, base_ch)
        self.stage3d = RSU5(base_ch * 2, base_ch, base_ch)
        self.stage2d = RSU6(base_ch * 2, base_ch, base_ch)
        self.stage1d = RSU7(base_ch * 2, base_ch, base_ch)

        # side output convolutions (produce feature maps prior to final 1x1)
        self.side1 = nn.Conv2d(base_ch, num_classes, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(base_ch, num_classes, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(base_ch, num_classes, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(base_ch, num_classes, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(base_ch, num_classes, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(base_ch, num_classes, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(num_classes * 6, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        hx1 = self.stage1(x)    # (B, C, H, W)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # Decoder
        hx5d = self.stage5d(torch.cat([_upsample_like(hx6, hx5), hx5], dim=1))
        hx4d = self.stage4d(torch.cat([_upsample_like(hx5d, hx4), hx4], dim=1))
        hx3d = self.stage3d(torch.cat([_upsample_like(hx4d, hx3), hx3], dim=1))
        hx2d = self.stage2d(torch.cat([_upsample_like(hx3d, hx2), hx2], dim=1))
        hx1d = self.stage1d(torch.cat([_upsample_like(hx2d, hx1), hx1], dim=1))

        # Side outputs
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d1 = _upsample_like(d1, x)
        d2 = _upsample_like(d2, x)
        d3 = _upsample_like(d3, x)
        d4 = _upsample_like(d4, x)
        d5 = _upsample_like(d5, x)
        d6 = _upsample_like(d6, x)

        # Fusion
        d0 = self.out_conv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))

        # Output is a tensor (B, num_classes, H, W). For compatibility we also
        # return the side outputs (each num_classes channels).
        return d0, (d1, d2, d3, d4, d5, d6)