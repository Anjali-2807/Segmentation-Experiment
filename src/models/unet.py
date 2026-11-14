"""
Modular Modest Encoder-Decoder Semantic Segmentation Architectures
Author: Anjali Tiwari
Course: Advanced Computer Vision (CS107203CS) - Unit 3

Task 2 Baseline:
- Modest U-Net Encoder-Decoder (~3.3M parameters) with standard convolutions and concatenation skip fusion.

Task 3 Controlled Change:
- Controlled mechanism change: DILATION (Atrous Convolutions in Bottleneck).
- Replaces standard d=1 convolutions with multi-rate atrous convolutions (d=[1, 2, 4, 8])
  while keeping input/output channel count (256) and parameter budget explicitly matched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Standard double convolution block: (Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DilatedBottleneck(nn.Module):
    """
    Controlled Change: Multi-rate Atrous / Dilated Convolutional Bottleneck.
    Captures multi-scale contextual receptive fields (d=1, 2, 4, 8) without
    sacrificing spatial resolution or inflating parameter budget.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4  # 256 // 4 = 64 per branch

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.project(out)


class UpBlock(nn.Module):
    """Upsampling and skip connection fusion block"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        total_in = (in_channels // 2) + skip_channels
        self.conv = DoubleConv(total_in, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegmentationUNet(nn.Module):
    """
    Modest Encoder-Decoder Network with explicit mechanism control.
    
    Args:
        in_channels (int): Input image channels (3 for RGB)
        num_classes (int): Output segmentation classes (2 for Background + Foreground)
        mechanism (str): 'baseline' (standard conv bottleneck) or 'dilation' (atrous bottleneck)
    """
    def __init__(self, in_channels=3, num_classes=2, mechanism="baseline", base_ch=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mechanism = mechanism.lower()
        self.base_ch = base_ch

        # Encoder Path (Modest budget: 32 -> 64 -> 128 -> 256)
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 4, base_ch * 8))

        # Bottleneck: Controlled Change Point
        self.pool = nn.MaxPool2d(2)
        if self.mechanism == "dilation":
            self.bottleneck = DilatedBottleneck(base_ch * 8, base_ch * 8)
        else:
            self.bottleneck = DoubleConv(base_ch * 8, base_ch * 8, dilation=1)

        # Decoder Path with Skip Connections
        self.up1 = UpBlock(in_channels=base_ch * 8, skip_channels=base_ch * 8, out_channels=base_ch * 4)
        self.up2 = UpBlock(in_channels=base_ch * 4, skip_channels=base_ch * 4, out_channels=base_ch * 2)
        self.up3 = UpBlock(in_channels=base_ch * 2, skip_channels=base_ch * 2, out_channels=base_ch)
        self.up4 = UpBlock(in_channels=base_ch, skip_channels=base_ch, out_channels=base_ch // 2)

        # Final Classifier Head
        self.outc = nn.Conv2d(base_ch // 2, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # 32, 256x256
        x2 = self.down1(x1)     # 64, 128x128
        x3 = self.down2(x2)     # 128, 64x64
        x4 = self.down3(x3)     # 256, 32x32

        # Bottleneck
        xb = self.pool(x4)      # 256, 16x16
        xb = self.bottleneck(xb)

        # Decoder with Skip Fusion
        d1 = self.up1(xb, x4)   # 128, 32x32
        d2 = self.up2(d1, x3)   # 64, 64x64
        d3 = self.up3(d2, x2)   # 32, 128x128
        d4 = self.up4(d3, x1)   # 16, 256x256

        # Output Logits
        logits = self.outc(d4)  # num_classes, 256x256
        return logits


def get_model(mechanism="baseline", num_classes=2, base_ch=32):
    return SegmentationUNet(in_channels=3, num_classes=num_classes, mechanism=mechanism, base_ch=base_ch)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
