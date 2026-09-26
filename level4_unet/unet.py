import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    U-Net 中最基本的卷积模块：

    Conv
    → BatchNorm
    → ReLU
    → Conv
    → BatchNorm
    → ReLU

    padding=1 保证 3×3 卷积前后 H/W 不变。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
    ):
        super().__init__()

        # ====================================================
        # Encoder
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            32,
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            32,
            64,
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            64,
            128,
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(
            128,
            256,
        )

        self.pool4 = nn.MaxPool2d(2)

        # ====================================================
        # Bottleneck
        # ====================================================

        self.bottleneck = DoubleConv(
            256,
            512,
        )

        # ====================================================
        # Decoder 4
        #
        # 512 channels
        # 16×16
        #
        # →
        #
        # 256 channels
        # 32×32
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )

        # up4 256 channels
        # +
        # enc4 256 channels
        #
        # cat → 512 channels
        self.dec4 = DoubleConv(
            512,
            256,
        )

        # ====================================================
        # Decoder 3
        # ====================================================

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            256,
            128,
        )

        # ====================================================
        # Decoder 2
        # ====================================================

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            128,
            64,
        )

        # ====================================================
        # Decoder 1
        # ====================================================

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            64,
            32,
        )

        # ====================================================
        # Output Layer
        #
        # [B,32,H,W]
        # →
        # [B,1,H,W]
        # ====================================================

        self.output_conv = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        # ====================================================
        # Encoder
        # ====================================================

        # [B,3,256,256]
        # →
        # [B,32,256,256]
        e1 = self.enc1(x)

        # →
        # [B,32,128,128]
        p1 = self.pool1(e1)

        # →
        # [B,64,128,128]
        e2 = self.enc2(p1)

        # →
        # [B,64,64,64]
        p2 = self.pool2(e2)

        # →
        # [B,128,64,64]
        e3 = self.enc3(p2)

        # →
        # [B,128,32,32]
        p3 = self.pool3(e3)

        # →
        # [B,256,32,32]
        e4 = self.enc4(p3)

        # →
        # [B,256,16,16]
        p4 = self.pool4(e4)

        # ====================================================
        # Bottleneck
        # ====================================================

        # →
        # [B,512,16,16]
        b = self.bottleneck(p4)

        # ====================================================
        # Decoder
        # ====================================================

        # [B,512,16,16]
        # →
        # [B,256,32,32]
        d4 = self.up4(b)

        # Skip Connection
        #
        # [B,256,32,32]
        # +
        # [B,256,32,32]
        #
        # →
        # [B,512,32,32]
        d4 = torch.cat(
            [d4, e4],
            dim=1,
        )

        # →
        # [B,256,32,32]
        d4 = self.dec4(d4)

        # →
        # [B,128,64,64]
        d3 = self.up3(d4)

        # →
        # [B,256,64,64]
        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        # →
        # [B,128,64,64]
        d3 = self.dec3(d3)

        # →
        # [B,64,128,128]
        d2 = self.up2(d3)

        # →
        # [B,128,128,128]
        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        # →
        # [B,64,128,128]
        d2 = self.dec2(d2)

        # →
        # [B,32,256,256]
        d1 = self.up1(d2)

        # →
        # [B,64,256,256]
        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        # →
        # [B,32,256,256]
        d1 = self.dec1(d1)

        # ====================================================
        # Output
        # ====================================================

        # →
        # [B,1,256,256]
        output = self.output_conv(d1)

        # Target 位于 [0,1]，
        # 因此将模型输出限制到 [0,1]。
        output = torch.sigmoid(output)

        return output