from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

model_dict = {
    "resnet18": [resnet18, 512],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "resnet101": [resnet101, 2048],
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 1024, kernel_size=3, stride=1, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(2048, out_channels, kernel_size=3, stride=1, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = x.flatten(1)

        return x


class ReObjNet(nn.Module):
    """ResNet + ConvBlock"""

    def __init__(
        self,
        name: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet50",
        pretrained=True,
        feat_dim=4096,
    ):
        super(ReObjNet, self).__init__()
        model, hidden_dim = model_dict[name]

        # ResNet
        self.encoder = model(weights="IMAGENET1K_V1") if pretrained else model()
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # ConvBlock
        self.conv_block = ConvBlock(hidden_dim * 2, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, fg, bg):
        fg_emb = self.encoder(fg)
        bg_emb = self.encoder(bg)
        joint_emb = torch.cat([fg_emb, bg_emb], dim=1)
        joint_emb = self.conv_block(joint_emb)

        return joint_emb
