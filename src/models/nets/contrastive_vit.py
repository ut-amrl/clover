from loguru import logger

import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal

from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

model_dict = {
    "vit_b_16": [vit_b_16, 768],
    "vit_b_32": [vit_b_32, 768],
    "vit_l_16": [vit_l_16, 1024],
    "vit_l_32": [vit_l_32, 1024],
}


class ContrastiveViT(nn.Module):
    """Vision Transformer + Projection Head for Contrastive Learning"""

    def __init__(
        self,
        name: Literal["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"] = "vit_b_16",
        pretrained: bool = True,
        in_channels: int = 3,
        image_size: int = 224,
        head: Literal["linear", "mlp"] = "mlp",
        proj_dim: int = 128,
        **kwargs,
    ):
        super(ContrastiveViT, self).__init__()

        # Backbone
        model, self.feat_dim = model_dict[name]
        self.encoder = (
            model(image_size=image_size, weights="IMAGENET1K_V1")
            if pretrained
            else model(image_size=image_size)
        )
        self.encoder.heads = nn.Identity()

        # Change the input channel
        if in_channels != 3:
            logger.info(f"Changing the input channels to {in_channels}")
            self.encoder.conv_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.encoder.hidden_dim,
                kernel_size=self.encoder.patch_size,
                stride=self.encoder.patch_size,
            )

        # Projection head
        if head == "linear":
            self.head = nn.Linear(self.feat_dim, proj_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, proj_dim),
            )
        else:
            raise NotImplementedError(f"Head not supported: {head}")

    def forward(self, x, use_head=True):
        features = self.encoder(x)

        if use_head:
            return F.normalize(self.head(features), dim=1)

        return features


if __name__ == "__main__":
    model = ContrastiveViT(name="vit_b_16", in_channels=3)

    image = torch.randn(2, 3, 224, 224)
    mask = torch.randint(0, 2, (2, 1, 224, 224))

    output = model(image, mask)
    print(output.shape)
