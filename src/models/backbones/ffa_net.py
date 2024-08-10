"""
Paper: Are These the Same Apple? Comparing Images Based on Object Intrinsics
https://github.com/s-tian/CUTE
"""

from typing import Literal
import torch
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    """https://github.com/filipradenovic/cnnimageretrieval-pytorch"""
    "x: BS x num tokens x embed_dim"
    return torch.mean(x.clamp(min=eps).pow(p), dim=(1, 2)).pow(1.0 / p)


class FFANet(torch.nn.Module):
    def __init__(
        self,
        name: Literal[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
            "dino_vits16",
            "dino_vits8",
            "dino_vitb16",
            "dino_vitb8",
        ],
        variant: Literal["Crop-Feat", "Crop-Img"],
        **kwargs
    ):
        """
        :param device: string or torch.device object to run the model on.
        :param carvekit_object_type: object type for foreground segmentation. Can be "object" or "hairs-like".
        We find that "object" works well for most images in the CUTE dataset as well as vehicle ReID.
        """
        super().__init__()
        if name.startswith("dinov2"):
            self.encoder = torch.hub.load("facebookresearch/dinov2", name)
            self.patch_size = 14
        elif name.startswith("dino"):
            self.encoder = torch.hub.load("facebookresearch/dino:main", name)
            self.patch_size = 16 if "16" in name else 8
        else:
            raise ValueError("Invalid model name.")

        self.name = name
        self.feat_dim = self.encoder.embed_dim
        self.variant = variant

    def forward_features(self, img):
        if self.name.startswith("dinov2"):
            features = self.encoder.forward_features(img)
            return features["x_norm_patchtokens"]
        elif self.name.startswith("dino"):
            features = self.encoder.get_intermediate_layers(img)[0]
            return features[:, 1:, :]  # remove the first token (CLS token)

    def forward(self, img, mask, **kwargs):
        B, C, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        H_grid, W_grid = H // self.patch_size, W // self.patch_size

        with torch.no_grad():
            if self.variant == "Crop-Feat":
                emb = self.forward_features(img)
            elif self.variant == "Crop-Img":
                cropped_img = img * mask
                emb = self.forward_features(cropped_img)
            else:
                raise ValueError(
                    "Invalid variant, only Crop-Feat and Crop-Img are supported."
                )

            grid = emb.view(B, H_grid, W_grid, self.feat_dim)
            resized_mask = F.interpolate(
                mask.float(), size=(H_grid, W_grid), mode="nearest"
            ).long()

            cropped_features = grid * resized_mask.permute(0, 2, 3, 1)
            return torch.mean(cropped_features, dim=(1, 2))
