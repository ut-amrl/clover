from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

model_dict = {
    "vit_b_16": [vit_b_16, 768],
    "vit_b_32": [vit_b_32, 768],
    "vit_l_16": [vit_l_16, 1024],
    "vit_l_32": [vit_l_32, 1024],
}


class WDISINet(nn.Module):
    def __init__(
        self,
        name: Literal["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"] = "vit_b_16",
        pretrained: bool = True,
        num_heads: int = 8,
        img_size: int = 224,
    ):
        super(WDISINet, self).__init__()
        model, hidden_dim = model_dict[name]
        N = img_size // 16
        self.feat_dim = 2 * (N**2 + 1) * hidden_dim

        self.vit = (
            model(image_size=img_size, weights="IMAGENET1K_V1")
            if pretrained
            else model(image_size=img_size)
        )

        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, fg, bg):
        fg_emb = self.vit_encode(fg)
        bg_emb = self.vit_encode(bg)
        joint_emb = torch.cat((fg_emb, bg_emb), dim=1)

        final_emb, _ = self.self_attention(
            joint_emb, joint_emb, joint_emb, need_weights=False
        )
        return final_emb.flatten(start_dim=1)

    def vit_encode(self, img):
        embedding = None

        def get_embedding(module, input, output):
            nonlocal embedding
            embedding = output

        handle = self.vit.encoder.layers[-1].register_forward_hook(get_embedding)
        self.vit(img)
        handle.remove()
        return embedding
