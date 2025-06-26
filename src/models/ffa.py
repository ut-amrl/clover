import torch
import torch.nn as nn


class FFA(nn.Module):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.net = net
        self.similarity_type = "cosine"

    def forward(self, batch, **kwargs):
        images = [image.cuda(non_blocking=True) for image in [img[0] for img in batch["images"]]]
        masks = [mask.cuda(non_blocking=True) for mask in [img[1] for img in batch["images"]]]

        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)

        return self.net(images, masks)
