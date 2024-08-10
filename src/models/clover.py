from typing import Literal
from tqdm import tqdm

import torch
import torch.nn as nn

from src.losses import SupConLoss, TripletLoss

from matplotlib import pyplot as plt


class CLOVER(nn.Module):
    def __init__(self, net, criterion, optimizer, scheduler=None, **kwargs):
        super(CLOVER, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer(params=net.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer) if scheduler else None
        self.similarity_type = "cosine"

    def forward(self, batch, mode: Literal["train", "eval"] = "train"):
        images = [img.cuda(non_blocking=True) for img in batch["images"]]
        images = torch.cat(images, dim=0)

        features = self.net(images, use_head=(mode == "train"))

        return features

    def train_step(self, batch):
        self.net.train()

        bsz = batch["label"].size(0)

        if isinstance(self.criterion, SupConLoss):
            features = self.forward(batch)
            labels = batch["label"].cuda(non_blocking=True)

            # compute loss
            features = torch.split(features, bsz, dim=0)
            features = torch.cat([f.unsqueeze(1) for f in features], dim=1)
            loss = self.criterion(features, labels)
        elif isinstance(self.criterion, TripletLoss):
            features = self.forward(batch)

            # compute loss
            anc_emb, pos_emb, neg_emb = torch.split(features, bsz, dim=0)
            loss = self.criterion(anc_emb, pos_emb, neg_emb)
        else:
            raise ValueError(f"Criterion {self.criterion} not supported")

        return loss
