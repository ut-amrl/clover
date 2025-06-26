import torch
import torch.nn as nn


class WhereDidISeeIt(nn.Module):
    def __init__(self, net, criterion, optimizer, scheduler=None, **kwargs):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer(params=net.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer) if scheduler else None
        self.similarity_type = "euclidean"

    def forward(self, batch, **kwargs):
        fg = [f.cuda(non_blocking=True) for f in [img[0] for img in batch["images"]]]
        bg = [b.cuda(non_blocking=True) for b in [img[1] for img in batch["images"]]]
        fg = torch.cat(fg, dim=0)
        bg = torch.cat(bg, dim=0)
        return self.net(fg, bg)

    def train_step(self, batch):
        self.net.train()

        bsz = batch["label"].size(0)
        features = self.forward(batch)

        criterion_name = type(self.criterion).__name__
        if criterion_name == "SupConLoss":
            features = torch.split(features, bsz, dim=0)
            features = torch.cat([f.unsqueeze(1) for f in features], dim=1)
            labels = batch["label"].cuda(non_blocking=True)
            loss = self.criterion(features, labels)
        elif criterion_name == "TripletLoss":
            anc_emb, pos_emb, neg_emb = torch.split(features, bsz, dim=0)
            loss = self.criterion(anc_emb, pos_emb, neg_emb)
        else:
            raise ValueError(f"Criterion {self.criterion} not supported")

        return loss
