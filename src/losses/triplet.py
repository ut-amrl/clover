import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        """Euclidean distance of embeddings between anchor and positive"""
        dist_positive = F.pairwise_distance(anchor_emb, positive_emb, p=2)  # (B,)
        dist_negative = F.pairwise_distance(anchor_emb, negative_emb, p=2)  # (B,)
        losses = F.relu(dist_positive - dist_negative + self.margin)  # (B,)
        return losses.mean()
