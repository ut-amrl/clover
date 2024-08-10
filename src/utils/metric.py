from typing import Literal

import torch
import torch.nn.functional as F


def get_top_indices(
    features,
    q_idx,
    r_indices,
    method: Literal["euclidean", "cosine"] = "cosine",
):
    """Get the top similarities between the query and reference features"""
    q_feautre = features[q_idx].unsqueeze(0)
    r_features = features[r_indices]

    if method == "cosine":
        similarity_scores = F.cosine_similarity(q_feautre, r_features, dim=1)
        sorted_indices = torch.argsort(similarity_scores, descending=True)
    elif method == "euclidean":
        similarity_scores = torch.norm(q_feautre - r_features, dim=1)
        sorted_indices = torch.argsort(similarity_scores, descending=False)

    return r_indices[sorted_indices]
