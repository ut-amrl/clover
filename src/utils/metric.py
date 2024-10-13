import os
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
    q_feature = features[q_idx].unsqueeze(0)
    r_features = features[r_indices]

    if method == "cosine":
        similarity_scores = F.cosine_similarity(q_feature, r_features, dim=1)
        sorted_indices = torch.argsort(similarity_scores, descending=True)
    elif method == "euclidean":
        similarity_scores = torch.norm(q_feature - r_features, dim=1)
        sorted_indices = torch.argsort(similarity_scores, descending=False)

    # Get top indices (excluding the query index)
    top_indices = r_indices[sorted_indices]
    top_indices = top_indices[top_indices != q_idx]

    return top_indices


def save_metrics(metrics: dict, file_path: str):
    """save metrics to a file (.csv)
    Args:
        metrics (dict): {test_case: {metric_type: AverageMeter}}
        file_path (str): file path to save the metrics
    """
    print(" - Saving metrics...")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        # Write the header (test cases)
        headers = ["metric"] + list(metrics.keys())
        f.write(",".join(headers) + "\n")

        # Write metrics for each metric_type
        metric_types = next(iter(metrics.values())).keys()
        for metric_type in metric_types:
            row = [metric_type]
            for test_case in metrics.values():
                row.append(f"{test_case[metric_type].avg:.4f}")
            f.write(",".join(row) + "\n")
