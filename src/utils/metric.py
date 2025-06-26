import os

import torch
import torch.nn.functional as F


def get_similarity_matrix(features, method="cosine"):
    if method == "cosine":
        features = F.normalize(features, dim=1)
        return features @ features.T
    elif method == "euclidean":
        return -torch.cdist(features, features)
    else:
        raise ValueError(method)


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
