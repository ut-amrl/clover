"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Data:   Oct 12, 2024
Description: This file contains functions to compute the weather and viewpoint variances between query and reference.
"""
import torch


def compute_weather_difference(weathers, q_idx, r_indices=None):
    """Compute the weather differences between query and reference.
    Args:
        weathers (list(torch.Tensor)): list of weather tensors
        q_idx (int): query index
        r_indices (list): list of reference indices

    Returns:
        weather_differences (torch.Tensor): tensor of weather variances (0: same, 1: different)
    """
    if r_indices is None:
        r_indices = torch.arange(len(weathers))

    weather_differences = (weathers[q_idx] != weathers[r_indices]).long()

    return weather_differences


def compute_viewpoint_difficulty(
    viewpoints, q_idx, r_indices=None, thresholds=[(10, 15), (30, 90)]
):
    """Compute the view difficulties between query and reference.
    Args:
        viewpoints (list(torch.Tensor)): list of viewpoint tensors
        q_idx (int): query index
        r_indices (list): list of reference indices
        thresholds (list(tuple)): list of distance and angle thresholds for easy and medium

    Returns:
        viewpoint_difficulties (torch.Tensor): tensor of view difficulties
                                               (0: easy, 1: medium, 2: hard)
        distances (torch.Tensor): tensor of norm differences
        angles (torch.Tensor): tensor of angles
    """
    if r_indices is None:
        r_indices = torch.arange(len(viewpoints))

    q_viewpoints = viewpoints[q_idx].unsqueeze(0)
    r_viewpoints = viewpoints[r_indices]

    q_norm = q_viewpoints.norm(dim=1, keepdim=True)
    r_norm = r_viewpoints.norm(dim=1, keepdim=True)
    q_normalized_viewpoints = q_viewpoints / q_norm
    r_normalized_viewpoints = r_viewpoints / r_norm

    distances = torch.abs(q_norm - r_norm).squeeze(1)
    dot_product = torch.matmul(q_normalized_viewpoints, r_normalized_viewpoints.T)
    dot_product.clamp_(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(dot_product)).squeeze(0)

    is_easy = (distances < thresholds[0][0]) & (angles < thresholds[0][1])
    is_medium = ~is_easy & (distances < thresholds[1][0]) & (angles < thresholds[1][1])
    viewpoint_difficulties = torch.zeros_like(distances, dtype=torch.long)  # easy
    viewpoint_difficulties[is_medium] = 1  # medium
    viewpoint_difficulties[~is_easy & ~is_medium] = 2  # hard

    return viewpoint_difficulties, distances, angles
