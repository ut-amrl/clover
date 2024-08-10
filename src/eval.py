from loguru import logger
import os
import time
from datetime import datetime
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.utils.misc import seed_everything, AverageMeter, load_model
from src.utils.metric import get_top_indices


def test(model, test_dataloader, cfg):
    print("\n * Testing the model...")
    model.net.eval()

    # Define test cases
    test_cases = {
        "all_all": {"env": torch.tensor([0, 1]), "view": torch.tensor([0, 1, 2])}
    }
    if test_dataloader.dataset.name == "CODa":
        test_cases.update(
            {
                "sim_all": {"env": torch.tensor([0]), "view": torch.tensor([0, 1, 2])},
                "diff_all": {"env": torch.tensor([1]), "view": torch.tensor([0, 1, 2])},
                "all_easy": {"env": torch.tensor([0, 1]), "view": torch.tensor([0])},
                "all_med": {"env": torch.tensor([0, 1]), "view": torch.tensor([1])},
                "all_hard": {"env": torch.tensor([0, 1]), "view": torch.tensor([2])},
            }
        )

    # Define metrics
    topk = [1, 5, 10]
    metrics = {
        test_case: {
            "mAP": AverageMeter(),
            **{f"top@{k}": AverageMeter() for k in topk},
            "num_same_instance": AverageMeter(),
            "num_ref_instance": AverageMeter(),
        }
        for test_case in test_cases.keys()
    }

    # Compute features
    N = len(test_dataloader.dataset)
    features = torch.zeros(N, model.net.feat_dim)
    labels = torch.zeros(N, dtype=torch.long)
    with torch.no_grad():
        curr_idx = 0
        for batch in tqdm(test_dataloader, desc="computing features"):
            bsz = batch["label"].size(0)
            features[curr_idx : curr_idx + bsz] = model.forward(batch, mode="eval")
            labels[curr_idx : curr_idx + bsz] = batch["label"]
            curr_idx += bsz

    # Compute metrics
    for q_idx, q_label in tqdm(enumerate(labels), total=N, desc="computing metrics"):
        # Filter reference indices
        same_label_mask = labels == q_label
        same_label_mask[q_idx] = 0
        diff_label_mask = labels != q_label

        if same_label_mask.sum() == 0:
            logger.warning(f"No same instance in referece set for query {q_idx}")
            continue

        # get top indices for all
        r_indices = torch.cat((torch.arange(q_idx), torch.arange(q_idx + 1, N)), dim=0)

        top_indices = get_top_indices(
            features, q_idx, r_indices, method=model.similarity_type
        )

        if test_dataloader.dataset.name == "CODa":
            pass
        elif test_dataloader.dataset.name == "ScanNet":
            same_label_indices = torch.where(same_label_mask)[0]
            ref_label_indices = torch.where(diff_label_mask)[0]
            env_conditions = torch.zeros(len(same_label_indices))
            view_conditions = torch.zeros(len(same_label_indices))

        # update metrics per test case
        for test_case, conditions in test_cases.items():
            env_mask = torch.isin(env_conditions, conditions["env"])
            view_mask = torch.isin(view_conditions, conditions["view"])
            test_same_label_indices = same_label_indices[env_mask & view_mask]

            # skip if no same instance in the reference set
            if test_same_label_indices.numel() == 0:
                continue

            # compute metrics for the test case
            test_indices = torch.cat((test_same_label_indices, ref_label_indices))
            filtered_top_indices = top_indices[torch.isin(top_indices, test_indices)]

            # update metrics
            hits = torch.eq(labels[filtered_top_indices], q_label)
            precisions = (
                torch.cumsum(hits, dim=0).float()
                / (torch.arange(len(hits)) + 1).float()
            )

            metrics[test_case]["mAP"].update(precisions[hits].mean().item())
            for k in topk:
                metrics[test_case][f"top@{k}"].update(float(hits[:k].any().item()))
            metrics[test_case]["num_same_instance"].update(len(test_same_label_indices))
            metrics[test_case]["num_ref_instance"].update(len(test_indices))

    # Log metrics
    print("\n * Test results")
    for test_case, test_metrics in metrics.items():
        print(f" - Test case: {test_case}")
        for key, val in test_metrics.items():
            print(f"   - {key}: {val.avg:.4f}")

    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    rprint(config_str)

    if cfg.get("seed"):
        seed_everything(cfg.seed)

    # Load dataset
    test_dataset = hydra.utils.instantiate(
        cfg.dataset,
        mode="test",
        transform=cfg.model.img_transform,
        method=cfg.model.criterion.name if cfg.model.get("criterion") else None,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Load model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path:
        load_model(model, cfg.ckpt_path)
    else:
        logger.warning("No checkpoint path provided.")

    if torch.cuda.is_available():
        model.net = model.net.cuda()
        cudnn.benchmark = True
    else:
        raise ValueError("There is no GPU available.")

    # train
    test(model, test_dataloader, cfg)


if __name__ == "__main__":
    main()
