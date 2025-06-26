import os

import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig
from rich import print as rprint
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.env_variances import compute_viewpoint_difficulty, compute_weather_difference
from utils.logging import log_test_metrics, print_config_tree, save_test_metrics
from utils.metric import get_similarity_matrix
from utils.misc import AverageMeter, load_model, seed_everything


@torch.no_grad()
def test(model, test_loader, cfg):
    print("\n * Testing the model...")
    model.net.eval()

    dataset_name = getattr(test_loader.dataset, "name", None)

    # Define test cases
    if dataset_name == "CODa":
        test_cases = {
            "same_easy": {"env": torch.tensor([0]), "view": torch.tensor([0])},
            "same_med": {"env": torch.tensor([0]), "view": torch.tensor([1])},
            "same_hard": {"env": torch.tensor([0]), "view": torch.tensor([2])},
            "same_all": {"env": torch.tensor([0]), "view": torch.tensor([0, 1, 2])},
            "diff_easy": {"env": torch.tensor([1]), "view": torch.tensor([0])},
            "diff_med": {"env": torch.tensor([1]), "view": torch.tensor([1])},
            "diff_hard": {"env": torch.tensor([1]), "view": torch.tensor([2])},
            "diff_all": {"env": torch.tensor([1]), "view": torch.tensor([0, 1, 2])},
            "all_easy": {"env": torch.tensor([0, 1]), "view": torch.tensor([0])},
            "all_med": {"env": torch.tensor([0, 1]), "view": torch.tensor([1])},
            "all_hard": {"env": torch.tensor([0, 1]), "view": torch.tensor([2])},
            "all_all": {"env": torch.tensor([0, 1]), "view": torch.tensor([0, 1, 2])},
        }
    elif dataset_name == "ScanNet":
        test_cases = {
            "all_all": {"env": torch.tensor([0, 1]), "view": torch.tensor([0, 1, 2])},
        }
    else:
        raise NotImplementedError(f"{dataset_name} not supported")

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

    # Compute features and aggregate labels, additional info
    N = len(test_loader.dataset)
    features = torch.zeros((N, model.net.feat_dim), dtype=torch.float, device="cpu")
    labels = torch.zeros(N, dtype=torch.long, device="cpu")

    # additional info (only for CODa)
    if dataset_name == "CODa":
        info = {
            "classes": torch.zeros(N, dtype=torch.long, device="cpu"),
            "sequences": torch.zeros(N, dtype=torch.long, device="cpu"),
            "weathers": torch.zeros(N, dtype=torch.long, device="cpu"),
            "viewpoints": torch.zeros((N, 3), dtype=torch.float, device="cpu"),
        }

    curr_idx = 0
    for batch in tqdm(test_loader, desc="computing features"):
        bsz = batch["label"].size(0)
        features[curr_idx : curr_idx + bsz] = model.forward(batch, mode="eval").cpu()
        labels[curr_idx : curr_idx + bsz] = batch["label"].cpu()
        if dataset_name == "CODa":
            info["classes"][curr_idx : curr_idx + bsz] = batch["class"].cpu()
            info["sequences"][curr_idx : curr_idx + bsz] = batch["sequence"].cpu()
            info["weathers"][curr_idx : curr_idx + bsz] = batch["weather"].cpu()
            info["viewpoints"][curr_idx : curr_idx + bsz] = batch["viewpoint"].cpu()
        curr_idx += bsz

    # Precompute similarity and sorted indices
    sim_matrix = get_similarity_matrix(features, method=model.similarity_type)
    sim_matrix.fill_diagonal_(-float("inf"))
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)  # (N, N)

    # Per-query metric computation loop
    for q_idx, q_label in tqdm(enumerate(labels), total=N, desc="computing metrics"):
        # Get top indices for the query
        top_indices = sorted_indices[q_idx]

        same_label_mask = labels == q_label
        same_label_mask[q_idx] = False  # exclude the query itself
        diff_label_mask = labels != q_label

        if same_label_mask.sum() == 0:
            logger.warning(f"No same instance in referece set for query {q_idx}")
            continue

        ## Get Reference indices
        if dataset_name == "CODa":
            # same label indices (exclude images from the same sequence)
            diff_sequence_mask = info["sequences"] != info["sequences"][q_idx]
            same_label_indices = torch.where(same_label_mask & diff_sequence_mask)[0]
            # reference indices (same class but different instance)
            same_class_mask = info["classes"] == info["classes"][q_idx]
            ref_label_indices = torch.where(diff_label_mask & same_class_mask)[0]
            # compute environmental conditions and viewpoint difficulties
            environment_differences = compute_weather_difference(
                info["weathers"], q_idx, same_label_indices
            )
            viewpoint_difficulties, _, _ = compute_viewpoint_difficulty(
                info["viewpoints"], q_idx, same_label_indices
            )
        elif dataset_name == "ScanNet":
            same_label_indices = torch.where(same_label_mask)[0]
            ref_label_indices = torch.where(diff_label_mask)[0]
            environment_differences = torch.zeros(len(same_label_indices))
            viewpoint_difficulties = torch.zeros(len(same_label_indices))
        else:
            raise NotImplementedError(f"{dataset_name} not supported")

        ## update metrics per test case
        for test_case, conditions in test_cases.items():
            env_mask = torch.isin(environment_differences, conditions["env"])
            viewpoint_mask = torch.isin(viewpoint_difficulties, conditions["view"])
            test_same_label_indices = same_label_indices[env_mask & viewpoint_mask]

            # skip if no same instance in the reference set
            if test_same_label_indices.numel() == 0:
                continue

            # compute metrics for the test case
            test_indices = torch.cat((test_same_label_indices, ref_label_indices))
            filtered_top_indices = top_indices[torch.isin(top_indices, test_indices)]

            # update metrics
            hits = labels[filtered_top_indices] == q_label
            precisions = torch.cumsum(hits, dim=0) / (torch.arange(len(hits)) + 1)

            metrics[test_case]["mAP"].update(precisions[hits].mean().item())
            for k in topk:
                metrics[test_case][f"top@{k}"].update(float(hits[:k].any().item()))
            metrics[test_case]["num_same_instance"].update(len(test_same_label_indices))
            metrics[test_case]["num_ref_instance"].update(len(test_indices))

    # Log metrics
    rprint("[bold]Test results[/bold]")
    for test_case, test_metrics in metrics.items():
        rprint(f"* Test case: [cyan]{test_case}[/cyan]")
        for key, val in test_metrics.items():
            rprint(f"   - {key}: {val.avg:.4f}")

    # Log to wandb
    if cfg.get("wandb"):
        log_test_metrics(metrics)

    # Save metrics
    test_metrics_file = os.path.join(cfg.paths.output_dir, "results", "metrics.csv")
    save_test_metrics(metrics, test_metrics_file)

    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    # resolve paths
    for key, path in cfg.paths.items():
        cfg.paths[key] = hydra.utils.to_absolute_path(path)

    print_config_tree(cfg, save_to_file=True)

    if cfg.seed:
        seed_everything(cfg.seed)

    if cfg.wandb:
        wandb.init(
            entity="ut-amrl-amanda",
            dir=cfg.paths.output_dir,
            project=cfg.project,
            name=cfg.name,
            notes=cfg.paths.output_dir,
            tags=["test", *cfg.tags],
            config={**cfg},
        )

    # Load dataset
    test_dataset = hydra.utils.instantiate(
        cfg.dataset,
        mode="test",
        transform=cfg.model.img_transform,
        method=None,  # No method for evaluation
    )

    test_loader = DataLoader(
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
    elif cfg.model.name in ["CLOVER", "ReOBJ", "WDISI"]:
        logger.warning("No checkpoint path provided.")

    if torch.cuda.is_available():
        model.net = model.net.cuda()
        cudnn.benchmark = True
    else:
        raise ValueError("There is no GPU available.")

    # train
    test(model, test_loader, cfg)


if __name__ == "__main__":
    main()
