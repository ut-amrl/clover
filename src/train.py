import os
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig
from rich import print as rprint
from torch.backends import cudnn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging import print_config_tree
from utils.metric import get_top_indices
from utils.misc import AverageMeter, load_model, save_model, seed_everything


def train(model, train_dataloader, val_dataloader, cfg):
    best_acc = 0
    best_epoch = 0
    for epoch in range(cfg.train.max_epochs):
        # Train for one epoch
        model.net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for idx, batch in enumerate(train_dataloader):
            step = epoch * len(train_dataloader) + idx
            data_time.update(time.time() - end)

            # Forward pass
            loss = model.train_step(batch)
            losses.update(loss.item(), batch["label"].size(0))

            # Backward pass
            model.optimizer.zero_grad()
            loss.backward()
            if getattr(cfg.train, "clip_grad_norm", None):
                clip_grad_norm_(model.net.parameters(), cfg.train.clip_grad_norm)
            model.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if cfg.wandb:
                wandb.log({"train/loss": loss.item()}, step=step)

            rprint(
                f"Epoch: [{epoch}/{cfg.train.max_epochs}] "
                f"Step: [{idx}/{len(train_dataloader)}] "
                f"Loss: {losses.avg:.4f} "
                f"Batch Time [sec]: {batch_time.avg:.2f} "
                f"Data Time [sec]: {data_time.avg:.2f}",
                end="\r",
            )

        # Validation
        val_acc = validate(model, val_dataloader, cfg, epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            ckpt_path = os.path.join(cfg.paths.output_dir, "ckpt", f"epoch_{epoch}.pth")
            save_model(model, epoch, cfg, ckpt_path)

        # Save last model
        ckpt_path = os.path.join(cfg.paths.output_dir, "ckpt", "last.pth")
        save_model(model, epoch, cfg, ckpt_path)

        # update learning rate
        if model.scheduler:
            model.scheduler.step()

        # Early stopping
        if (epoch - best_epoch) >= cfg.train.early_stopping:
            rprint(f"Early stopping at epoch {epoch}")
            break


@torch.no_grad()
def validate(model, val_dataloader, cfg, epoch=0):
    print("\n\n * Validating the model...")
    model.net.eval()

    dataset_name = getattr(val_dataloader.dataset, "name", None)

    # Define metrics
    topk = [1, 5, 10]
    metrics = {
        "mAP": AverageMeter(),
        **{f"top@{k}": AverageMeter() for k in topk},
        "num_same_instance": AverageMeter(),
        "num_ref_instance": AverageMeter(),
    }

    # Compute features
    N = len(val_dataloader.dataset)
    features = torch.zeros(N, model.net.feat_dim, device="cpu")
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
    for batch in tqdm(val_dataloader, desc="computing features"):
        bsz = batch["label"].size(0)
        features[curr_idx : curr_idx + bsz] = model.forward(batch, mode="eval").cpu()
        labels[curr_idx : curr_idx + bsz] = batch["label"].cpu()
        if dataset_name == "CODa":
            info["classes"][curr_idx : curr_idx + bsz] = batch["class"].cpu()
            info["sequences"][curr_idx : curr_idx + bsz] = batch["sequence"].cpu()
            info["weathers"][curr_idx : curr_idx + bsz] = batch["weather"].cpu()
            info["viewpoints"][curr_idx : curr_idx + bsz] = batch["viewpoint"].cpu()
        curr_idx += bsz

    # Compute metrics
    for q_idx, q_label in tqdm(enumerate(labels), total=N, desc="computing metrics"):
        # Filter reference indices
        same_label_mask = labels == q_label
        same_label_mask[q_idx] = 0
        diff_label_mask = labels != q_label
        if dataset_name == "CODa":
            # same label indices (exclude images from the same sequence)
            diff_sequence_mask = info["sequences"] != info["sequences"][q_idx]
            same_label_indices = torch.where(same_label_mask & diff_sequence_mask)[0]
            # reference indices (same class but different instance)
            same_class_mask = info["classes"] == info["classes"][q_idx]
            diff_label_indices = torch.where(diff_label_mask & same_class_mask)[0]
            pass
        elif dataset_name == "ScanNet":
            same_label_indices = torch.where(same_label_mask)[0]
            diff_label_indices = torch.where(diff_label_mask)[0]
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        # Skip if there is no same instance in the reference set
        if len(same_label_indices) == 0:
            rprint(f"WARNING: Query index {q_idx} has no same instance in the reference set.")
            continue

        # Get top indices
        r_indices = torch.cat((same_label_indices, diff_label_indices), dim=0)
        top_indices = get_top_indices(features, q_idx, r_indices, method=model.similarity_type)

        # Update metrics
        hits = torch.eq(labels[top_indices], q_label)
        precisions = torch.cumsum(hits, dim=0) / torch.arange(1, len(hits) + 1)

        metrics["mAP"].update(precisions[hits].mean().item())
        for k in topk:
            metrics[f"top@{k}"].update(float(hits[:k].any().item()))
        metrics["num_same_instance"].update(len(same_label_indices))
        metrics["num_ref_instance"].update(len(r_indices))

    # Log metrics
    for key, val in metrics.items():
        rprint(f" - [bold]{key}[/bold]: {val.avg:.4f}")

    # Log to wandb
    if cfg.wandb:
        log_dict = {f"val/{key}": meter.avg for key, meter in metrics.items()}
        wandb.log(log_dict, step=epoch)
        metrics_table = wandb.Table(columns=["name", *metrics.keys()])
        metrics_table.add_data(cfg.name, *[meter.avg for meter in metrics.values()])
        wandb.log({"val/metrics": metrics_table}, step=epoch)

    return metrics["mAP"].avg


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
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
            tags=["train", *cfg.tags],
            config={**cfg},
        )

    # Load dataset
    train_dataset = hydra.utils.instantiate(
        cfg.dataset,
        mode="train",
        transform=cfg.model.img_transform,
        method=cfg.model.criterion.name if cfg.model.get("criterion") else None,
    )
    val_dataset = hydra.utils.instantiate(
        cfg.dataset,
        mode="val",
        transform=cfg.model.img_transform,
        method=cfg.model.criterion.name if cfg.model.get("criterion") else None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Load model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path:
        load_model(model, cfg.ckpt_path)

    if torch.cuda.is_available():
        model.net = model.net.cuda()
        model.criterion = model.criterion.cuda()
        cudnn.benchmark = True
    else:
        raise ValueError("There is no GPU available.")

    # train
    train(model, train_dataloader, val_dataloader, cfg)


if __name__ == "__main__":
    main()
