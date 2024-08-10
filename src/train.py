from loguru import logger
import os
import time
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.utils.misc import seed_everything, AverageMeter, load_model, save_model
from src.utils.metric import get_top_indices


def train(model, train_dataloader, val_dataloader, cfg):
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, cfg.train.max_epochs + 1):
        # Train for one epoch
        model.net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for idx, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)

            loss = model.train_step(batch)
            losses.update(loss.item(), batch["label"].size(0))

            model.optimizer.zero_grad()
            loss.backward()
            if getattr(cfg.train, "clip_grad_norm", None):
                clip_grad_norm_(model.net.parameters(), cfg.train.clip_grad_norm)

            model.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % cfg.train.log_interval == 0:
                rprint(
                    f"Epoch: [{epoch}/{cfg.train.max_epochs}] "
                    f"Step: [{idx}/{len(train_dataloader)}] "
                    f"Loss: {losses.avg:.4f} "
                    f"Batch time: {batch_time.avg:.2f}s "
                    f"Data time: {data_time.avg:.2f}s"
                )

        # Validation
        val_acc = validate(model, val_dataloader)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            ckpt_path = os.path.join(cfg.paths.output_dir, "ckpt", f"epoch_{epoch}.pth")
            save_model(model, epoch, cfg, ckpt_path)

        # Save last model
        ckpt_path = os.path.join(cfg.paths.output_dir, "ckpt", f"last.pth")
        save_model(model, epoch, cfg, ckpt_path)

        # update learning rate
        if model.scheduler:
            model.scheduler.step()

        # Early stopping
        if (epoch - best_epoch) >= cfg.train.early_stopping:
            rprint(f"Early stopping at epoch {epoch}")
            break


def validate(model, val_dataloader):
    print("\n * Validating the model...")
    model.net.eval()

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
    features = torch.zeros(N, model.net.feat_dim)
    labels = torch.zeros(N, dtype=torch.long)
    with torch.no_grad():
        curr_idx = 0
        for batch in tqdm(val_dataloader, desc="computing features..."):
            bsz = batch["label"].size(0)
            features[curr_idx : curr_idx + bsz] = model.forward(batch, mode="eval")
            labels[curr_idx : curr_idx + bsz] = batch["label"]
            curr_idx += bsz

    # Compute metrics
    for q_idx, q_label in tqdm(enumerate(labels), total=N, desc="computing metrics..."):
        # Filter reference indices
        same_label_mask = labels == q_label
        same_label_mask[q_idx] = 0

        if same_label_mask.sum() == 0:
            logger.warning(f"No same instance in reference set for query {q_idx}")
            continue

        r_indices, num_ref = None, None
        if val_dataloader.dataset.name == "CODa":
            pass
        elif val_dataloader.dataset.name == "ScanNet":
            r_indices = torch.cat(
                (torch.arange(q_idx), torch.arange(q_idx + 1, N)), dim=0
            )
            num_ref = N - 1
        else:
            raise ValueError(f"Dataset {val_dataloader.dataset.name} not supported")

        # Get top indices
        top_indices = get_top_indices(
            features, q_idx, r_indices, method=model.similarity_type
        )

        # Update metrics
        hits = torch.eq(labels[top_indices], q_label)
        precisions = (
            torch.cumsum(hits, dim=0).float() / (torch.arange(len(hits)) + 1).float()
        )

        metrics["mAP"].update(precisions[hits].mean().item())
        for k in topk:
            metrics[f"top@{k}"].update(float(hits[:k].any().item()))
        metrics["num_same_instance"].update(same_label_mask.sum().item())
        metrics["num_ref_instance"].update(num_ref)

    # Log metrics
    for key, val in metrics.items():
        rprint(f" - {key}: {val.avg:.4f}")

    return metrics["mAP"].avg


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    rprint(config_str)

    if cfg.get("seed"):
        seed_everything(cfg.seed)

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
