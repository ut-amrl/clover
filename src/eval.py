from loguru import logger
import os
import time
from datetime import datetime
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.losses import SupConLoss, TripletLoss
from src.utils.misc import seed_everything, AverageMeter, load_model, save_model


def test(model, test_dataloader, cfg):
    pass


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
        method=cfg.model.criterion.name,
        transform=cfg.model.img_transform,
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
        logger.warning("No checkpoint path provided. Using random weights.")

    if torch.cuda.is_available():
        model.net = model.net.cuda()
        cudnn.benchmark = True
    else:
        raise ValueError("There is no GPU available.")

    # train
    test(model, test_dataloader, cfg)


if __name__ == "__main__":
    main()
