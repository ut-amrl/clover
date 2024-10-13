import os
import random
from rich import print as rprint

import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(model, ckpt_path):
    rprint(f"==> Loading model from {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.net.load_state_dict(ckpt["net"])
        if hasattr(model, "optimizer"):
            model.optimizer.load_state_dict(ckpt["optimizer"])
        if hasattr(model, "scheduler"):
            model.scheduler.load_state_dict(ckpt["scheduler"])
        if "cfg" in ckpt:
            model.cfg = ckpt["cfg"]
    except Exception as e:
        rprint(f"Error loading model: {e}")
        raise e


def save_model(model, epoch, cfg, ckpt_path):
    rprint(f"==> Saving model to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    state = {
        "net": model.net.state_dict(),
        "optimizer": (
            model.optimizer.state_dict() if hasattr(model, "optimizer") else None
        ),
        "scheduler": (
            model.scheduler.state_dict() if hasattr(model, "scheduler") else None
        ),
        "epoch": epoch,
        "cfg": cfg,
    }
    torch.save(state, ckpt_path)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
