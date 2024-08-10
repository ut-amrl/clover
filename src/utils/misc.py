import os
import random

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
    print(f"==> Loading model from {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.net.load_state_dict(state_dict)


def save_model(model, epoch, cfg, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print(f"==> Saving model to {ckpt_path}\n")
    state = {
        "net": model.net.state_dict(),
        "optimizer": model.optimizer.state_dict(),
        "scheduler": model.scheduler.state_dict(),
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
