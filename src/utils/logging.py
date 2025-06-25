import os
import pathlib

import rich.syntax
import rich.tree
import wandb
from omegaconf import DictConfig, OmegaConf


def print_config_tree(cfg: DictConfig, save_to_file: bool = False):
    tree = rich.tree.Tree("CONFIG")
    # add all the other fields to queue (not specified in `print_order`)
    queue = []
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=False)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)

    if save_to_file:
        file_path = pathlib.Path(cfg.paths.output_dir, "config_tree.log")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            rich.print(tree, file=file)

def log_test_metrics(metrics: dict):
    """Log test metrics to wandb
    Args:
        metrics (dict): {test_case: {metric_type: AverageMeter}}
    """
    columns = ["metric", *metrics.keys()]
    metrics_table = wandb.Table(columns=columns)

    metric_types = next(iter(metrics.values())).keys()
    for metric_type in metric_types:
        row = [metric_type]
        for test_case in metrics.values():
            row.append(f"{test_case[metric_type].avg:.4f}")
        metrics_table.add_data(*row)

    # Log the table to WandB
    wandb.log({"test/metrics": metrics_table})

def save_test_metrics(metrics: dict, file_path: str):
    """save metrics to a file (.csv)
    Args:
        metrics (dict): {test_case: {metric_type: AverageMeter}}
        file_path (str): file path to save the metrics
    """
    print(" - Saving metrics...")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        # Write the header (test cases)
        headers = ["metric", *metrics.keys()]
        f.write(",".join(headers) + "\n")

        # Write metrics for each metric_type
        metric_types = next(iter(metrics.values())).keys()
        for metric_type in metric_types:
            row = [metric_type]
            for test_case in metrics.values():
                row.append(f"{test_case[metric_type].avg:.4f}")
            f.write(",".join(row) + "\n")
