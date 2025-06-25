"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Oct 7, 2024
Description: UT Campus Object Dataset (CODA)

Paper: https://arxiv.org/abs/2309.13549
"""

import json
import pathlib
import random
from collections import defaultdict
from operator import itemgetter
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from loguru import logger
from natsort import natsorted
from rich import box
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset
from tqdm import tqdm


class CODataset(Dataset):

    classes_id = {
        "Tree": 0,
        "Pole": 1,
        "Bollard": 2,
        # "Informational_Sign": 3,
        # "Traffic_Sign": 4,
        # "Trash_Can": 5,
        # "Fire_Hydrant": 6,
        # "Emergency_Phone": 7,
    }
    weathers_id = {"sunny": 0, "cloudy": 1, "dark": 2, "rainy": 3}

    target_sequences = set(range(23)) - {8, 14, 15}
    target_region = {"x": [-np.inf, np.inf], "y": [-np.inf, np.inf]}

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "val", "test"],
        split: Literal["region", "sequence", "random", "class"],
        transform: Callable,
        method: Literal["SupCon", "Triplet"] = None,
        **kwargs,
    ):
        self.name = "CODa"
        self.root_dir = pathlib.Path(root_dir)
        self.mode = mode
        self.split = split
        self.method = method if mode == "train" else None
        self.transform = transform
        self.n_views = kwargs.get("n_views", 2)
        self.skip = kwargs.get(f"skip_{mode}", 1)

        # Split-specific parameters
        if split == "region":
            self.target_region = kwargs.get(f"{mode}_region")
        elif split == "sequence":
            self.target_sequences = kwargs.get(f"{mode}_sequences", [])
        elif split == "random":
            self.train_ratio = kwargs.get("train_ratio", 0.7)
        elif split == "class":
            self.classes_id = {
                class_name: class_id
                for class_id, class_name in enumerate(kwargs.get(f"{mode}_classes"))
            }

        # Load dataset
        self._load_global_instances()
        self._load_dataset()

        # Print dataset information
        self._print_dataset_info()

    def _load_global_instances(self):
        classes_json = {}
        for class_name in self.classes_id:
            class_file = self.root_dir / "3d_bbox/global" / f"{class_name}.json"
            if class_file.exists():
                classes_json[class_name] = json.load(open(class_file, "r"))
            else:
                logger.warning(f"Global instances not found: {class_file}")

        # {class_id: {instance_id: [x, y, z, h, l, w, r, p, y]}}
        self.global_instances = defaultdict(lambda: defaultdict(list))
        count = 0
        for class_name, instance_json in classes_json.items():
            for instance in instance_json["instances"]:
                class_id = self.classes_id[class_name]
                instance_id = int(instance["id"])
                self.global_instances[class_id][instance_id] = [
                    float(instance["cX"]),
                    float(instance["cY"]),
                    float(instance["cZ"]),
                    float(instance["h"]),
                    float(instance["l"]),
                    float(instance["w"]),
                    float(instance["r"]),
                    float(instance["p"]),
                    float(instance["y"]),
                ]
                count += 1

    def _load_dataset(self):
        unique_label_map = {}
        unique_label = 0

        # {class: {label: [idx1, idx2, ...]}}
        self.instance_indices = defaultdict(lambda: defaultdict(list))
        self.image_files = []
        self.classes = []
        self.labels = []
        self.bboxes = []
        self.segmentations = []
        self.sequences = []
        self.weathers = []
        self.viewpoints = []

        # Get Annotation Files to load
        annotation_files = []
        for cam in ["cam0", "cam1"]:
            for seq_dir in natsorted(
                [d for d in self.root_dir.glob(f"annotations/{cam}/*") if d.is_dir()]
            ):
                # NOTE: Sequence-based split
                if (
                    self.split == "sequence"
                    and int(seq_dir.name) not in self.target_sequences
                ):
                    continue

                # Anno files is per image -- need to associate image id to each instance
                for annotation_file in natsorted([f for f in seq_dir.glob("*.json")]):
                    annotation_files.append(annotation_file)

        # Load instance-level annotations
        for annotation_file in tqdm(
            annotation_files[:: self.skip], desc="Loading annotations", leave=False
        ):
            try:
                annotation = json.load(open(annotation_file, "r"))
                info = annotation["info"]
                image_file = (
                    self.root_dir
                    / "2d_raw"
                    / info["camera"]
                    / str(info["sequence"])
                    / (info["image_file"] + ".jpg")
                )

                for instance in annotation["instances"]:
                    class_name = instance["class"]
                    # NOTE: Class-based split
                    if class_name not in self.classes_id:
                        continue

                    class_id = self.classes_id[class_name]
                    instance_id = int(instance["id"])
                    cX = self.global_instances[class_id][instance_id][0]
                    cY = self.global_instances[class_id][instance_id][1]
                    cZ = self.global_instances[class_id][instance_id][2]

                    # NOTE: Region-based split
                    if (
                        cX < self.target_region["x"][0]
                        or cX > self.target_region["x"][1]
                        or cY < self.target_region["y"][0]
                        or cY > self.target_region["y"][1]
                    ):
                        continue

                    # Assign unique label
                    if (class_name, instance_id) not in unique_label_map:
                        unique_label_map[(class_name, instance_id)] = unique_label
                        unique_label += 1
                    label = unique_label_map[(class_name, instance_id)]

                    # Append to dataset
                    idx = len(self.image_files)
                    self.instance_indices[class_id][label].append(idx)
                    self.image_files.append(image_file)
                    self.classes.append(class_id)
                    self.labels.append(label)
                    self.bboxes.append(instance["bbox"])
                    self.segmentations.append(instance["segmentation"])
                    self.weathers.append(self.weathers_id[info["weather_condition"]])
                    self.sequences.append(int(info["sequence"]))
                    self.viewpoints.append(
                        (np.array([cX, cY, cZ]) - np.array(info["pose"][:3])).tolist()
                    )
            except Exception as e:
                logger.error(f"Failed to load {annotation_file}: {e}")
                continue

        # Remove instances with a single view
        self._remove_single_view_instances()

        # NOTE: Random-based split
        if self.split == "random":
            self._random_split()

    def _remove_single_view_instances(self):
        """Removes instances that only have a single view."""
        new_instance_indices = defaultdict(lambda: defaultdict(list))
        selected_indices = []
        last_index = 0
        for class_id, instances in self.instance_indices.items():
            for label, indices in instances.items():
                # Skip instances with a single view
                if len(indices) < 2:
                    continue
                new_instance_indices[class_id][label] = range(
                    last_index, last_index + len(indices)
                )
                selected_indices.extend(indices)
                last_index += len(indices)

        # Update instance indices and filter the dataset
        self.instance_indices = new_instance_indices
        self._filter_dataset(selected_indices)

    def _random_split(self):
        new_instance_indices = defaultdict(lambda: defaultdict(list))
        selected_indices = []
        last_index = 0
        for class_id, instances in self.instance_indices.items():
            # Shuffle instances and split based on the train ratio
            all_instances = random.sample(list(instances.keys()), len(instances))
            selected_instances = (
                all_instances[: int(len(instances) * self.train_ratio)]
                if self.mode == "train"
                else all_instances[int(len(instances) * self.train_ratio) :]
            )

            # Update instance indices
            for instance_id in selected_instances:
                new_instance_indices[class_id][instance_id] = range(
                    last_index, last_index + len(instances[instance_id])
                )
                selected_indices.extend(instances[instance_id])
                last_index += len(instances[instance_id])

        # Update instance indices and filter the dataset
        self.instance_indices = new_instance_indices
        self._filter_dataset(selected_indices)

    def _filter_dataset(self, selected_indices):
        """Utility method to filter the dataset based on selected indices."""
        selector = itemgetter(*selected_indices)
        self.image_files = selector(self.image_files)
        self.classes = selector(self.classes)
        self.labels = selector(self.labels)
        self.bboxes = selector(self.bboxes)
        self.segmentations = selector(self.segmentations)
        self.sequences = selector(self.sequences)
        self.weathers = selector(self.weathers)
        self.viewpoints = selector(self.viewpoints)

    def _print_dataset_info(self):
        console = Console()
        console.rule(
            f"[bold]CODa ({self.mode}) dataset path: '{self.root_dir}'[/bold]",
            style="dim",
        )
        console.print(
            f"Split: [bold cyan]{self.split}[/bold cyan], "
            f"Method: [bold cyan]{self.method}[/bold cyan], "
            f"Skip: [cyan]{self.skip}[/cyan]",
            end=", ",
        )
        if self.split == "region":
            console.print(f"Region: [yellow]{self.target_region}[/yellow]")
        elif self.split == "sequence":
            console.print(f"Sequences: [yellow]{self.target_sequences}[/yellow]")
        elif self.split == "random":
            console.print(f"Train Ratio: [yellow]{self.train_ratio}[/yellow]")
        elif self.split == "class":
            console.print(f"Classes: [yellow]{list(self.classes_id.keys())}[/yellow]")

        table = Table(title="Dataset Statistics", box=box.ROUNDED)
        table.add_column("Class", justify="left", style="cyan", no_wrap=True)
        table.add_column(
            "Instances (selected / total)", justify="center", style="magenta"
        )
        table.add_column("Views", justify="right", style="green")
        sum_total, sum_selected, sum_views = 0, 0, 0
        for class_name, class_id in self.classes_id.items():
            instances = self.instance_indices[class_id]
            total = len(self.global_instances[class_id])
            selected = len(instances)
            views = sum(len(indices) for indices in instances.values())
            sum_total += total
            sum_selected += selected
            sum_views += views
            table.add_row(class_name, f"{selected:,} / {total:,}", f"{views:,}")
        # Add a separator row for visual distinction
        table.add_row("─" * 25, "─" * 25, "─" * 15)

        # Add total row after the separator
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{sum_selected:,} / {sum_total:,}[/bold]",
            f"[bold]{sum_views:,}[/bold]",
        )
        console.print(table)
        console.rule(style="dim", align="left")

    def get_positive_idx(self, idx):
        pos_cls = self.classes[idx]
        pos_label = self.labels[idx]
        return np.random.choice(self.instance_indices[pos_cls][pos_label], 1)[0]

    def get_negative_idx(self, idx):
        label = self.labels[idx]

        # Same class vs. different class for negative samples
        neg_cls = self.classes[idx]
        if len(self.instance_indices[neg_cls]) < 2 or np.random.rand() < 0.5:
            potential_classes = [
                cls
                for cls in self.instance_indices
                if len(self.instance_indices[cls]) > 1
            ]
            neg_cls = np.random.choice(potential_classes)

        # remove the positive label
        neg_labels = list(self.instance_indices[neg_cls].keys())
        if label in neg_labels:
            neg_labels.remove(label)

        neg_label = np.random.choice(neg_labels, 1)[0]
        return np.random.choice(self.instance_indices[neg_cls][neg_label], 1)[0]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get indices for the views
        # SupCon: [n_view1, n_view2, ...]
        # Triplet: [anchor, positive, negative]
        indices = [idx]
        if self.mode == "train":
            if self.method == "SupCon":
                indices.extend(
                    [self.get_positive_idx(idx) for _ in range(self.n_views - 1)]
                )
            elif self.method == "Triplet":
                pos_idx = self.get_positive_idx(idx)
                neg_idx = self.get_negative_idx(idx)
                indices.extend([pos_idx, neg_idx])
            else:
                raise ValueError(f"Invalid method: {self.method}")

        def load_img(img_file):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        data = {
            "images": [
                self.transform(
                    img=load_img(self.image_files[i]),
                    bbox=self.bboxes[i],
                    segmentation=self.segmentations[i],
                    mode="train" if self.mode == "train" else "eval",
                )
                for i in indices
            ],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "class": torch.tensor(self.classes[idx], dtype=torch.long),
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.long),
            "weather": torch.tensor(self.weathers[idx], dtype=torch.long),
            "viewpoint": torch.tensor(self.viewpoints[idx], dtype=torch.float),
        }
        return data


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def transform(img, bbox, segmentation, mode):
        return img, bbox

    train_sequence = [0, 1, 2, 6, 10, 12]
    test_sequence = [3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 18, 19, 20, 21, 22]
    train_region = {"x": [-np.inf, 0.0], "y": [-np.inf, np.inf]}
    test_region = {"x": [0.0, np.inf], "y": [-np.inf, np.inf]}
    train_classes = ["Tree", "Pole", "Bollard"]
    test_classes = [
        "Informational_Sign",
        "Traffic_Sign",
        "Trash_Can",
        "Fire_Hydrant",
        "Emergency_Phone",
    ]
    # split = "sequence"
    # split = "region"
    split = "class"
    # split = "random"

    mode = "train"
    # mode = "test"

    dataset = CODataset(
        "data/CODa",
        mode=mode,
        split=split,
        transform=transform,
        # method="SupCon",
        method="Triplet",
        train_sequences=train_sequence,
        test_sequences=test_sequence,
        train_region=train_region,
        test_region=test_region,
        train_classes=train_classes,
        test_classes=test_classes,
    )

    for i in range(10):
        data = dataset[i]

        anc_img = data["images"][0][0]
        pos_img = data["images"][1][0]
        neg_img = data["images"][2][0]

        anc_bbox = data["images"][0][1]
        pos_bbox = data["images"][1][1]
        neg_bbox = data["images"][2][1]

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(anc_img)
        axes[0].axis("off")
        axes[0].set_title("Anchor")
        axes[0].add_patch(
            plt.Rectangle(
                (anc_bbox[0], anc_bbox[1]),
                anc_bbox[2] - anc_bbox[0],
                anc_bbox[3] - anc_bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        axes[1].imshow(pos_img)
        axes[1].axis("off")
        axes[1].set_title("Positive")
        axes[1].add_patch(
            plt.Rectangle(
                (pos_bbox[0], pos_bbox[1]),
                pos_bbox[2] - pos_bbox[0],
                pos_bbox[3] - pos_bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        axes[2].imshow(neg_img)
        axes[2].axis("off")
        axes[2].set_title("Negative")
        axes[2].add_patch(
            plt.Rectangle(
                (neg_bbox[0], neg_bbox[1]),
                neg_bbox[2] - neg_bbox[0],
                neg_bbox[3] - neg_bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )

        plt.savefig(f"output/{i}.png")
