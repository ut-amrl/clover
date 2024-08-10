"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Aug 7, 2024
Description: ScanNet dataset loader for CLOVER
"""

from loguru import logger
from typing import Literal, Callable
import pathlib
import json
from tqdm import tqdm
from collections import defaultdict
import random

import numpy as np
import cv2

from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    train_scenes_file = "data/ScanNet/index/scannet_train.txt"
    val_scenes_file = "data/ScanNet/index/scannet_val.txt"
    test_scenes_file = "data/ScanNet/index/scannet_test.txt"

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "val", "test"],
        transform: Callable,
        method: Literal["SupCon", "Triplet"] = None,
        n_imgs_per_instance: int = 8,
        **kwargs,
    ):
        self.name = "ScanNet"
        self.root_dir = pathlib.Path(root_dir)
        self.mode = mode
        self.method = method if mode == "train" else None
        self.transform = transform
        self.n_views = kwargs.get("n_views", 2)
        self.n_imgs_per_instance = n_imgs_per_instance

        # self.scene_ids = self._get_scene_ids()
        self.scene_ids = ["scene0000_00"]

        print("-" * 80)
        print(f"ScanNet dataset path: '{root_dir}'")
        print(f"- Mode: {mode}, Method: {method}, # of scenes: {len(self.scene_ids)}")

        self._load_dataset()

        if self.mode in ["val", "test"]:
            self._adjust_instance_images()
            print(
                f"Adjusted the number of images per instance: {len(self.image_files)}"
            )

        print("-" * 80)

    def _get_scene_ids(self):
        if self.mode == "train":
            with open(self.train_scenes_file, "r") as f:
                scene_ids = f.read().splitlines()
        elif self.mode == "val":
            with open(self.val_scenes_file, "r") as f:
                scene_ids = f.read().splitlines()
        elif self.mode == "test":
            with open(self.test_scenes_file, "r") as f:
                scene_ids = f.read().splitlines()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # NOTE: instance IDs are unique within the same scene
        # e.g., scene0000_00, scene0000_01 have different instance IDs for the same object
        unique_scene_ids = {}
        for scene_id in scene_ids:
            base_scene_id = scene_id.split("_")[0]
            if base_scene_id not in unique_scene_ids:
                unique_scene_ids[base_scene_id] = scene_id

        return list(unique_scene_ids.values())

    def _load_dataset(self):
        unique_label_map = {}
        unique_label = 0

        # {class: {label: [idx1, idx2, ...]}}
        self.instance_indices_map = defaultdict(lambda: defaultdict(list))
        self.image_files = []
        self.classes = []
        self.labels = []
        self.bboxes = []
        self.segmentations = []
        self.vectors = []

        for scene_id in tqdm(self.scene_ids, desc=f"Loading {self.mode} dataset"):
            annotation_file = self.root_dir / "2d_instance" / f"{scene_id}.json"
            if not annotation_file.exists():
                logger.warning(f"Annotation file not found: {annotation_file}")
                continue

            sens_dir = self.root_dir / "scans" / scene_id / "sens"
            with open(annotation_file, "r") as f:
                data = json.load(f)

                for frame_data in data["frames"]:
                    frame_id = frame_data["frameId"]
                    img_file = str(sens_dir / f"frame-{frame_id:06d}.color.jpg")

                    for instance_data in frame_data["instances"]:
                        class_name = instance_data["label"]
                        instance_id = instance_data["instanceId"]

                        if (scene_id, instance_id) not in unique_label_map:
                            unique_label_map[(scene_id, instance_id)] = unique_label
                            unique_label += 1
                        label = unique_label_map[(scene_id, instance_id)]

                        idx = len(self.image_files)
                        self.instance_indices_map[class_name][label].append(idx)
                        self.image_files.append(img_file)
                        self.classes.append(class_name)
                        self.labels.append(label)
                        self.bboxes.append(instance_data["bbox"])
                        self.segmentations.append(instance_data["segmentation"])

    def _adjust_instance_images(self):
        # filter out instances with less than n_imgs_per_instance images
        filtered_instance_indices_map = defaultdict(lambda: defaultdict(list))
        for class_name, label_map in self.instance_indices_map.items():
            for label, indices in label_map.items():
                if len(indices) < self.n_imgs_per_instance:
                    continue

                filtered_instance_indices_map[class_name][label] = random.sample(
                    indices, self.n_imgs_per_instance
                )

        # reorganize the data
        new_idx = 0
        for class_name, label_map in filtered_instance_indices_map.items():
            for label, indices in label_map.items():
                for idx in indices:
                    self.image_files[new_idx] = self.image_files[idx]
                    self.classes[new_idx] = class_name
                    self.labels[new_idx] = label
                    self.bboxes[new_idx] = self.bboxes[idx]
                    self.segmentations[new_idx] = self.segmentations[idx]
                    new_idx += 1

        self.instance_indices_map = filtered_instance_indices_map
        self.image_files = self.image_files[:new_idx]
        self.classes = self.classes[:new_idx]
        self.labels = self.labels[:new_idx]
        self.bboxes = self.bboxes[:new_idx]
        self.segmentations = self.segmentations[:new_idx]

    def get_positive_idx(self, idx):
        pos_cls = self.classes[idx]
        pos_label = self.labels[idx]
        return np.random.choice(self.instance_indices_map[pos_cls][pos_label], 1)[0]

    def get_negative_idx(self, idx):
        class_name = self.classes[idx]
        label = self.labels[idx]

        # Same class vs. different class for negative samples
        neg_cls = class_name
        if np.random.rand() < 0.5 or len(self.instance_indices_map[neg_cls]) < 2:
            potential_classes = [
                cls
                for cls in self.instance_indices_map.keys()
                if cls != class_name and len(self.instance_indices_map[cls]) > 1
            ]
            neg_cls = np.random.choice(potential_classes)

        # remove the positive label
        neg_labels = list(self.instance_indices_map[neg_cls].keys())
        if label in neg_labels:
            neg_labels.remove(label)

        neg_label = np.random.choice(neg_labels, 1)[0]
        return np.random.choice(self.instance_indices_map[neg_cls][neg_label], 1)[0]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        def load_img(img_file):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

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

        data = {
            "label": self.labels[idx],
            "class": self.classes[idx],
            "images": [
                self.transform(
                    img=load_img(self.image_files[i]),
                    bbox=self.bboxes[i],
                    segmentation=self.segmentations[i],
                    mode="train" if self.mode == "train" else "eval",
                )
                for i in indices
            ],
        }

        return data


if __name__ == "__main__":
    dataset = ScanNetDataset("data/ScanNet", "train")

    import matplotlib.pyplot as plt

    for i in range(10):
        data = dataset[i]
        img = data["anc_img"]
        pos_img = data["pos_img"]

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].axis("off")
        axes[0].set_title("Anchor")
        axes[0].add_patch(
            plt.Rectangle(
                (data["anc_bbox"][0], data["anc_bbox"][1]),
                data["anc_bbox"][2] - data["anc_bbox"][0],
                data["anc_bbox"][3] - data["anc_bbox"][1],
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
                (data["pos_bbox"][0], data["pos_bbox"][1]),
                data["pos_bbox"][2] - data["pos_bbox"][0],
                data["pos_bbox"][3] - data["pos_bbox"][1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        axes[2].imshow(data["neg_img"])
        axes[2].axis("off")
        axes[2].set_title("Negative")
        axes[2].add_patch(
            plt.Rectangle(
                (data["neg_bbox"][0], data["neg_bbox"][1]),
                data["neg_bbox"][2] - data["neg_bbox"][0],
                data["neg_bbox"][3] - data["neg_bbox"][1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )

        plt.show()
