import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


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
        **kwargs,
    ):
        self.name = "ScanNet"
        self.mode = mode
        self.root_dir = Path(root_dir)

        self.method = method if mode == "train" else None
        self.transform = transform
        self.n_views = kwargs.get("n_views", 2)

        self.scene_ids = self._get_scene_ids()

        print("-" * 80)
        print(f"ScanNet dataset path: '{root_dir}'")
        print(f"- Mode: {mode}, Method: {method}, # of scenes: {len(self.scene_ids)}")

        self._load_dataset()

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
        class_name_id_map = {}

        # {class: {label: [idx1, idx2, ...]}}
        tmp_instance_data = defaultdict(lambda: defaultdict(list))

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

                    # assign globally unique label for the instance across all scenes
                    label = unique_label_map.get((scene_id, instance_id), None)
                    if label is None:
                        label = unique_label
                        unique_label_map[(scene_id, instance_id)] = label
                        unique_label += 1

                    # assign globally unique class ID
                    class_id = class_name_id_map.setdefault(class_name, len(class_name_id_map))

                    tmp_instance_data[class_id][label].append(
                        {
                            "img_file": img_file,
                            "class_id": class_id,
                            "label": label,
                            "bbox": instance_data["bbox"],
                            "segmentation": instance_data["segmentation"],
                        }
                    )

        self.instance_indices_map = defaultdict(lambda: defaultdict(list))
        self.image_files = []
        self.classes, self.labels, self.bboxes, self.segmentations = [], [], [], []
        for class_id, label_map in tmp_instance_data.items():
            for label, samples in label_map.items():
                if len(samples) < 2:
                    continue  # skip singleton instances

                for sample in samples:
                    idx = len(self.image_files)
                    self.image_files.append(sample["img_file"])
                    self.classes.append(sample["class_id"])
                    self.labels.append(sample["label"])
                    self.bboxes.append(sample["bbox"])
                    self.segmentations.append(sample["segmentation"])
                    self.instance_indices_map[class_id][label].append(idx)

    def get_positive_idx(self, idx):
        pos_cls = self.classes[idx]
        pos_label = self.labels[idx]
        return np.random.choice(self.instance_indices_map[pos_cls][pos_label], 1)[0]

    def get_negative_idx(self, idx):
        # Same class vs. different class for negative samples
        neg_cls = self.classes[idx]
        if len(self.instance_indices_map[neg_cls]) < 2 or np.random.rand() < 0.5:
            potential_classes = [
                cls
                for cls in self.instance_indices_map.keys()
                if cls != self.classes[idx] and len(self.instance_indices_map[cls]) > 1
            ]
            neg_cls = np.random.choice(potential_classes)

        # remove the positive label
        neg_labels = list(self.instance_indices_map[neg_cls].keys())
        if self.labels[idx] in neg_labels:
            neg_labels.remove(self.labels[idx])

        neg_label = np.random.choice(neg_labels, 1)[0]
        return np.random.choice(self.instance_indices_map[neg_cls][neg_label], 1)[0]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """get positive and negative samples for the given index
        'SupCon' method: returns multiple views of the same instance (view1, view2, ...)
        'Triplet' method: returns (anchor, positive, negative)
        """
        indices = [idx]
        if self.mode == "train":
            if self.method == "SupCon":
                indices.extend([self.get_positive_idx(idx) for _ in range(self.n_views - 1)])
            elif self.method == "Triplet":
                pos_idx = self.get_positive_idx(idx)
                neg_idx = self.get_negative_idx(idx)
                indices.extend([pos_idx, neg_idx])
            else:
                raise ValueError(f"Not supported method: {self.method}")

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
        }

        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def transform(img, bbox, segmentation, mode):
        return img, bbox

    dataset = ScanNetDataset("data/ScanNet", "train", transform=transform, method="Triplet")

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

        plt.show()
