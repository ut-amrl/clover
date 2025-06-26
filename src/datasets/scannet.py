import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from loguru import logger
from rich import print as rprint
from torch.utils.data import Dataset
from tqdm import tqdm


class ScanNetDataset(Dataset):
    """ScanNet dataset for object re-identification"""

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "val", "test"],
        transform: Callable,
        frame_gap: int,
        method: Literal["SupCon", "Triplet"] | None = None,
        **kwargs,
    ):
        self.name = "ScanNet"
        self.mode = mode
        self.root_dir = Path(root_dir)

        self.transform = transform
        self.frame_gap = 1 if mode == "train" else frame_gap
        self.method = method if mode == "train" else None
        self.n_views = kwargs.get("n_views", 2)

        scene_index_file = kwargs.get(f"{mode}_index_file", None)
        scene_ids = self._get_scene_ids(scene_index_file)

        self._load_dataset(scene_ids)

        rprint("-" * 80)
        rprint(f"ScanNet dataset path: '{root_dir}'")
        rprint(f"mode: [bold]{mode}[/bold], frame gap: {self.frame_gap}")
        if self.method:
            rprint(f"method: [bold]{self.method}[/bold]")
        rprint(f"# of scenes: {len(scene_ids)}")
        rprint(
            f"# of images / classes / instances: "
            f"{len(self.image_files)} / {len(set(self.classes))} / {len(set(self.labels))}"
        )
        rprint("-" * 80)

    def _get_scene_ids(self, scene_index_file):
        if not scene_index_file or not Path(scene_index_file).exists():
            raise FileNotFoundError(
                f"Scene index file for mode '{self.mode}' not found: {scene_index_file}"
            )

        with open(scene_index_file, "r") as f:
            scene_ids = f.read().splitlines()

        # NOTE: In ScanNet, instance IDs are unique within the same scene
        # e.g., scene0000_00, scene0000_01 have different instance IDs for the same object
        # Therefore, we keep only one scene ID per base scene
        return list({scene_id.split("_")[0]: scene_id for scene_id in scene_ids}.values())

    def _load_dataset(self, scene_ids):
        unique_label_map = {}
        unique_label = 0
        class_name_id_map = {}

        # {class: {label: [idx1, idx2, ...]}}
        tmp_instance_data = defaultdict(lambda: defaultdict(list))

        # For skipping near-duplicate frames in val/test
        instance_frame_memory = defaultdict(list)

        for scene_id in tqdm(scene_ids, desc=f"Loading {self.mode} dataset"):
            annotation_file = self.root_dir / "2d_instance" / f"{scene_id}.json"
            if not annotation_file.exists():
                rprint("[red]Warning:[/red] Annotation file not found:", annotation_file)
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

                    # Skip near-duplicate frames in val/test
                    # NOTE: To make task non-trivial,
                    if self.mode in ["val", "test"]:
                        prev_frames = instance_frame_memory.get(label, [])
                        if any(abs(pf - frame_id) <= self.frame_gap for pf in prev_frames):
                            continue
                        instance_frame_memory[label].append(frame_id)

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
        filtered_cnt = 0
        for class_id, label_map in tmp_instance_data.items():
            for label, samples in label_map.items():
                if len(samples) < 2:
                    filtered_cnt += 1
                    continue  # skip singleton instances

                for sample in samples:
                    idx = len(self.image_files)
                    self.image_files.append(sample["img_file"])
                    self.classes.append(sample["class_id"])
                    self.labels.append(sample["label"])
                    self.bboxes.append(sample["bbox"])
                    self.segmentations.append(sample["segmentation"])
                    self.instance_indices_map[class_id][label].append(idx)

        if filtered_cnt > 0:
            logger.warning(
                f"Filtered out {filtered_cnt} singleton instances. "
                "These instances have less than 2 samples in the dataset."
            )

    def get_positive_idx(self, idx):
        pos_cls = self.classes[idx]
        pos_label = self.labels[idx]
        candidates = [i for i in self.instance_indices_map[pos_cls][pos_label] if i != idx]
        if len(candidates) < 1:
            logger.error(
                f"No positive samples found for index {idx} (class: {pos_cls}, label: {pos_label})."
            )
            return idx
        return np.random.choice(candidates)

    def get_negative_idx(self, idx):
        anchor_cls = self.classes[idx]
        anchor_label = self.labels[idx]

        # All available labels for this class except the anchor's label
        neg_labels = [l for l in self.instance_indices_map[anchor_cls].keys() if l != anchor_label]
        if neg_labels:
            # same class, different instance
            neg_cls = anchor_cls
            neg_label = np.random.choice(neg_labels)
        else:
            # no other labels in the same class, choose a different class
            other_classes = [c for c in self.instance_indices_map.keys() if c != anchor_cls]
            neg_cls = np.random.choice(other_classes)
            neg_label = np.random.choice(list(self.instance_indices_map[neg_cls].keys()))
        candidates = self.instance_indices_map[neg_cls][neg_label]
        return np.random.choice(candidates)

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

        images = []
        for i in indices:
            img = load_img(self.image_files[i])
            transformed = self.transform(
                img=img,
                bbox=self.bboxes[i],
                segmentation=self.segmentations[i],
                mode="train" if self.mode == "train" else "eval",
            )
            images.append(transformed)

        data = {"images": images, "label": self.labels[idx], "class": self.classes[idx]}
        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def transform(img, bbox, segmentation, mode):
        return img, bbox

    dataset = ScanNetDataset("data/ScanNet", "train", transform=transform, method="Triplet")

    for i in range(1000):
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]

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

        plt.tight_layout()
        plt.savefig(f"figure/scannet_sample_{i}.png")
        plt.close(fig)
