"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Dec 14, 2023
Description: UT Campus Object Dataset (CODA)

Paper: https://arxiv.org/abs/2309.13549
"""

import sys
import pathlib
from natsort import natsorted
from typing import Literal
import json
from collections import defaultdict
import random
import yaml
import itertools
from typing import List, Tuple
from tqdm import tqdm
from operator import itemgetter
import math

import numpy as np
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.preprocess import (
    crop_image,
    augment_bbox,
    segment_to_mask,
    bbox_from_segment,
    random_erasing,
    joint_rotation,
)


class CODataset(Dataset):
    limited_classes_id = {
        "Tree": 0,
        "Pole": 1,
        "Bollard": 2,
        "Informational_Sign": 3,
        "Traffic_Sign": 4,
        "Trash_Can": 5,
        "Fire_Hydrant": 6,
        "Emergency_Phone": 7,
    }
    all_sequences = set(range(23)) - {8, 14, 15}
    weathers_id = {"sunny": 0, "cloudy": 1, "dark": 2, "rainy": 3}
    cams_id = {
        "cam0": 0,
        "cam1": 1,
    }
    extra_classes_id = {
        "Tree": 0,
        "Pole": 1,
        "Bollard": 2,
        "Informational_Sign": 3,
        "Traffic_Sign": 4,
        "Trash_Can": 5,
        "Fire_Hydrant": 6,
        "Emergency_Phone": 7,
    }

    def __init__(
        self,
        root_dir: str,
        mode: Literal["train", "val", "test"] = "train",
        transform: transforms.Compose = None,
        augmentation: bool = True,
        split_method: Literal["region", "sequence", "random", "class", "all"] = "all",
        image_type: Literal[
            "normal", "cropped", "fg+bg", "image+bbmask", "fg", "bg", "image+segmask"
        ] = "normal",
        sample_type: Literal["single", "n_views", "triplet"] = "single",
        num_positives: int = 1,
        seed: int = 42,
        extension: str = "png",
        **kwargs,
    ):
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"

        self.root_dir = pathlib.Path(root_dir)
        self.images_dir = self.root_dir / "2d_raw"
        self.annotations_dir = self.root_dir / "annotations"
        self.global_instances_dir = self.root_dir / "3d_bbox" / "global"
        self.calibrations_dir = self.root_dir / "calibrations"

        if split_method == "class":
            self.classes_id = self.extra_classes_id
        else:
            self.classes_id = self.limited_classes_id

        self.classes_name = {v: k for k, v in self.classes_id.items()}

        self.mode = mode
        self.split_method = split_method
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.augmentation = augmentation
        self.image_type = image_type
        self.sample_type = sample_type
        self.num_positives = num_positives
        self.skip = kwargs.get(f"{mode}_skip", 1)
        self.seed = seed
        self.extension = extension

        self.crop = kwargs.get("crop", True)
        self.margin = kwargs.get("margin", 10)
        self.square = kwargs.get("square", True)

        # Set seed
        random.seed(seed)
        np.random.seed(seed)

        print("-" * 70)
        print(f"Loading CODa dataset from {root_dir}")
        print(f"- Mode: {mode}")
        print(f"- Split Method: {split_method}")
        if split_method == "region":
            print(f"- Train Region: {kwargs.get('train_region')}")
            print(f"- Test Region: {kwargs.get('test_region')}")
        elif split_method == "sequence":
            print(f"- Train Sequences: {kwargs.get('train_sequences')}")
            print(f"- Test Sequences: {kwargs.get('test_sequences')}")
        elif split_method == "random":
            print(f"- Train Ratio: {kwargs.get('train_ratio')}")
        elif split_method == "class":
            print(f"- Train Classes: {kwargs.get('train_classes')}")
            print(f"- Test Classes: {kwargs.get('test_classes')}")
        print(f"- Image Type: {image_type}")
        print(f"- Sample Type: {sample_type}")
        print(f"- # Positives: {num_positives}")
        print(f"- Skip: {kwargs.get(f'{mode}_skip', 1)}")
        print(f"- Use Augmentations: {self.augmentation}")
        print(f"- crop: {self.crop}, margin: {self.margin}, square: {self.square}")
        print("-" * 70)

        # Region-based split
        self.train_region = kwargs.get(
            "train_region", {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}
        )
        self.test_region = kwargs.get(
            "test_region", {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}
        )
        # Random split
        self.train_ratio = kwargs.get("train_ratio", 0.7)
        # Sequence-based split
        self.train_sequences = kwargs.get("train_sequences", [])
        self.test_sequences = kwargs.get("test_sequences", [])
        # Classes-based split
        self.train_classes = kwargs.get("train_classes", list(self.classes_id.keys()))
        self.test_classes = kwargs.get("test_classes", list(self.classes_id.keys()))

        # Load dataset
        self._load_global_instances()
        self._load_dataset()
        self._load_camera_parameters()
        if self.split_method == "random":
            self._random_split()

        # Print dataset statistics
        print("class", set(self.classes))
        print("instance", len(set(self.labels)))
        for class_name, class_id in self.classes_id.items():
            total = len(self.global_instances[class_id])
            selected = len(self.instance_indices[class_id])
            images = sum(
                len(indices) for indices in self.instance_indices[class_id].values()
            )
            print(
                f" ({class_name})".ljust(25),
                f"({selected} / {total}) instances".ljust(25),
                f"({images}) images",
            )
        print("-" * 70)

    def _load_global_instances(self):
        classes_json = {
            class_name: json.load(
                open(self.global_instances_dir / f"{class_name}.json", "r")
            )
            for class_name in self.classes_id.keys()
        }

        # {class_id: {instance_id: [x, y, z, h, l, w, r, p, y]}}
        self.global_instances = defaultdict(lambda: defaultdict(list))
        count = 0
        for class_name, instance_json in classes_json.items():
            class_id = self.classes_id[class_name]
            for instance in instance_json["instances"]:
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

        print(f"Loaded {count} global instances")

    def _load_dataset(self):
        print("Loading dataset images...")

        # Map unique label to class_id and instance_id
        unique_label_map = {}
        unique_label = 0

        # Length is # bb
        self.image_files = []
        self.classes = []  # class_id
        self.instances = []  # instance_id
        self.labels = []  # unique_label_idx
        self.bboxes = []  # [x1, y1, x2, y2]
        self.segmentations = []  # [[x1, y1, x2, y2, ...], ...]
        self.weathers = []
        self.sequences = []
        self.cams = []  # 0: cam0, 1: cam1
        self.poses = []  # [x, y, z, qw, qx, qy, qz]
        self.vectors = []  # [x, y, z] (cam to object vector)

        # Gives index for observations of an object with given class_id and instance id
        self.instance_indices = defaultdict(lambda: defaultdict(list))

        annotation_files = []
        for cam in self.cams_id.keys():
            for seq_dir in natsorted(
                [d for d in self.annotations_dir.glob(f"{cam}/*")]
            ):
                if self.split_method == "sequence":
                    seq = int(seq_dir.name)
                    if self.mode == "train" and seq not in self.train_sequences:
                        continue
                    elif (
                        self.mode in ["val", "test"] and seq not in self.test_sequences
                    ):
                        continue

                # Anno files is per image -- need to associate image id to each instance
                for annotation_file in natsorted([f for f in seq_dir.glob("*.json")]):
                    annotation_files.append(annotation_file)

        if self.mode in ["val", "test"]:
            random.shuffle(annotation_files)
            if self.mode == "val":
                annotation_files = annotation_files[: len(annotation_files) // 2]
            else:
                annotation_files = annotation_files[len(annotation_files) // 2 :]

        print(f"Total {len(annotation_files)} annotation files")

        # different starting index for train and test
        for anno_file in tqdm(
            annotation_files[:: self.skip], leave=False, disable=not sys.stderr.isatty()
        ):
            try:
                annotation = json.load(open(anno_file, "r"))
                info = annotation["info"]
                image_file = (
                    self.images_dir
                    / info["camera"]
                    / str(info["sequence"])
                    / (info["image_file"] + f".{self.extension}")
                )

                for instance in annotation["instances"]:
                    class_name = instance["class"]
                    if class_name not in self.classes_id:
                        continue

                    class_id = self.classes_id[class_name]
                    instance_id = int(instance["id"])

                    # Classes-based split
                    if self.split_method == "class":
                        if (
                            self.mode == "train"
                            and class_name not in self.train_classes
                        ):
                            continue
                        elif (
                            self.mode in ["val", "test"]
                            and class_name not in self.test_classes
                        ):
                            continue

                    cX = self.global_instances[class_id][instance_id][0]
                    cY = self.global_instances[class_id][instance_id][1]
                    cZ = self.global_instances[class_id][instance_id][2]

                    # Region-based split
                    if self.split_method == "region":
                        if self.mode == "train" and (
                            cX < self.train_region["x_min"]
                            or cX > self.train_region["x_max"]
                            or cY < self.train_region["y_min"]
                            or cY > self.train_region["y_max"]
                        ):
                            continue

                        elif self.mode in ["val", "test"] and (
                            cX < self.test_region["x_min"]
                            or cX > self.test_region["x_max"]
                            or cY < self.test_region["y_min"]
                            or cY > self.test_region["y_max"]
                        ):
                            continue

                    # Assign unique label
                    if (class_id, instance_id) not in unique_label_map:
                        unique_label_map[(class_id, instance_id)] = unique_label
                        unique_label += 1

                    # Append data
                    self.image_files.append(str(image_file))
                    self.classes.append(class_id)
                    self.instances.append(instance_id)
                    self.labels.append(unique_label_map[(class_id, instance_id)])

                    self.bboxes.append(instance["bbox"])
                    self.segmentations.append(instance["segmentation"])

                    self.weathers.append(self.weathers_id[info["weather_condition"]])
                    self.sequences.append(int(info["sequence"]))
                    self.cams.append(self.cams_id[cam])
                    self.poses.append(info["pose"])
                    self.vectors.append(
                        np.array([cX, cY, cZ]) - np.array(info["pose"][:3]).tolist()
                    )

                    # map for same instances for positive sampling
                    index = len(self.image_files) - 1
                    self.instance_indices[class_id][instance_id].append(index)
            except Exception as e:
                print(f"Error: {e} - {anno_file}")
                continue

        print(f"Loaded {len(self)} images")

    def _random_split(self):
        new_instance_indices = defaultdict(lambda: defaultdict(list))

        selected_indices = []
        last_index = 0
        for class_id, instances in self.instance_indices.items():
            # Shuffle instances from global instances
            all_instances = list(self.global_instances[class_id].keys())
            random.shuffle(all_instances)

            num_train = int(len(all_instances) * self.train_ratio)
            train_instances = all_instances[:num_train]
            test_instances = all_instances[num_train:]
            selected_instances = (
                train_instances if self.mode == "train" else test_instances
            )

            # Update instance indices
            for instance_id in selected_instances:
                if instance_id not in instances or len(instances[instance_id]) == 0:
                    continue

                new_instance_indices[class_id][instance_id] = range(
                    last_index, last_index + len(instances[instance_id])
                )
                selected_indices.extend(instances[instance_id])
                last_index += len(instances[instance_id])

        self.instance_indices = new_instance_indices
        # Update dataset
        selector = itemgetter(*selected_indices)
        self.image_files = selector(self.image_files)
        self.classes = selector(self.classes)
        self.instances = selector(self.instances)
        self.labels = selector(self.labels)
        self.bboxes = selector(self.bboxes)
        self.segmentations = selector(self.segmentations)
        self.weathers = selector(self.weathers)
        self.sequences = selector(self.sequences)
        self.cams = selector(self.cams)
        self.poses = selector(self.poses)
        self.vectors = selector(self.vectors)

    def _load_camera_parameters(self):
        self._cam_params = defaultdict(lambda: defaultdict(dict))

        # {cam_id: {seq_id: {K: intrinsic, img_size, D: distortion}}}
        seq_dirs = natsorted([d for d in self.calibrations_dir.glob("*")])
        for seq_dir in seq_dirs:
            for cam, cam_id in self.cams_id.items():
                cam_param_file = seq_dir / f"calib_{cam}_intrinsics.yaml"
                param = self.load_camera_params(cam_param_file)
                self._cam_params[cam_id][int(seq_dir.name)] = param
        print("Loaded camera parameters")

    def get_cam_params(self, idx):
        cam_id = self.cams[idx]
        seq_id = self.sequences[idx]
        return self._cam_params[cam_id][seq_id]

    @staticmethod
    def load_camera_params(cam_param_file: str) -> dict:
        with open(cam_param_file, "r") as f:
            params = yaml.safe_load(f)
            intrinsic = np.array(params["camera_matrix"]["data"]).reshape(3, 3)
            image_size = (params["image_width"], params["image_height"])
            distortion = np.array(params["distortion_coefficients"]["data"])

        return {"K": intrinsic, "img_size": image_size, "D": distortion}

    def process_img_crop(
        self,
        img: Image,
        bbox: List[int],
        margin: int = 10,
        square: bool = False,
    ):
        """Process the image by cropping it with a bounding box"""
        # augment (shift bbox, vary margin, offset)
        offset = (0.0, 0.0)
        if self.mode == "train" and self.augmentation:
            img = random_erasing(img, bbox)  # Random Erasing
            bbox = augment_bbox(bbox, img.size)
            margin *= random.randint(0, 5)
            offset = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))

        # Crop image
        img = crop_image(img, bbox, margin=margin, square=square, offset=offset)[0]
        return self.transform(img)

    def process_img_fg_bg(
        self,
        img: Image,
        segmentations: List[List[int]],
        crop: bool = False,
        margin: int = 0,
        square: bool = False,
    ):
        """Crop image and split foreground and background"""
        img_np = np.array(img)

        # create mask from segmentations
        mask = segment_to_mask(segmentations, img.size)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        bbox = bbox_from_segment(segmentations)

        # split foreground and background
        img_fg = Image.fromarray(img_np * mask)
        img_bg = Image.fromarray(img_np * (~mask))

        # crop image
        if crop:
            img_fg = crop_image(img_fg, bbox, margin=margin, square=square)[0]
            img_bg = crop_image(img_bg, bbox, margin=margin, square=square)[0]

        return self.transform(img_fg), self.transform(img_bg)

    def process_img_bbmask(
        self,
        img: Image,
        bbox: List[int],
        crop: bool = False,
        margin: int = 10,
        square: bool = False,
    ):
        """Resize image and mask to the same size (with data augmentation)"""

        # Data Augmentation
        offset = (0.0, 0.0)
        if self.mode == "train" and self.augmentation:
            img = random_erasing(img, bbox)  # Random Erasing
            bbox = augment_bbox(bbox, img.size)
            margin *= random.randint(0, 5)
            offset = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))

        # Create mask
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)

        # Data Augmentation
        if self.mode == "train" and self.augmentation:
            img, mask = joint_rotation(img, mask)  # Random Rotate

        # Crop image around the bbox
        if crop:
            img, crop_bbox = crop_image(
                img, bbox, margin=margin, square=square, offset=offset
            )
            mask = mask.crop(crop_bbox)

        # transform and match mask size to image size
        transformed_img = self.transform(img)
        resized_mask = mask.resize(
            (transformed_img.shape[2], transformed_img.shape[1]), Image.NEAREST
        )
        resized_mask = torch.from_numpy(np.array(resized_mask) / 255.0)
        resized_mask = (resized_mask > 0.5).long().unsqueeze(0)

        return transformed_img, resized_mask

    def process_img_segmask(
        self,
        img: Image,
        segmentations: List[List[int]],
        crop: bool = False,
        margin: int = 10,
        square: bool = False,
    ):
        """Resize image and mask to the same size (with data augmentation)"""

        # Data Augmentation
        bbox = bbox_from_segment(segmentations)
        offset = (0.0, 0.0)
        if self.mode == "train" and self.augmentation:
            img = random_erasing(img, bbox)  # Random Erasing
            bbox = augment_bbox(bbox, img.size)
            margin *= random.randint(0, 5)
            offset = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))

        # Create Mask from segmentations
        mask = segment_to_mask(segmentations, img.size).astype(np.uint8) * 255
        mask = Image.fromarray(mask, "L")

        # Data Augmentation
        if self.mode == "train" and self.augmentation:
            img, mask = joint_rotation(img, mask)  # Random Rotate

        # Crop image around the segmentation
        if crop:
            img, crop_bbox = crop_image(
                img, bbox, margin=margin, square=square, offset=offset
            )
            mask = mask.crop(crop_bbox)

        # transform and match mask size to image size
        transformed_img = self.transform(img)
        resized_mask = mask.resize(
            (transformed_img.shape[2], transformed_img.shape[1]), Image.NEAREST
        )
        resized_mask = torch.from_numpy(np.array(resized_mask) / 255.0)
        resized_mask = (resized_mask > 0.5).long().unsqueeze(0)

        return transformed_img, resized_mask

    def get_positive_indices(self, idx, num=1):
        class_id = self.classes[idx]
        instance_id = self.instances[idx]
        return np.random.choice(self.instance_indices[class_id][instance_id], num)

    def get_negative_idx(self, idx):
        class_id = self.classes[idx]
        instance_id = self.instances[idx]
        classes = self.instance_indices.keys()

        if random.random() < 0.3:
            neg_cls = random.choice([cls for cls in classes if cls != class_id])
        else:
            neg_cls = class_id

        instances = list(self.instance_indices[neg_cls].keys())
        if neg_cls == class_id:
            instances.remove(instance_id)

        if len(instances) == 0:
            neg_cls = random.choice([cls for cls in classes if cls != class_id])
            instances = list(self.instance_indices[neg_cls].keys())

        neg_instance_id = random.choice(instances)
        return np.random.choice(self.instance_indices[neg_cls][neg_instance_id])

    def _preprocess_img(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")

        if self.image_type == "normal":
            return [self.transform(img)]
        elif self.image_type == "cropped":
            return [
                self.process_img_crop(
                    img,
                    self.bboxes[idx],
                    margin=self.margin,
                    square=self.square,
                )
            ]
        elif self.image_type in ["fg+bg", "fg", "bg"]:
            return self.process_img_fg_bg(
                img,
                self.segmentations[idx],
                crop=self.crop,
                margin=self.margin,
                square=self.square,
            )
        elif self.image_type == "image+bbmask":
            return self.process_img_bbmask(
                img,
                self.bboxes[idx],
                crop=self.crop,
                margin=self.margin,
                square=self.square,
            )
        elif self.image_type == "image+segmask":
            return self.process_img_segmask(
                img,
                self.segmentations[idx],
                crop=self.crop,
                margin=self.margin,
                square=self.square,
            )
        else:
            raise ValueError(f"Invalid Type: {self.image_type}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        info = {
            "class": self.classes[idx],
            "instance": self.instances[idx],
            "sequence": self.sequences[idx],
            "weather": self.weathers[idx],
        }

        if self.sample_type == "single" or self.mode in ["val", "test"]:
            images = self._preprocess_img(idx)
            vectors = torch.tensor(self.vectors[idx]).unsqueeze(0)
        elif self.sample_type == "n_views":
            pos_indices = self.get_positive_indices(idx, self.num_positives)
            images = [
                image for i in [idx, *pos_indices] for image in self._preprocess_img(i)
            ]
            vectors = torch.stack(
                [torch.tensor(self.vectors[i]) for i in [idx, *pos_indices]]
            )
        elif self.sample_type == "triplet":
            pos_idx = self.get_positive_indices(idx)[0]
            neg_idx = self.get_negative_idx(idx)
            images = [
                image
                for i in [idx, pos_idx, neg_idx]
                for image in self._preprocess_img(i)
            ]
            vectors = torch.stack(
                [torch.tensor(self.vectors[i]) for i in [idx, pos_idx, neg_idx]]
            )
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

        return images, self.labels[idx], vectors, info

    def get_indices_for_localization(
        self, sequences: List[int], min_instances: int = 3
    ):
        """Get indices of images with more than min_instances"""
        indices_per_image = defaultdict(list)
        for idx in range(len(self)):
            img_file = self.image_files[idx]
            seq = self.sequences[idx]
            if seq in sequences:
                indices_per_image[img_file].append(idx)

        # [ [img1_inst1_idx, img1_inst2_idx, ...], [img2_inst1_idx, ...], ...]
        indices = []
        for idxs in indices_per_image.values():
            if len(idxs) >= min_instances:
                indices.append(idxs)

        return indices

    def get_statistics(self):
        """Get Dataset Statistics"""
        print(f"\n=============== CODa {self.mode} Dataset Statistics ============")
        output_dir = pathlib.Path("results/statistics")
        output_dir.mkdir(parents=True, exist_ok=True)

        instance_weather_counts = defaultdict(lambda: defaultdict(dict))
        num_images_per_instance = defaultdict(dict)
        num_bboxes_per_image = defaultdict(int)

        for idx in range(len(self)):
            class_id = self.classes[idx]
            instance_id = self.instances[idx]
            weather_id = self.weathers[idx]

            instance_weather_counts[class_id][weather_id][instance_id] = (
                instance_weather_counts[class_id][weather_id].get(instance_id, 0) + 1
            )
            num_images_per_instance[class_id][instance_id] = (
                num_images_per_instance[class_id].get(instance_id, 0) + 1
            )
            num_bboxes_per_image[self.image_files[idx]] += 1

        # Number of bboxes per images
        num_bbox_data = list(num_bboxes_per_image.values())
        # visualize_cdf(
        #     {"Number of Bboxes": list(num_bboxes_per_image.values())},
        #     title="Number of Bboxes per Image",
        #     xlabel="Bboxes",
        #     ylabel="CDF",
        #     out_file=output_dir / f"{self.name}_bboxes.png",
        # )
        for num_bbox in range(1, max(num_bbox_data) + 1):
            print(f"{num_bbox} Bboxes: {num_bbox_data.count(num_bbox) / len(self)}")

        for class_name, class_id in self.classes_id.items():
            print(f"Class: {class_name}")
            print(f"Number of Instances: {len(num_images_per_instance[class_id])}")

            # Image Distribution per Instance
            n_images = list(num_images_per_instance[class_id].values())
            print("\nImage Distribution per Instance")
            print(
                f"avg:{np.mean(n_images):.2f}\t",
                f"std:{np.std(n_images):.2f}\t",
                f"total:{np.sum(n_images)}",
            )

            # Weather Condition Distribution per Instance
            print("\nWeather Condition Distribution")
            for weather_name, weather_id in self.weathers_id.items():
                n_images = list(instance_weather_counts[class_id][weather_id].values())
                if len(n_images) == 0:
                    continue
                print(
                    f" ({weather_name})".ljust(15),
                    f"avg:{np.mean(n_images):.2f}\t",
                    f"std:{np.std(n_images):.2f}\t",
                    f"total:{np.sum(n_images)}\t",
                    f"n:{len(n_images)}",
                )

            # # Pairwise view difficulty
            # test_cases = {
            #     "easy": (0,),
            #     "medium": (1,),
            #     "hard": (2,),
            #     "easy+medium": (0, 1),
            #     "medium+hard": (1, 2),
            #     "all": (0, 1, 2),
            # }
            # counts = {case: [] for case in test_cases.keys()}
            # distances = []
            # angles = []
            # for instance_id, indices in tqdm(
            #     self.instance_indices[class_id].items(),
            #     leave=False,
            #     disable=not sys.stderr.isatty(),
            # ):
            #     if len(indices) < 2:
            #         continue

            #     # Pairwise view difficulty
            #     tmp_counts = {case: 0 for case in test_cases.keys()}
            #     pairs = list(itertools.combinations(indices, 2))
            #     for idx1, idx2 in pairs:
            #         vector1 = np.array(self.vectors[idx1])
            #         vector2 = np.array(self.vectors[idx2])
            #         d = abs(np.linalg.norm(vector1) - np.linalg.norm(vector2))
            #         a = np.degrees(
            #             np.arccos(
            #                 np.dot(vector1, vector2)
            #                 / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            #             )
            #         )

            #         distances.append(d)
            #         angles.append(a)

            #         difficulty = None
            #         if d < 10.0 and a < 10.0:
            #             difficulty = 0
            #         elif d < 50.0 and a < 90.0:
            #             difficulty = 1
            #         else:
            #             difficulty = 2

            #         for case, condition in test_cases.items():
            #             if difficulty in condition:
            #                 tmp_counts[case] += 1

            #     for case, count in tmp_counts.items():
            #         counts[case].append(count)

            # print("\nPairwise View Difficulty")
            # for case, case_counts in counts.items():
            #     print(
            #         f" ({case})".ljust(15),
            #         f"avg:{np.mean(case_counts):.2f}\t",
            #         f"std:{np.std(case_counts):.2f}\t",
            #         f"total:{np.sum(case_counts)}\t",
            #         f"n:{len(case_counts)}",
            #     )
            # print("-------------------------------------------------------------------")

            # # plot histogram (x: distance, y: angle, z: count)
            # distances = np.array(distances)
            # angles = np.array(angles)
            # distance_bins = np.linspace(distances.min(), distances.max(), 30)
            # angle_bins = np.linspace(angles.min(), angles.max(), 30)

            # H, distance_edges, angle_edges = np.histogram2d(
            #     distances, angles, bins=(distance_bins, angle_bins)
            # )

            # # Create a 3D plot
            # fig = plt.figure(figsize=(60, 40))
            # ax = fig.add_subplot(111, projection="3d")

            # for i in range(len(distance_edges) - 1):
            #     for j in range(len(angle_edges) - 1):
            #         # Calculate the center of each bin to place the bar correctly
            #         x = (distance_edges[i] + distance_edges[i + 1]) / 2
            #         y = (angle_edges[j] + angle_edges[j + 1]) / 2
            #         z = 0  # Base of the bar

            #         dx = (distance_edges[i + 1] - distance_edges[i]) * 0.8
            #         dy = (angle_edges[j + 1] - angle_edges[j]) * 0.8
            #         dz = H[i, j]  # Height of the bar (count)

            #         if dz > 0:  # Only plot bars with non-zero height
            #             ax.bar3d(x, y, z, dx, dy, dz)

            # ax.set_title(f"# of Pairs per View Difficulty ({class_name})")
            # ax.set_xlabel("Distance (m)")
            # ax.set_ylabel("Angle (deg)")
            # ax.set_zlabel("Count")
            # ax.set_xticks(distance_edges)
            # ax.set_yticks(angle_edges)
            # fig.savefig(output_dir / f"{self.name}_{class_name}_pairwise.png")
            # plt.close()


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = CODataset(
        root_dir="/home/dongmyeong/Projects/datasets/CODa",
        mode="test",
        split_method="region",
        transform=transform,
        augmentation=True,
        image_type="fg+bg",
        # image_type="cropped",
        sample_type="triplet",
        num_positives=1,
        extension="jpg",
        train_ratio=0.7,
        skip=1,
        train_sequences=range(22),
        test_sequences=[],
        train_region={
            "x_min": -float("inf"),
            "x_max": 0,
            "y_min": -float("inf"),
            "y_max": float("inf"),
        },
        test_region={
            "x_min": 0,
            "x_max": float("inf"),
            "y_min": -float("inf"),
            "y_max": float("inf"),
        },
    )

    dataset.get_statistics()
