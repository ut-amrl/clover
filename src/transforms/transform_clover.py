from typing import Literal, List, Tuple
import random
import math

import numpy as np
import cv2
from PIL import Image

from torchvision import transforms

from .common import crop_image


class TransformCLOVER:
    def __init__(
        self,
        img_size: int,
        margin: int = 10,
        square: bool = True,
        augmentation: bool = True,
    ):
        self.img_size = img_size
        self.margin = margin
        self.square = square
        self.augmentation = augmentation

        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.img_size, interpolation=Image.BICUBIC),
                transforms.ColorJitter(
                    brightness=(0.6, 1.4),
                    contrast=(0.6, 1.4),
                    saturation=(0.6, 1.4),
                    hue=(-0.2, 0.2),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.img_size, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, img, bbox, mode: Literal["train", "eval"], **kwargs):
        if mode == "train" and self.augmentation:
            img = random_erasing(img, bbox)
            bbox = augment_bbox(bbox, img.shape[:2])
            img, bbox = rotate_image_and_bbox(img, bbox, max_angle=10)

        img = crop_image(img, bbox, self.margin, self.square)

        return (
            self.train_transform(img) if mode == "train" else self.eval_transform(img)
        )


### augmentation functions


def random_erasing(img, bbox: List[int], p=0.8, s=(0.1, 0.3), r=(0.3, 3), value=None):
    """Randomly erase rectangle region in an image except for the bbox region"""
    assert p >= 0 and p <= 1, "p must be in [0, 1]"
    assert s[0] >= 0 and s[1] <= 1, "s must be in [0, 1]"
    assert s[1] > s[0] and r[1] > r[0], "s and r must be in increasing order"
    assert not value or (value >= 0 and value <= 255), "value must be in [0, 255]"

    if random.random() > p:
        return img

    imgH, imgW, imgC = img.shape
    area = imgH * imgW
    log_r = [math.log(r[0]), math.log(r[1])]
    for _ in range(10):
        erase_area = area * random.uniform(s[0], s[1])
        aspect_ratio = math.exp(random.uniform(*log_r))

        h = int(round(math.sqrt(erase_area * aspect_ratio)))
        w = int(round(math.sqrt(erase_area / aspect_ratio)))
        if not (h < imgH and w < imgW):
            continue

        # random left-top point
        y = random.randint(0, imgH - h)
        x = random.randint(0, imgW - w)
        erase_bbox = [x, y, x + w, y + h]

        is_overlap = (
            erase_bbox[2] > bbox[0]
            and erase_bbox[0] < bbox[2]
            and erase_bbox[3] > bbox[1]
            and erase_bbox[1] < bbox[3]
        )

        if not is_overlap:
            if value is None:
                img[y : y + h, x : x + w, :] = np.random.randint(
                    0, 256, (h, w, imgC), dtype=np.uint8
                )
            else:
                img[y : y + h, x : x + w, :] = np.full(
                    (h, w, imgC), value, dtype=np.uint8
                )
            break

    return img


def augment_bbox(
    bbox: List[int], image_size: Tuple[int, int], shift_ratio: float = 0.1
):
    """
    Shift the bounding box by a random ratio of its width and height.
    Args:
        bbox (List[int]): The bounding box coordinates [x1, y1, x2, y2].
        shift_ratio (float): The maximum ratio to shift the bounding box.
    Returns:
        List[int]: The shifted bounding box coordinates [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size

    width, height = x2 - x1, y2 - y1
    shifts = lambda length: int(length * random.uniform(-shift_ratio, shift_ratio))
    augmented_bbox = [
        max(0, x1 + shifts(width)),
        max(0, y1 + shifts(height)),
        min(img_w, x2 + shifts(width)),
        min(img_h, y2 + shifts(height)),
    ]
    return augmented_bbox


def rotate_image_and_bbox(
    img: np.ndarray, bbox: List[int], max_angle: float
) -> Tuple[np.ndarray, List[int]]:
    """Rotate image and bounding box, and get the bounding box that contours the rotated bounding box."""
    imgH, imgW = img.shape[:2]

    angle = random.uniform(-max_angle, max_angle)

    # Center of the image
    center = (imgH // 2, imgW // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    rotated_img = cv2.warpAffine(img, rotation_matrix, (imgW, imgH))

    # Calculate the original bounding box coordinates
    x1, y1, x2, y2 = bbox
    bbox_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # Rotate the bounding box coordinates
    ones = np.ones(shape=(len(bbox_corners), 1))
    bbox_corners = np.hstack([bbox_corners, ones])
    rotated_bbox_corners = rotation_matrix.dot(bbox_corners.T).T

    # Get the new bounding box coordinates
    x_coords = rotated_bbox_corners[:, 0]
    y_coords = rotated_bbox_corners[:, 1]
    x1_new, y1_new = np.min(x_coords), np.min(y_coords)
    x2_new, y2_new = np.max(x_coords), np.max(y_coords)

    # Ensure the coordinates are within the image boundaries
    x1_new = max(0, int(x1_new))
    y1_new = max(0, int(y1_new))
    x2_new = min(imgH, int(x2_new))
    y2_new = min(imgW, int(y2_new))

    # Return the rotated image and the new bounding box
    return rotated_img, [x1_new, y1_new, x2_new, y2_new]
