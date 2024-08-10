from typing import List, Tuple

import numpy as np
import cv2


def crop_image(img: np.ndarray, bbox: List[int], margin: int, square: bool):
    """Crop image with margin"""
    imgH, imgW = img.shape[:2]

    # Add margin to the bounding box
    x1 = max(0, bbox[0] - margin)
    y1 = max(0, bbox[1] - margin)
    x2 = min(imgW, bbox[2] + margin)
    y2 = min(imgH, bbox[3] + margin)

    if not square:
        return img[y1:y2, x1:x2]

    # Make the bounding box square
    max_length = max(x2 - x1, y2 - y1)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # center the square bounding box
    x1 = center_x - max_length // 2
    y1 = center_y - max_length // 2
    x2 = x1 + max_length
    y2 = y1 + max_length

    # calcuate the padding
    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)
    pad_x2 = max(0, x2 - imgW)
    pad_y2 = max(0, y2 - imgH)

    # Adjust bbox to be within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(imgW, x2)
    y2 = min(imgH, y2)

    # Crop the image
    cropped_img = img[y1:y2, x1:x2]

    # Pad the image to make it square
    if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
        cropped_img = np.pad(
            cropped_img,
            ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return cropped_img


def segment_to_mask(segmentations: List[List[int]], image_size: Tuple[int, int]):
    """
    Convert segmentation to mask.

    Args:
        segmentations (List[List[int]]): A list of lists of points, where each list represents
                                         polygon vertices in the format [x1, y1, x2, y2, ...].
        image_size (Tuple[int, int]): The size of the output mask (height, width).

    Returns:
        np.ndarray: A binary mask of shape (height, width) where pixels inside polygons are 1.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for segmentation in segmentations:
        if len(segmentation) % 2 != 0:
            raise ValueError("Segmentation must have even number of elements")
        polygon = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [polygon], 255)

    return mask.astype(bool)
