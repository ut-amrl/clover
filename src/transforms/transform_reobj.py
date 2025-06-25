from typing import Literal

import numpy as np
from PIL import Image
from torchvision import transforms

from .common import crop_image, segment_to_mask


class TransformReOBJ:
    def __init__(
        self,
        img_size: int,
        margin: int = 10,
        square: bool = False,
    ):
        self.img_size = (img_size, img_size)
        self.margin = margin
        self.square = square

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

    def __call__(
        self, img, bbox, segmentation, mode: Literal["train", "eval"], **kwargs
    ):
        # create a mask from the segmentation
        mask = segment_to_mask(segmentation, img.shape[:2])
        mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)

        # split foreground and background
        img_fg = (img * mask).astype(img.dtype)
        img_bg = (img * (1 - mask)).astype(img.dtype)

        # crop the images
        img_fg = crop_image(img_fg, bbox, self.margin, self.square)
        img_bg = crop_image(img_bg, bbox, self.margin, self.square)

        # apply transforms
        if mode == "train":
            img_fg = self.train_transform(img_fg)
            img_bg = self.train_transform(img_bg)
        elif mode == "eval":
            img_fg = self.eval_transform(img_fg)
            img_bg = self.eval_transform(img_bg)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return img_fg, img_bg
