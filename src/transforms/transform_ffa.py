import numpy as np
from torchvision import transforms
from PIL import Image

from .common import segment_to_mask, crop_image


class TransformFFA:
    def __init__(self, img_size: int, margin: int = 10, square: bool = True):
        self.img_size = (img_size, img_size)
        self.margin = margin
        self.square = square

        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.img_size, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.img_size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, img, bbox, segmentation, **kwargs):
        mask = segment_to_mask(segmentation, img.shape[:2])
        mask = mask[:, :, np.newaxis].astype(np.float32)

        img = crop_image(img, bbox, self.margin, self.square)
        mask = crop_image(mask, bbox, self.margin, self.square)

        return self.img_transform(img), self.mask_transform(mask)
