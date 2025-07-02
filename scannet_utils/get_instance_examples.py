import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def extract_and_save_views(dataset, save_dir: str, max_instances: int = 100):
    os.makedirs(save_dir, exist_ok=True)

    for _ in tqdm(range(len(dataset))[:max_instances]):
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]

        # --- Collect aspect ratios for all views ---
        aspect_ratios = []
        for v in range(dataset.n_views):
            img, bbox = data["images"][v]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w == 0 or h == 0:
                aspect_ratios.append((v, float("inf")))
            else:
                aspect_ratio = max(w / h, h / w)
                aspect_ratios.append((v, abs(aspect_ratio - 1)))

        # --- Select 6 most square views ---
        top_views = sorted(aspect_ratios, key=lambda x: x[1])[:6]
        selected_view_ids = [v for v, _ in top_views]

        fig, ax = plt.subplots(1, 6, figsize=(12, 2))

        for i, v in enumerate(selected_view_ids):
            img, bbox = data["images"][v]

            orig_x1, orig_y1, orig_x2, orig_y2 = bbox

            # --- Apply padding to bbox for cropping ---
            padding = 50
            padded_x1 = max(0, orig_x1 - padding)
            padded_y1 = max(0, orig_y1 - padding)
            padded_x2 = min(img.shape[1], orig_x2 + padding)
            padded_y2 = min(img.shape[0], orig_y2 + padding)

            # --- Square crop centered on the padded bbox center ---
            cx = (padded_x1 + padded_x2) // 2
            cy = (padded_y1 + padded_y2) // 2
            size = max(padded_x2 - padded_x1, padded_y2 - padded_y1)
            half = size // 2

            sx1 = max(0, cx - half)
            sy1 = max(0, cy - half)
            sx2 = min(img.shape[1], cx + half)
            sy2 = min(img.shape[0], cy + half)

            # --- Crop the image ---
            img_cropped = img[sy1:sy2, sx1:sx2]

            # --- Adjust original bbox coordinates relative to the crop ---
            draw_x = orig_x1 - sx1
            draw_y = orig_y1 - sy1
            draw_w = orig_x2 - orig_x1
            draw_h = orig_y2 - orig_y1

            # --- Plot ---
            ax[i].imshow(img_cropped)
            rect = patches.Rectangle(
                (draw_x, draw_y), draw_w, draw_h, linewidth=3, edgecolor="red", facecolor="none"
            )
            ax[i].add_patch(rect)
            ax[i].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"instance_{idx}.png"))
        plt.close(fig)


if __name__ == "__main__":
    from datasets.scannet import ScanNetDataset

    def transform(img, bbox, **kwargs):
        return img, bbox

    dataset = ScanNetDataset(
        root_dir="data/ScanNet",
        mode="train",
        transform=transform,
        method="SupCon",
        n_views=8,
        frame_gap=50,
        train_index_file="data/ScanNet/index/scannet_train.txt",
    )

    extract_and_save_views(dataset, save_dir="figure/scannet/examples")
