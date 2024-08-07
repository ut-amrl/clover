"""Preprocess ScanNet scenes for object re-identification"""

import argparse
import pathlib
from natsort import natsorted
import json
import jsbeautifier
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
import cv2

################################ Build SAM2 model ##############################

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "/home/dongmyeong/Projects/others/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)

################################################################################

EXCLUDE_LABELS = {
    "wall",
    "floor",
    "toilet",
    "ceiling",
    "doorframe",
    "door",
    "curtain",
    "kitchen counter",
    "sink",
}


def is_bbox_at_corner(bbox, image_size):
    x1, y1, x2, y2 = bbox
    height, width = image_size
    edges_touch = [False, False, False, False]  # top, left, bottom, right

    if y1 == 0:
        edges_touch[0] = True
    if x1 == 0:
        edges_touch[3] = True
    if y2 == height:
        edges_touch[2] = True
    if x2 == width:
        edges_touch[1] = True

    return sum(edges_touch) >= 2


def get_annotations(img_file, input_masks):
    bboxes = [None] * len(input_masks)
    segmentations = [None] * len(input_masks)

    input_indices = []
    input_bboxes = []
    for idx, mask in enumerate(input_masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            bbox = [x, y, x + w, y + h]
            if is_bbox_at_corner(bbox, mask.shape):
                continue

            input_indices.append(idx)
            input_bboxes.append(bbox)

    if not input_indices:
        return bboxes, segmentations

    # SAM2 prediction
    image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    SAM2_PREDICTOR.set_image(image)

    input_bboxes = np.array(input_bboxes)
    masks, scores, _ = SAM2_PREDICTOR.predict(
        point_coords=None,
        point_labels=None,
        box=input_bboxes,
        multimask_output=False,
    )

    # Get 2D instance annotation from SAM2 prediction
    for idx, mask, score in zip(input_indices, masks, scores):
        mask = mask.squeeze()
        score = score.squeeze()

        mask_overlap = (mask * input_masks[idx]).sum() / input_masks[idx].sum()
        print(
            f"Overlap: {mask_overlap:.2f}, Score: {score:.2f}, Mask area: {mask.sum()}"
        )

        # Filter out (occlusion, low score, small mask) cases
        if mask_overlap < 0.5 or score < 0.8 or mask.sum() < 10000:
            continue

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get the approximate polygon of contours
        approx_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:
                continue
            eps = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)
            approx_contours.append(approx)

        if not approx_contours:
            continue

        # Get bbox
        x, y, w, h = cv2.boundingRect(np.vstack(approx_contours))
        if w < 100 or h < 100:
            continue

        bboxes[idx] = [x, y, x + w, y + h]
        segmentations[idx] = [c.ravel().tolist() for c in approx_contours]

    return bboxes, segmentations


def main(args):
    img_files = natsorted(args.img_dir.glob("*.jpg"))[::100]
    instance_files = natsorted(args.instance_dir.glob("*.png"))[::100]
    aggregation = json.load(open(args.aggregation_file))
    instance_labels = {seg["id"]: seg["label"] for seg in aggregation["segGroups"]}

    data = {"sceneId": f"scannet.{args.scene_id}", "frames": []}

    # Process each frame
    for framd_id in tqdm(range(len(img_files)), desc=f"Processing {args.scene_id}"):
        img_file = img_files[framd_id]
        instance_file = instance_files[framd_id]

        instance = cv2.imread(str(instance_file), cv2.IMREAD_UNCHANGED)
        unique_instances = set(instance.flatten()) - {0}

        frame_data = {"frameId": framd_id, "instances": []}

        # split the instance masks
        ids, labels, masks = [], [], []
        for instance_id in unique_instances:
            # NOTE: subtract 1 to match the instance ID in the aggregation file
            label = instance_labels[instance_id - 1]

            if label in EXCLUDE_LABELS:
                continue

            ids.append(instance_id - 1)
            labels.append(label)
            mask = (instance == instance_id).astype(np.uint8)
            masks.append(mask)

        # Get 2D instance annotations
        bboxes, segmentations = get_annotations(img_file, masks)
        for bbox, segmentation, id, label in zip(bboxes, segmentations, ids, labels):
            if bbox is None:
                continue

            print(
                f"Instance ID: {id}, Label: {label}, bbox_w: {bbox[2] - bbox[0]}, bbox_h: {bbox[3] - bbox[1]}"
            )
            img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.imshow(img)
            plt.axis("off")
            plt.gca().add_patch(
                plt.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )
            # draw the polygon
            for seg in segmentation:
                plt.plot(seg[0::2], seg[1::2], color="green", linewidth=2)
            plt.show()

            frame_data["instances"].append(
                {
                    "instanceId": int(id),
                    "label": label,
                    "bbox": bbox,
                    "segmentation": segmentation,
                }
            )

        data["frames"].append(frame_data)

    # Save the data
    with open(args.output_file, "w") as f:
        print(f"Saving the data to {args.output_file}")
        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        f.write(jsbeautifier.beautify(json.dumps(data), opts))


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ScanNet scenes for object re-identification"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the ScanNet dataset root directory",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        required=True,
        help="ScanNet scene ID",
    )
    args = parser.parse_args()

    args.root_dir = pathlib.Path(args.root_dir)
    args.scene_dir = args.root_dir / "scans" / args.scene_id
    args.output_file = args.root_dir / "2d_instance" / f"{args.scene_id}.json"
    args.output_file.parent.mkdir(exist_ok=True, parents=True)

    args.img_dir = args.scene_dir / "sens"
    args.instance_dir = args.scene_dir / "instance-filt"
    args.aggregation_file = args.scene_dir / f"{args.scene_id}.aggregation.json"

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
