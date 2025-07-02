import argparse
import json
import os
import pathlib

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from utils.geometry import transform_3d_bbox

# fmt: off
CLASSES = {
    "Tree":               {"id": 0, "color": (0, 1.0, 0)},
    "Pole":               {"id": 1, "color": (0, 0, 1.0)},
    "Bollard":            {"id": 2, "color": (1.0, 0, 0)},
    "Informational_Sign": {"id": 3, "color": (1.0, 1.0, 0)},
    "Traffic_Sign":       {"id": 4, "color": (1.0, 0, 1.0)},
    "Trash_Can":          {"id": 5, "color": (0, 1.0, 1.0)},
    "Fire_Hydrant":       {"id": 6, "color": (0.5, 0.5, 0.5)},
    "Emergency_Phone":    {"id": 7, "color": (0.5, 0.5, 0.5)},
    "Railing":            {"id": 8, "color": (0.5, 0.5, 0.5)},
    "Fence":              {"id": 9, "color": (0.5, 0.5, 0.5)},
    "Bike_Rack":          {"id": 10, "color": (0.5, 0.5, 0.5)},
    "Floor_Sign":         {"id": 11, "color": (0.5, 0.5, 0.5)},
    "Traffic_Arm":        {"id": 12, "color": (0.5, 0.5, 0.5)},
    "Traffic_Light":      {"id": 13, "color": (0.5, 0.5, 0.5)},
    "Table":              {"id": 14, "color": (0.5, 0.5, 0.5)},
    "Chair":              {"id": 15, "color": (0.5, 0.5, 0.5)},
    "Door":               {"id": 16, "color": (0.5, 0.5, 0.5)},
    "Wall_Sign":          {"id": 17, "color": (0.5, 0.5, 0.5)},
    "Bench":              {"id": 18, "color": (0.5, 0.5, 0.5)},
    "Couch":              {"id": 19, "color": (0.5, 0.5, 0.5)},
    "Parking_Kiosk":      {"id": 20, "color": (0.5, 0.5, 0.5)},
    "Mailbox":            {"id": 21, "color": (0.5, 0.5, 0.5)},
    "Water_Fountain":     {"id": 22, "color": (0.5, 0.5, 0.5)},
    "ATM":                {"id": 23, "color": (0.5, 0.5, 0.5)},
    "Fire_Alarm":         {"id": 24, "color": (0.5, 0.5, 0.5)},
}
# fmt: on


def main(args):
    global_bboxes = {class_name: [] for class_name in CLASSES}

    # Load all 3D bounding boxes into the global frame
    for seq in tqdm(args.sequences, desc="Loading local 3D bounding boxes"):
        pose_file = args.pose_dir / f"{seq}.txt"
        pose_np = np.loadtxt(pose_file).reshape(-1, 8)

        for frame, pose in enumerate(pose_np):
            os1_3d_bbox_file = args.os1_3d_bbox_dir / f"{seq}" / f"3d_bbox_os1_{seq}_{frame}.json"
            if not os.path.exists(os1_3d_bbox_file):
                continue

            # bboxes per frame
            bboxes = load_3d_annotations(pose[1:], os1_3d_bbox_file)

            # accumulate
            for class_name in CLASSES:
                global_bboxes[class_name].extend(bboxes[class_name])

    # Get fitted 3D bounding boxes
    for class_name in CLASSES:
        # Cluster and fit 3D bounding boxes
        fitted_bboxes = cluster_fit_3d_bboxes(global_bboxes[class_name])

        # Global 3D bounding boxes
        global_bboxes_json = {
            "class": class_name,
            "instances": [
                {
                    "id": instances["id"],
                    **dict(
                        zip(
                            ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"],
                            instances["3d_bbox"],
                        )
                    ),
                }
                for instances in fitted_bboxes
            ],
        }

        logger.debug(
            f"Class: {class_name},  # instances: {len(global_bboxes_json['instances'])},  "
            f"# bboxes: {len(global_bboxes[class_name])}"
        )

        # Save the results
        out_file = args.global_3d_bbox_dir / f"{class_name}.json"
        with open(out_file, "w") as f:
            json.dump(global_bboxes_json, f, indent=4)


def load_3d_annotations(pose: np.ndarray, annotation_file: str):
    """
    Load 3D bounding box in the global frame

    Args:
        pose: (7, ) LiDAR pose in the global frame (x, y, z, qw, qx, qy, qz)
        annotation_file: The annotation file (.json file) (bbox in the LiDAR frame)

    Returns:
        bboxes: dict of 3D bounding boxes in the global frame
                {class_name: [bbox1, bbox2, ...]}
    """
    Twl = np.eye(4)
    Twl[:3, 3] = pose[:3]
    Twl[:3, :3] = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()

    bboxes_frame = {class_name: [] for class_name in CLASSES}

    for instance in json.load(open(annotation_file))["3dbbox"]:
        class_name = instance["classId"].replace(" ", "_")
        if class_name in CLASSES:
            bbox = np.array(
                [instance[attr] for attr in ["cX", "cY", "cZ", "l", "w", "h", "r", "p", "y"]]
            )
            global_bbox = transform_3d_bbox(bbox, Twl)
            bboxes_frame[class_name].append(global_bbox)

    return bboxes_frame


def cluster_fit_3d_bboxes(bboxes, threshold=1.0, min_samples=2):
    """
    Clustering and fitting 3D bounding boxes

    Args:
        bboxes: list of 3D bounding boxes (cX, cY, cZ, l, w, h, r, p, y)
        threshold: DBSCAN threshold
        min_samples: DBSCAN min_samples

    Returns:
        fitted_3d_bboxes: list of fitted 3D bounding boxes (cX, cY, cZ, l, w, h, r, p, y)
    """
    if len(bboxes) == 0:
        return []

    # Clustering
    centroids = np.array([bbox[:3] for bbox in bboxes])
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(centroids)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    # Fitting
    fitted_3d_bboxes = []
    for unique_label in tqdm(unique_labels):
        indices = np.where(labels == unique_label)[0]

        if len(indices) < min_samples:
            continue

        clustered_bboxes = np.array([bboxes[i] for i in indices])

        # Fit an 3d bbox
        fitted_bbox = fit_3d_bbox(clustered_bboxes)

        fitted_result = {"id": int(unique_label), "3d_bbox": fitted_bbox, "indices": indices}
        fitted_3d_bboxes.append(fitted_result)

    return fitted_3d_bboxes


def fit_3d_bbox(bboxes):
    """
    fit an 3d bbox to the given 3d bboxes by averaging
    """

    cX, cY, cZ, l, w, h = np.median(bboxes[:, :6], axis=0)
    # r, p, y = average_rpy(bboxes[:, 6:9])
    r, p, y = np.median(bboxes[:, 6:9], axis=0)

    return np.array([cX, cY, cZ, l, w, h, r, p, y])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/CODa")
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21],
        help="The sequences to process",
    )
    args = parser.parse_args()

    # input
    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pose_dir = args.dataset_dir / "poses" / "correct"
    args.os1_3d_bbox_dir = args.dataset_dir / "3d_bbox" / "os1"

    # output
    args.global_3d_bbox_dir = args.dataset_dir / "3d_bbox" / "global"
    os.makedirs(args.global_3d_bbox_dir, exist_ok=True)

    main(args)
