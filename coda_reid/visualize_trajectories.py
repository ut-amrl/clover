import argparse
import json
import os
import pathlib

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    for seq in tqdm(args.sequences):
        pose_file = args.pose_dir / f"{seq}.txt"
        pose_np = np.loadtxt(pose_file).reshape(-1, 8)

        x, y = pose_np[:, 1], pose_np[:, 2]
        ax.plot(x, y, label=f"Seq {seq}")

    ax.set_title("2D Trajectories (x-y)")
    ax.set_xlabel("x [meters]")
    ax.set_ylabel("y [meters]")
    ax.axis("equal")  # Preserve aspect ratio
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure/coda_reid/trajectories.png")


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

    main(args)
