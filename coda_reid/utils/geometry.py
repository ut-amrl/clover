import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_3d_bbox(bbox: np.ndarray, extrinsic: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Represent 3D bounding box with the given extrinsic matrix.
    example: bbox_3d in LiDAR frame -> bbox_3d in map frame,
             transformation should be LiDAR to map transformation matrix

    Args:
        bbox: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]
        extrinsic: (4, 4) extrinsic matrix
        degrees: bool, if True, r, p, y are in degrees

    Returns:
        transformed_bbox_3d: (9, ) array of [cX, cY, cZ, l, w, h, r, p, y]
    """
    assert bbox.shape == (9,), f"{bbox.shape} != (9, )"
    assert extrinsic.shape == (4, 4), f"{extrinsic.shape} != (4, 4)"
    cX, cY, cZ, l, w, h, r, p, y = bbox

    # Tlb
    bbox_frame = np.eye(4)
    bbox_frame[:3, 3] = np.array([cX, cY, cZ])
    bbox_frame[:3, :3] = R.from_euler("xyz", [r, p, y], degrees=degrees).as_matrix()

    # Twb = Twl @ Tlb
    transformed_bbox_frame = extrinsic @ bbox_frame

    transformed_bbox_3d = np.zeros(9)
    transformed_bbox_3d[:3] = transformed_bbox_frame[:3, 3]
    transformed_bbox_3d[3:6] = np.array([l, w, h])
    transformed_bbox_3d[6:9] = R.from_matrix(transformed_bbox_frame[:3, :3]).as_euler(
        "xyz", degrees=degrees
    )
    return transformed_bbox_3d
