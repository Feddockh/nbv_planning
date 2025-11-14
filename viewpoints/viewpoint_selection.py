from typing import Optional
import numpy as np

from viewpoints.viewpoint import Viewpoint
from scene.roi import ROI
from scene.scene_representation import SceneRepresentation


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix (world <- camera)."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)

def _compute_ray_strides(width: int, height: int, num_rays: int) -> tuple[int, int]:
    """
    Compute horizontal and vertical stride (pixel step) for roughly num_rays total.
    Returns (stride_u, stride_v)
    """
    if num_rays <= 0:  # use all pixels
        return 1, 1
    total = width * height
    step = int(np.sqrt(total / num_rays))
    return max(1, step), max(1, step)

def compute_information_gain(
    viewpoint: Viewpoint,
    scene_representation: SceneRepresentation,
    fov: float = 60.0,        # horizontal FOV (deg)
    width: int = 640,
    height: int = 480,
    max_range: float = 3.0,
    resolution: float = 0.1,  # ray marching step (m)
    num_rays: int = -1,
    roi: Optional[ROI] = None
) -> float:
    """
    Compute the information gain for a given viewpoint based on the octomap.
    
    Information gain is the average number of unknown voxels discovered per ray.
    For each ray:
    - Count unknown voxels along the ray (within ROI if specified)
    - Stop at max_range or when hitting an occupied voxel
    
    IG = (1/|R|) * sum_r (# unknown voxels along ray r)
    
    Args:
        viewpoint (Viewpoint): The candidate viewpoint.
        scene_representation (SceneRepresentation): The scene representation (OctoMap).
        fov (float): Camera horizontal field of view in degrees.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        max_range (float): Maximum sensor range in meters.
        resolution (float): Sampling resolution along each ray in meters.
        num_rays (int): Number of rays to cast (if -1, use all pixels).
        roi (ROI): Optional region of interest to constrain counting.

    Returns:
        float: The average number of unknown voxels per ray.
    """

    if width <= 0 or height <= 0 or max_range <= 0 or resolution <= 0:
        return 0.0

    # Determine pixel stride based on desired ray count
    stride_u, stride_v = _compute_ray_strides(width, height, num_rays)

    # Camera pose and intrinsics
    cam_pos = np.asarray(viewpoint.position, dtype=np.float64)
    R_wc = _quat_to_rotmat_xyzw(np.asarray(viewpoint.orientation, dtype=np.float64))  # world<-cam

    hfov = np.deg2rad(float(fov))
    focal = (width / 2.0) / np.tan(hfov / 2.0)
    cx = (width  - 1) * 0.5
    cy = (height - 1) * 0.5

    # Octomap access
    tree = scene_representation.octree
    max_depth = int(tree.getTreeDepth())

    # Marching setup
    num_steps = int(np.floor(max_range / resolution))
    if num_steps <= 0:
        return 0.0

    total_unknown = 0
    num_rays_cast = 0

    for v in range(0, height, stride_v):
        for u in range(0, width, stride_u):
            # Ray in camera frame (z forward, x right, y down in image coords)
            x = (u - cx) / focal
            y = (v - cy) / focal
            ray_cam = np.array([x, y, 1.0], dtype=np.float64)
            ray_cam /= np.linalg.norm(ray_cam)

            # Transform to world frame
            ray_world = R_wc @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            unknown_count = 0

            # March along the ray
            for i in range(1, num_steps + 1):
                t = i * resolution
                point = cam_pos + t * ray_world

                # ROI gating: skip points outside ROI
                if roi is not None and not roi.contains(point):
                    continue

                # Check voxel state
                node = tree.search(point, depth=max_depth)
                if node is None:
                    # Unknown voxel - increment counter
                    unknown_count += 1
                else:
                    # Known voxel: if occupied, stop the ray
                    if tree.isNodeOccupied(node):
                        break
                    # If free, continue marching

            total_unknown += unknown_count
            num_rays_cast += 1

    # Return average unknown voxels per ray
    return float(total_unknown / num_rays_cast) if num_rays_cast > 0 else 0.0
