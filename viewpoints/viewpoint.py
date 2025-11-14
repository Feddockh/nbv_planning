import numpy as np
from dataclasses import dataclass

from bodies.robot import Robot
from vision import RobotCamera
from utils import get_rotation_matrix, multiply_transforms
from scene.objects import DebugCoordinateFrame, DebugPoints
from utils import get_euler, get_quaternion


@dataclass
class Viewpoint:
    """Represents a candidate viewpoint for NBV planning"""
    position: np.ndarray             # 3D camera position
    orientation: np.ndarray          # Camera orientation (quaternion)
    target: np.ndarray               # 3D point the camera looks at
    information_gain: float = 0.0    # IG score
    cost: float = 0.0                # Movement cost
    utility: float = 0.0             # Final utility (IG - α*cost)
    joint_angles: np.ndarray = None  # Robot joint angles for this viewpoint
    
    def __lt__(self, other):
        """For max-heap ordering (higher utility first)"""
        return self.utility > other.utility

def compute_viewpoint_joint_angles(robot: Robot, viewpoint: Viewpoint, robot_camera: RobotCamera = None) -> np.ndarray:
    """
    Compute the robot joint angles to achieve the given viewpoint.
    
    Args:
        robot: Robot instance
        viewpoint: Viewpoint object with position and orientation
        robot_camera: Optional RobotCamera object with end effector to camera offset
        
    Returns:
        joint_angles: np.ndarray of joint angles to reach the viewpoint
    """
    # Default end effector to camera offset (identity)
    camera_offset_pos = np.array([0, 0, 0], dtype=np.float64)
    camera_offset_orient = np.array([0, 0, 0, 1], dtype=np.float64)
    
    if robot_camera is not None:
        camera_offset_pos = np.array(robot_camera.camera_offset_pos, dtype=np.float64)
        camera_offset_orient = np.array(robot_camera.camera_offset_orient, dtype=np.float64)

    # Apply the offset: ee_pose = camera_pose * inverse(camera_offset)
    # This computes the end effector pose needed so that with the offset applied,
    # the camera ends up at the desired viewpoint
    
    # Inverse the camera offset (position and orientation)
    camera_offset_orient_inv = np.array([
        -camera_offset_orient[0],
        -camera_offset_orient[1],
        -camera_offset_orient[2],
        camera_offset_orient[3]
    ])
    
    # Transform the camera offset by its inverse orientation to get inverse offset position
    offset_pos_rotated = get_rotation_matrix(camera_offset_orient_inv) @ camera_offset_pos
    camera_offset_pos_inv = -offset_pos_rotated
    
    # Compute end effector pose: ee_pose = multiply_transforms(camera_pose, inverse_offset)
    ee_pos, ee_orient = multiply_transforms(
        viewpoint.position,
        viewpoint.orientation,
        camera_offset_pos_inv,
        camera_offset_orient_inv
    )
    
    # Solve inverse kinematics to get joint angles
    joint_angles = robot.ik(
        target_joint=robot.end_effector,
        target_pos=ee_pos,
        target_orient=ee_orient,
        use_current_joint_angles=True
    )
    return joint_angles

def visualize_viewpoint(viewpoint: Viewpoint, local_env=None, coordinate_frame: bool = True):
    debug_ids = []

    if coordinate_frame:
        # Visualize a coordinate frame at the viewpoint position and orientation
        debug_ids = DebugCoordinateFrame(position=viewpoint.position,
                                         orientation=get_euler(viewpoint.orientation),
                                         axis_length=0.05,
                                         axis_radius=0.05,
                                         local_env=local_env)
    else:
        # Visualize a point at the viewpoint position
        debug_ids = DebugPoints(points=np.array([viewpoint.position]),
                                colors=np.array([[0, 0, 1]]),
                                point_size=5.0,
                                local_env=local_env)

    return debug_ids