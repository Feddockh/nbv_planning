import numpy as np
import pybullet as p
import env
from scipy.spatial.transform import Rotation as R


def get_euler(quaternion, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return np.array(quaternion) if len(quaternion) == 3 else np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=local_env.id))

def get_quaternion(euler, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return R.from_matrix(euler).as_quat() if np.array(euler).ndim > 1 else (np.array(euler) if len(euler) == 4 else np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=local_env.id)))

def get_rotation_matrix(quaternion, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return np.array(p.getMatrixFromQuaternion(get_quaternion(quaternion), physicsClientId=local_env.id)).reshape((3,3))

def get_axis_angle(quaternion, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    q = get_quaternion(quaternion)
    sqrt = np.sqrt(1-q[-1]**2)
    return np.array([q[0]/sqrt, q[1]/sqrt, q[2]/sqrt]), 2*np.arccos(q[-1])

def get_difference_quaternion(q1, q2, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return p.getDifferenceQuaternion(get_quaternion(q1), get_quaternion(q2), physicsClientId=local_env.id)

def quaternion_product(q1, q2, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    # Return Hamilton product of 2 quaternions
    return p.multiplyTransforms([0, 0, 0], get_quaternion(q1), [0, 0, 0], get_quaternion(q2), physicsClientId=local_env.id)[1]

def multiply_transforms(p1, q1, p2, q2, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return p.multiplyTransforms(p1, get_quaternion(q1), p2, get_quaternion(q2), physicsClientId=local_env.id)

def rotate_point(point, quaternion, local_env=None):
    local_env = local_env if local_env is not None else env.envir
    return p.multiplyTransforms([0, 0, 0], get_quaternion(quaternion), point, [0, 0, 0, 1], physicsClientId=local_env.id)[0]

def quat_to_normal(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to the camera's forward/viewing direction (+Z axis in camera frame)."""
    r = R.from_quat(q)
    # In the camera frame, +Z is the forward axis
    forward_cam = np.array([0, 0, 1])
    normal_world = r.apply(forward_cam)
    return normal_world / np.linalg.norm(normal_world)
