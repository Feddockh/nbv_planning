import numpy as np
import heapq
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from viewpoints.viewpoint import Viewpoint


def compute_fibonacci_sphere(position: np.ndarray,
                             orientation: np.ndarray,
                             radius: float,
                             num_samples: int = 20,
                             hemisphere: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate points on a Fibonacci sphere (or hemisphere) around a center point.
    
    Args:
        position: 3D position point
        orientation: Quaternion [x, y, z, w] defining the sphere's orientation
        radius: Sphere radius
        num_samples: Number of points to generate
        hemisphere: If True, only generate upper hemisphere (useful for cameras)
        
    Returns:
        Tuple of (List of 3D positions on the sphere, List of orientations as quaternions)
    """
    points = []
    orientations = []
    
    # Convert quaternion to rotation matrix to extract basis vectors
    base_rot = R.from_quat(orientation)
    R_base = base_rot.as_matrix()
    
    # Extract basis vectors from rotation matrix
    # X-axis (right), Y-axis (up), Z-axis (forward)
    sphere_x = R_base[:, 0]
    sphere_y = R_base[:, 1]
    sphere_z = R_base[:, 2]
    
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians
    
    for i in range(num_samples):
        if hemisphere:
            # Map to hemisphere (z >= 0)
            y = 1 - (i / float(num_samples - 1))  # y goes from 1 to 0
            if y < 0:
                continue
        else:
            y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # Transform from local sphere coordinates to world coordinates
        # Local coordinates: x, y, z on unit sphere
        local_point = np.array([x, y, z])
        
        # Scale by radius and transform to world coordinates using basis vectors
        point = position + radius * (local_point[0] * sphere_x + local_point[1] * sphere_y + local_point[2] * sphere_z)
        points.append(point)
        
        # Orientation: camera points inward toward the center (position)
        # Direction from viewpoint to center
        direction = position - point
        direction = direction / np.linalg.norm(direction)
        
        # The sphere's forward direction at this point
        forward = local_point[0] * sphere_x + local_point[1] * sphere_y + local_point[2] * sphere_z
        forward = forward / np.linalg.norm(forward)
        
        # Check if forward and direction are not parallel
        if np.abs(np.dot(forward, direction)) < 0.9999:
            # Rotation axis is perpendicular to both
            rot_axis = np.cross(forward, direction)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            rot_angle = np.arccos(np.clip(np.dot(forward, direction), -1, 1))
            
            rot = R.from_rotvec(rot_angle * rot_axis)
            quat = rot.as_quat()
        else:
            # Direction is parallel to forward
            if np.dot(forward, direction) > 0:
                quat = np.array([0, 0, 0, 1])  # Identity
            else:
                quat = np.array([1, 0, 0, 0])  # 180 degree rotation around x
        
        orientations.append(quat)

    return points, orientations

def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimal-rotation quaternion [x,y,z,w] mapping unit vector a -> b in world.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])            # identity
    if dot < -1.0 + 1e-8:
        # 180°: pick any axis ⟂ a
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis /= (np.linalg.norm(axis) + 1e-12)
        return np.array([axis[0], axis[1], axis[2], 0.0])  # 180° about axis
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)
    invs = 1.0 / s
    return np.array([axis[0]*invs, axis[1]*invs, axis[2]*invs, 0.5*s], dtype=float)

def compute_spherical_cap(
    position: np.ndarray,
    orientation: np.ndarray,          # quaternion [x, y, z, w] (world)
    radius: float,
    angular_resolution: float,         # radians
    max_theta: float                   # radians
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate (roughly) evenly spaced points and orientations over a spherical cap
    centered on the camera's forward axis (+Z of the given orientation), with
    NO roll (minimal twist). Each orientation points from the point on the cap
    back toward the cap center (i.e., inward).

    Args:
        position: (3,) cap center in world coordinates.
        orientation: (4,) base orientation quaternion [x,y,z,w] (world).
        radius: sphere radius.
        angular_resolution: desired angular spacing on the sphere (radians).
        max_theta: half-angle of the cap (radians).

    Returns:
        points: List[(3,)] 3D positions on the spherical cap (world).
        orientations: List[(4,)] quaternions [x,y,z,w] (world) with minimal roll.
    """
    if angular_resolution <= 0 or max_theta < 0:
        raise ValueError("angular_resolution must be > 0 and max_theta >= 0")

    points: List[np.ndarray] = []
    quats:  List[np.ndarray] = []

    base_rot = R.from_quat(orientation)      # world←camera
    R_base = base_rot.as_matrix()
    cap_x = R_base[:, 0]                     # local +X (down, by your convention)
    cap_y = R_base[:, 1]                     # local +Y (left)
    cap_z = R_base[:, 2]                     # local +Z (forward)

    # Build polar rings (include center)
    thetas = [0.0]
    t = angular_resolution
    while t <= max_theta + 1e-12:
        thetas.append(t)
        t += angular_resolution

    for ring_idx, theta in enumerate(thetas):
        if theta == 0.0:
            # Center point (unique)
            local = np.array([0.0, 0.0, radius])
            p = position + local[0]*cap_x + local[1]*cap_y + local[2]*cap_z
            points.append(p)

            # Looking inward toward the center => forward points from p -> position
            # Minimal rotation from base forward (cap_z) to direction (position - p)
            # direction = position - p
            direction = p - position # Outward
            direction /= (np.linalg.norm(direction) + 1e-12)
            quat_delta = _quat_from_two_vectors(cap_z, direction)
            q_world = (R.from_quat(quat_delta) * base_rot).as_quat()
            quats.append(q_world)
            continue

        # Approximately equal arc-length spacing:
        # samples per ring ≈ ring circumference / angular_resolution
        ring_circum = 2.0 * np.pi * np.sin(theta)
        m = max(1, int(np.ceil(ring_circum / (angular_resolution + 1e-12))))

        # Stagger alternate rings to reduce alignment artifacts
        phi_offset = (np.pi / m) if (ring_idx % 2 == 1) else 0.0

        for k in range(m):
            phi = (2.0 * np.pi * k) / m + phi_offset

            # Local point on the sphere (camera frame: z forward, y left, x down)
            local_x = radius * np.sin(theta) * np.cos(phi)
            local_y = radius * np.sin(theta) * np.sin(phi)
            local_z = radius * np.cos(theta)

            # To world
            p = position + local_x * cap_x + local_y * cap_y + local_z * cap_z
            points.append(p)

            # Orientation: from this point, look back to the center (inward)
            # direction = position - p
            direction = p - position # Outward
            direction /= (np.linalg.norm(direction) + 1e-12)

            # Minimal rotation from base forward to the desired direction
            quat_delta = _quat_from_two_vectors(cap_z, direction)
            q_world = (R.from_quat(quat_delta) * base_rot).as_quat()  # compose with base
            quats.append(q_world)

    return points, quats

def generate_planar_spherical_candidates(position: np.ndarray,
                               orientation: np.ndarray,
                               half_extent: float,
                               spatial_resolution: float,
                               max_theta: float,
                               angular_resolution: float) -> List[Viewpoint]:
    """
    Generate viewpoints on a plane with spherical cap orientations.
    
    Creates a grid of positions on a plane and for each position,
    generates multiple orientations using a spherical cap distribution.
    
    Args:
        position: 3D center point of the plane
        orientation: Quaternion [x, y, z, w] defining the plane's orientation
        half_extent: Half-size of the plane grid (generates [-extent, extent] on plane)
        spatial_resolution: Distance between viewpoint positions on the plane
        max_theta: Maximum angle (degrees) to vary roll and pitch from base orientation
        angular_resolution: Angular step size (degrees) for roll/pitch variations
        
    Returns:
        List of Viewpoint objects with all position/orientation combinations
    """
    
    # Convert quaternion to rotation matrix to extract basis vectors
    base_rot = R.from_quat(orientation)
    R_base = base_rot.as_matrix()
    
    # Extract basis vectors from rotation matrix
    # X-axis (right), Y-axis (up), Z-axis (forward)
    plane_x = R_base[:, 0]
    plane_y = R_base[:, 1]
    normal = R_base[:, 2]
    
    # Generate grid of positions on the plane
    grid_positions = []
    x_range = np.arange(-half_extent, half_extent + spatial_resolution, spatial_resolution)
    y_range = np.arange(-half_extent, half_extent + spatial_resolution, spatial_resolution)
    
    for x in x_range:
        for y in y_range:
            # Position on plane
            pos = position + x * plane_x + y * plane_y
            grid_positions.append(pos)
    
    # Generate orientation variations (roll and pitch offsets)
    orientations = []
    
    # Base orientation from input quaternion
    base_quat = orientation

    # Convert max_theta and angular_resolution from degrees to radians
    max_theta = np.deg2rad(max_theta)
    angular_resolution = np.deg2rad(angular_resolution)

    # Check if no orientation variations, just use base orientation
    if max_theta == 0 or angular_resolution == 0:
        orientations.append(base_quat)

    # Otherwise, get offsets from the spherical cap
    else:
        orientations = compute_spherical_cap(
            position=np.array([0.0, 0.0, 0.0]),  # Dummy position
            orientation=base_quat,
            radius=1.0,  # Unit sphere for direction offsets
            angular_resolution=angular_resolution,
            max_theta=max_theta
        )[1]  # Only need orientations

    # Create Viewpoint objects for each position with each orientation
    viewpoints = []
    for position in grid_positions:
        for orientation in orientations:
            viewpoint = Viewpoint(
                position=position,
                orientation=orientation,
                target=position + normal,  # Look-at point is along normal direction
                information_gain=0.0,  # Will be computed later
                cost=0.0,
                utility=0.0,
                joint_angles=None
            )
            viewpoints.append(viewpoint)
    
    return viewpoints

