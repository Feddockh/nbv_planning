import numpy as np
import heapq
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from viewpoints.viewpoint import Viewpoint


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
    max_theta: float,                  # radians
    look_at_center: bool = False,      # orientation behavior
    use_positive_z: bool = True        # cap direction
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate (roughly) evenly spaced points and orientations over a spherical cap
    with NO roll (minimal twist).
    
    The spherical cap is computed in the reference frame defined by the orientation
    quaternion. The cap extends along the Z-axis of this frame (either +Z or -Z).

    Args:
        position: (3,) cap center in world coordinates.
        orientation: (4,) base orientation quaternion [x,y,z,w] (world) that defines
                     the reference frame for the spherical cap. The cap is computed
                     along the Z-axis of this frame.
        radius: sphere radius.
        angular_resolution: desired angular spacing on the sphere (radians).
        max_theta: half-angle of the cap (radians).
        look_at_center: If True, orientations point inward toward the cap center.
                       If False, orientations point outward away from the center.
        use_positive_z: If True, compute cap along +Z axis of the orientation frame.
                       If False, compute cap along -Z axis of the orientation frame.

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
    cap_x = R_base[:, 0]                     # local +X (right/down, by your convention)
    cap_y = R_base[:, 1]                     # local +Y (up/left)
    cap_z = R_base[:, 2]                     # local +Z (forward)
    
    # Flip Z direction if using negative Z axis
    z_sign = 1.0 if use_positive_z else -1.0

    # Build polar rings (include center)
    thetas = [0.0]
    t = angular_resolution
    while t <= max_theta + 1e-12:
        thetas.append(t)
        t += angular_resolution

    for ring_idx, theta in enumerate(thetas):
        if theta == 0.0:
            # Center point (unique)
            local = np.array([0.0, 0.0, z_sign * radius])
            p = position + local[0]*cap_x + local[1]*cap_y + local[2]*cap_z
            points.append(p)

            # Compute orientation based on look_at_center flag
            if look_at_center:
                # Looking inward toward the center
                direction = position - p  # Inward
            else:
                # Looking outward away from center
                direction = p - position  # Outward
            
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
            local_z = z_sign * radius * np.cos(theta)

            # To world
            p = position + local_x * cap_x + local_y * cap_y + local_z * cap_z
            points.append(p)

            # Compute orientation based on look_at_center flag
            if look_at_center:
                # Looking inward toward the center
                direction = position - p  # Inward
            else:
                # Looking outward away from center
                direction = p - position  # Outward
            
            direction /= (np.linalg.norm(direction) + 1e-12)

            # Minimal rotation from base forward to the desired direction
            quat_delta = _quat_from_two_vectors(cap_z, direction)
            q_world = (R.from_quat(quat_delta) * base_rot).as_quat()  # compose with base
            quats.append(q_world)

    return points, quats

def generate_planar_spherical_cap_candidates(position: np.ndarray,
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

def sample_views_from_hemisphere(center: np.ndarray,
                                 base_orientation: np.ndarray,
                                 min_radius: float,
                                 max_radius: Optional[float] = None,
                                 num_samples: int = 20,
                                 use_positive_z: bool = False,
                                 z_bias_sigma: float = 0.3,
                                 min_distance: float = 0.1,
                                 max_attempts: int = 1000) -> List[Viewpoint]:
    """
    Randomly sample viewpoints from a hemispherical shell or surface around a center point.
    
    Uses Gaussian distribution biased toward the Z-axis with minimum distance constraints
    between samples. If max_radius equals min_radius or is not provided, samples on the 
    hemisphere surface. The hemisphere is oriented along either the +Z or -Z axis of the 
    base_orientation frame. Each viewpoint's orientation faces toward the center.
    
    Args:
        center: 3D center point of the hemisphere
        base_orientation: Quaternion [x, y, z, w] defining the hemisphere's reference frame.
                         The hemisphere extends along the Z-axis of this frame.
        min_radius: Minimum radius of the hemispherical shell (or the surface radius)
        max_radius: Maximum radius of the hemispherical shell. If None or equal to min_radius,
                   samples on the hemisphere surface at min_radius.
        num_samples: Number of viewpoints to randomly sample
        use_positive_z: If True, hemisphere extends along +Z axis of base_orientation.
                       If False, hemisphere extends along -Z axis.
        z_bias_sigma: Standard deviation for Gaussian sampling of theta angle. 
                     Smaller values (e.g., 0.2) concentrate samples near Z-axis,
                     larger values (e.g., 0.5) spread them more uniformly. Default 0.3.
                     The distribution is truncated to [0, pi/2].
        min_distance: Minimum distance between sampled viewpoint positions to avoid clustering.
                     Default 0.1 meters.
        max_attempts: Maximum number of attempts to sample viewpoints before giving up.
    
    Returns:
        List of Viewpoint objects with sampled positions and orientations facing the center
    """
    if min_radius < 0:
        raise ValueError("min_radius must be non-negative")
    if max_radius is not None and max_radius < min_radius:
        raise ValueError("max_radius must be >= min_radius")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if z_bias_sigma <= 0:
        raise ValueError("z_bias_sigma must be positive")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    
    # Determine if sampling on surface or within shell
    sample_surface = (max_radius is None or np.isclose(max_radius, min_radius))
    
    # Extract basis vectors from base orientation
    base_rot = R.from_quat(base_orientation)
    R_base = base_rot.as_matrix()
    cap_x = R_base[:, 0]  # local +X
    cap_y = R_base[:, 1]  # local +Y
    cap_z = R_base[:, 2]  # local +Z (hemisphere axis)
    
    # Z-direction sign
    z_sign = 1.0 if use_positive_z else -1.0
    
    viewpoints = []
    positions = []  # Track positions for minimum distance check
    
    attempts = 0
    while len(viewpoints) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample radius based on mode
        if sample_surface:
            # Fixed radius on the hemisphere surface
            r = min_radius
        else:
            # Uniformly sample radius in the shell volume
            # Use r^3 for volume uniform sampling
            u = np.random.uniform(0, 1)
            r_cubed = min_radius**3 + u * (max_radius**3 - min_radius**3)
            r = r_cubed ** (1.0/3.0)
        
        # Gaussian sampling biased toward Z-axis (theta near 0)
        # Sample from truncated Gaussian in [0, pi/2] with mean at 0
        theta = np.abs(np.random.normal(0, z_bias_sigma))
        # Clamp to hemisphere range [0, pi/2]
        theta = min(theta, np.pi / 2.0)
        
        # Uniform sampling for azimuthal angle
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Convert spherical to Cartesian in local frame
        local_x = r * np.sin(theta) * np.cos(phi)
        local_y = r * np.sin(theta) * np.sin(phi)
        local_z = z_sign * r * np.cos(theta)
        
        # Transform to world coordinates
        position = center + local_x * cap_x + local_y * cap_y + local_z * cap_z
        
        # Check minimum distance constraint
        if min_distance > 0 and len(positions) > 0:
            distances = np.linalg.norm(np.array(positions) - position, axis=1)
            if np.min(distances) < min_distance:
                continue  # Too close to existing point, reject and try again
        
        # Accept this point
        positions.append(position)
        
        # Compute orientation facing toward the center
        direction = center - position  # Inward direction
        direction /= (np.linalg.norm(direction) + 1e-12)
        
        # Compute quaternion that rotates cap_z to point toward center
        quat_delta = _quat_from_two_vectors(cap_z, direction)
        orientation = (R.from_quat(quat_delta) * base_rot).as_quat()
        
        # Create viewpoint
        viewpoint = Viewpoint(
            position=position,
            orientation=orientation,
            target=center,  # Look at the center
            information_gain=0.0,
            cost=0.0,
            utility=0.0,
            joint_angles=None
        )
        viewpoints.append(viewpoint)
    
    if len(viewpoints) < num_samples:
        print(f"Warning: Only generated {len(viewpoints)}/{num_samples} viewpoints. "
              f"Try reducing min_distance or increasing the sampling region.")
    
    return viewpoints