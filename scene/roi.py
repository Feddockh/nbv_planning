import numpy as np
from scene.objects import DebugLines


class ROI:
    """Base class for Regions of Interest (ROI)."""
    
    def contains(self, points):
        raise NotImplementedError("Subclasses must implement contains()")

    def visualize(self, env=None, rgba=[1, 0, 0, 0.3]):
        raise NotImplementedError("Subclasses must implement visualize()")
    
    def to_bounds(self):
        """
        Convert ROI to bounds array format [x_min, y_min, z_min, x_max, y_max, z_max].
        
        Returns:
            np.ndarray: Bounds array or None if conversion not supported
        """
        raise NotImplementedError("Subclasses may optionally implement to_bounds()")

class RectangleROI(ROI):
    def __init__(self, center, half_extents):
        self.center = np.array(center, dtype=np.float64)
        self.half_extents = np.array(half_extents, dtype=np.float64)

    def contains(self, points):
        points = np.atleast_1d(np.asarray(points, dtype=np.float64))
        
        # Handle single point (shape (3,))
        if points.ndim == 1:
            if len(points) != 3:
                raise ValueError(f"Point must be 3D, got shape {points.shape}")
            relative = np.abs(points - self.center)
            return np.all(relative <= self.half_extents)
        
        # Handle multiple points (shape (N, 3))
        if points.ndim == 2:
            if points.shape[1] != 3:
                raise ValueError(f"Points must be Nx3, got shape {points.shape}")
            relative = np.abs(points - self.center)
            return np.all(relative <= self.half_extents, axis=1)
        
        raise ValueError(f"Points must be 1D or 2D, got shape {points.shape}")

    def visualize(self, lines_rgb=[0, 0, 1], line_width=2, env=None):
        """Draw the region of interest as a box"""
        roi_min, roi_max = self.center - self.half_extents, self.center + self.half_extents
        corners = [
            [roi_min[0], roi_min[1], roi_min[2]], [roi_max[0], roi_min[1], roi_min[2]],
            [roi_max[0], roi_max[1], roi_min[2]], [roi_min[0], roi_max[1], roi_min[2]],
            [roi_min[0], roi_min[1], roi_max[2]], [roi_max[0], roi_min[1], roi_max[2]],
            [roi_max[0], roi_max[1], roi_max[2]], [roi_min[0], roi_max[1], roi_max[2]],
        ]
        debug_ids = []
        # Bottom face
        for i in range(4):
            debug_id = DebugLines(corners[i], corners[(i+1)%4], lines_rgb=lines_rgb, line_width=line_width)
            debug_ids.extend(debug_id)
        # Top face
        for i in range(4):
            debug_id = DebugLines(corners[4+i], corners[4+(i+1)%4], lines_rgb=lines_rgb, line_width=line_width)
            debug_ids.extend(debug_id)
        # Vertical edges
        for i in range(4):
            debug_id = DebugLines(corners[i], corners[4+i], lines_rgb=lines_rgb, line_width=line_width)
            debug_ids.extend(debug_id)
        return debug_ids

    def to_bounds(self):
        """
        Convert RectangleROI to bounds array format [x_min, y_min, z_min, x_max, y_max, z_max].
        
        Returns:
            np.ndarray: Bounds array with shape (6,)
        """
        roi_min = self.center - self.half_extents
        roi_max = self.center + self.half_extents
        return np.concatenate([roi_min, roi_max])

class SphereROI(ROI):
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)

    def contains(self, points):
        points = np.atleast_1d(np.asarray(points, dtype=np.float64))
        
        # Handle single point (shape (3,))
        if points.ndim == 1:
            if len(points) != 3:
                raise ValueError(f"Point must be 3D, got shape {points.shape}")
            distance = np.linalg.norm(points - self.center)
            return distance <= self.radius
        
        # Handle multiple points (shape (N, 3))
        if points.ndim == 2:
            if points.shape[1] != 3:
                raise ValueError(f"Points must be Nx3, got shape {points.shape}")
            distances = np.linalg.norm(points - self.center, axis=1)
            return distances <= self.radius
        
        raise ValueError(f"Points must be 1D or 2D, got shape {points.shape}")
    
    def visualize(self, env=None, rgba=[1, 0, 0, 0.3]):
        if env is None:
            return
        
        # Create a sphere mesh for visualization
        sphere = env.add_sphere(radius=self.radius, position=self.center, rgba=rgba)
        return sphere
