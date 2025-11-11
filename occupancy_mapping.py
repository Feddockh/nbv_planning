"""
Occupancy Mapping Utilities

Functions for frontier detection, ray tracing, and analysis of octomap data.
Useful for next-best-view planning and active perception.
"""

import sys
import os
import numpy as np
import pyoctomap
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../mengine'))
import mengine as m
from utils import DebugPoints


def find_roi_frontiers(octree: pyoctomap.OcTree, 
                       roi_bounds: np.ndarray,
                       resolution: float,
                       min_unknown_neighbors: int = 1) -> List[np.ndarray]:
    """
    Find ROI frontier voxels - free voxels near ROI that border unknown space.
    
    Algorithm:
    1. Check 6-neighborhoods of all ROI voxels for free nodes
    2. For free nodes, check their 6-neighborhoods for unknown neighbors
    3. Free nodes with at least one unknown neighbor are frontiers
    
    Args:
        octree: The OcTree to analyze
        roi_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
        resolution: Voxel resolution
        min_unknown_neighbors: Minimum unknown neighbors to qualify
        
    Returns:
        List of frontier positions (numpy arrays)
    """
    roi_min, roi_max = roi_bounds[:3], roi_bounds[3:]
    frontiers = []
    checked = set()
    
    # Sample points within ROI to check neighborhoods
    # Use resolution-sized grid
    x_range = np.arange(roi_min[0], roi_max[0] + resolution, resolution)
    y_range = np.arange(roi_min[1], roi_max[1] + resolution, resolution)
    z_range = np.arange(roi_min[2], roi_max[2] + resolution, resolution)
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                roi_point = np.array([x, y, z], dtype=np.float64)
                
                # Check 6-neighborhood for free nodes
                neighbors = [
                    roi_point + np.array([resolution, 0, 0]),
                    roi_point + np.array([-resolution, 0, 0]),
                    roi_point + np.array([0, resolution, 0]),
                    roi_point + np.array([0, -resolution, 0]),
                    roi_point + np.array([0, 0, resolution]),
                    roi_point + np.array([0, 0, -resolution]),
                ]
                
                for neighbor in neighbors:
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple in checked:
                        continue
                    checked.add(neighbor_tuple)
                    
                    # Check if neighbor is free
                    node = octree.search(neighbor)
                    if node is None or octree.isNodeOccupied(node):
                        continue
                    
                    # Check if this free node has unknown neighbors
                    unknown_count = 0
                    second_neighbors = [
                        neighbor + np.array([resolution, 0, 0]),
                        neighbor + np.array([-resolution, 0, 0]),
                        neighbor + np.array([0, resolution, 0]),
                        neighbor + np.array([0, -resolution, 0]),
                        neighbor + np.array([0, 0, resolution]),
                        neighbor + np.array([0, 0, -resolution]),
                    ]
                    
                    for second_neighbor in second_neighbors:
                        second_node = octree.search(second_neighbor)
                        if second_node is None:  # Unknown
                            unknown_count += 1
                    
                    if unknown_count >= min_unknown_neighbors:
                        # Only add frontier if it's within or on the boundary of ROI
                        # Allow small tolerance (half resolution) for numerical precision
                        tolerance = resolution * 0.5
                        if (roi_min[0] - tolerance <= neighbor[0] <= roi_max[0] + tolerance and
                            roi_min[1] - tolerance <= neighbor[1] <= roi_max[1] + tolerance and
                            roi_min[2] - tolerance <= neighbor[2] <= roi_max[2] + tolerance):
                            frontiers.append(neighbor)
    
    return frontiers

def visualize_frontiers_mengine(frontiers: List[np.ndarray], 
                                color: List[float] = [1, 0, 0], 
                                point_size: float = 5.0):
    """
    Visualize frontiers using mengine debug drawing.
    
    Args:
        frontiers: List of frontier positions
        color: RGB color for frontier points
    """
    
    if len(frontiers) == 0:
        return
    
    # Draw frontier points only
    positions = np.array(frontiers)
    colors = np.tile(color, (len(frontiers), 1))
    debug_idxs = m.DebugPoints(positions, points_rgb=colors, size=point_size)
    return debug_idxs

def visualize_octomap(octree: pyoctomap.OcTree,
                    roi_bounds: Optional[np.ndarray] = None,
                    free_color: List[float] = [0, 1, 0],     # Green for free
                    occupied_color: List[float] = [1, 1, 0], # Yellow for occupied
                    point_size: float = 5.0,
                    max_points: int = 10000) -> dict:
    """
    Visualize the octomap with different colors for free and occupied voxels.
    
    Args:
        octree: The OcTree to visualize
        roi_bounds: Optional [x_min, y_min, z_min, x_max, y_max, z_max] to limit visualization
        free_color: RGB color for free voxels [0-1]
        occupied_color: RGB color for occupied voxels [0-1]
        point_size: Size of debug points
        max_points: Maximum number of points to visualize (for performance)
        
    Returns:
        Dictionary with statistics about visualized voxels
    """
    free_points = []
    occupied_points = []
    
    point_count = 0
    
    # Iterate through all leaf nodes in the octree
    for leaf_it in octree.begin_leafs():
        if point_count >= max_points:
            print(f"WARN: Reached max_points limit ({max_points}). Increase limit to see more voxels.")
            break
        
        try:
            coord = leaf_it.getCoordinate()
            point = np.array(coord, dtype=np.float64)
            
            # Check if in ROI (if specified)
            if roi_bounds is not None:
                if not (roi_bounds[0] <= point[0] <= roi_bounds[3] and
                       roi_bounds[1] <= point[1] <= roi_bounds[4] and
                       roi_bounds[2] <= point[2] <= roi_bounds[5]):
                    continue
            
            # Get the node and check occupancy
            node = leaf_it.current_node
            if node is None:
                continue
            
            # Classify as occupied or free
            if octree.isNodeOccupied(node):
                occupied_points.append(point)
            else:
                free_points.append(point)
            
            point_count += 1
            
        except Exception as e:
            continue
    
    # Convert to numpy arrays
    free_points = np.array(free_points) if free_points else np.zeros((0, 3))
    occupied_points = np.array(occupied_points) if occupied_points else np.zeros((0, 3))
    
    # Visualize free voxels (green)
    if len(free_points) > 0:
        free_colors = np.tile(free_color, (len(free_points), 1))
        debug_idxs = DebugPoints(free_points, points_rgb=free_colors, size=point_size)
    
    # Visualize occupied voxels (yellow)
    if len(occupied_points) > 0:
        occupied_colors = np.tile(occupied_color, (len(occupied_points), 1))
        debug_idxs = DebugPoints(occupied_points, points_rgb=occupied_colors, size=point_size)
    
    # Print statistics
    stats = {
        'free_voxels': len(free_points),
        'occupied_voxels': len(occupied_points),
        'total_visualized': len(free_points) + len(occupied_points),
        'total_tree_size': octree.size()
    }
    
    print(f"\nOctomap Visualization:")
    print(f"   Free voxels (green): {stats['free_voxels']}")
    print(f"   Occupied voxels (yellow): {stats['occupied_voxels']}")
    print(f"   Total visualized: {stats['total_visualized']}")
    print(f"   Total tree size: {stats['total_tree_size']}")
    
    return stats, debug_idxs

