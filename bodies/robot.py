# Code from mengine by Zachary Erickson
# Adapted by Hayden Feddock

import os
import time
from typing import List
import numpy as np
import pybullet as p

import env
from bodies.body import Body
from scene.objects import DebugPoints

class Robot(Body):
    def __init__(self, body, env, controllable_joints, end_effector, gripper_joints, action_duplication=None, action_multiplier=1):
        self.end_effector = end_effector # Used to get the pose of the end effector
        self.gripper_joints = gripper_joints # Gripper actuated joints
        self.action_duplication = action_duplication # The Stretch RE1 robot has a telescoping arm. The 4 linear actuators should be treated as a single actuator
        self.action_multiplier = action_multiplier
        # TODO: remove joint limits from wheels and continuous actuators
        # if self.mobile:
        #     self.controllable_joint_lower_limits[:len(self.wheel_joint_indices)] = -np.inf
        #     self.controllable_joint_upper_limits[:len(self.wheel_joint_indices)] = np.inf
        super().__init__(body, env, controllable_joints)

    def set_gripper_position(self, positions, set_instantly=False, force=500):
        self.control(positions, joints=self.gripper_joints, gains=np.array([0.05]*len(self.gripper_joints)), forces=[force]*len(self.gripper_joints), velocity_control=False, set_instantly=set_instantly)


class ManipulationWorkspace:
    """
    3D voxel grid for fast O(1) workspace queries.
    Discretizes the workspace into voxels and marks reachable regions.
    """
    def __init__(self, robot: Robot, bounds: List[float] = None, resolution=0.05):
        """
        Args:
            robot: Robot instance
            bounds: [x_min, y_min, z_min, x_max, y_max, z_max] or None
            resolution: voxel size in meters
        """
        self.robot = robot
        self.resolution = resolution
        
        # Initialize bounds and grid
        self.bounds = None
        self.grid_size = None
        self.grid = None
        
        # Use the provided bounds for now, will be updated when learning
        if bounds is not None:
            self.bounds = np.array(bounds, dtype=np.float64)
            # Calculate grid dimensions
            self.grid_size = np.ceil((self.bounds[3:] - self.bounds[:3]) / resolution).astype(int)
            # Initialize empty grid (False = unreachable, True = reachable)
            self.grid = np.zeros(self.grid_size, dtype=bool)
            print(f"Workspace grid initialized: {self.grid_size} voxels ({np.prod(self.grid_size)} total)")
        else:
            print(f"Workspace initialized without bounds. Call learn() to compute bounds and grid.")
    
    def world_to_voxel(self, point):
        """Convert world coordinates to voxel indices."""
        if self.bounds is None:
            raise ValueError("Workspace bounds are not set.")
        point = np.array(point)
        voxel_indices = np.floor((point - self.bounds[:3]) / self.resolution).astype(int)
        return voxel_indices
    
    def voxel_to_world(self, voxel_indices):
        """Convert voxel indices to world coordinates (voxel center)."""
        if self.bounds is None: 
            raise ValueError("Workspace bounds are not set.")
        return self.bounds[:3] + (voxel_indices + 0.5) * self.resolution
    
    def is_valid_voxel(self, voxel_indices):
        """Check if voxel indices are within grid bounds."""
        if self.grid_size is None:
            raise ValueError("Workspace grid size is not set.")
        return np.all(voxel_indices >= 0) and np.all(voxel_indices < self.grid_size)
    
    def mark_reachable(self, points):
        """
        Mark voxels as reachable based on a list of points.
        
        Args:
            points: Nx3 array of world coordinates
        """
        if self.grid is None:
            raise ValueError("Workspace grid is not initialized.")
        for point in points:
            voxel = self.world_to_voxel(point)
            if self.is_valid_voxel(voxel):
                self.grid[tuple(voxel)] = True
    
    def is_reachable(self, point):
        """
        O(1) lookup: Check if a point is in the reachable workspace.
        
        Args:
            point: [x, y, z] world coordinates
            
        Returns:
            bool: True if reachable, False otherwise
        """
        if self.grid is None:
            raise ValueError("Workspace grid is not initialized.")
        voxel = self.world_to_voxel(point)
        if not self.is_valid_voxel(voxel):
            return False
        return self.grid[tuple(voxel)]
    
    def get_reachable_volume(self):
        """Calculate the volume of reachable workspace in cubic meters."""
        if self.grid is None:
            raise ValueError("Workspace grid is not initialized.")
        num_reachable = np.sum(self.grid)
        voxel_volume = self.resolution ** 3
        return num_reachable * voxel_volume
    
    def get_reachable_voxel_centers(self):
        """Get world coordinates of all reachable voxel centers."""
        if self.grid is None:
            raise ValueError("Workspace grid is not initialized.")
        reachable_indices = np.argwhere(self.grid)
        return np.array([self.voxel_to_world(idx) for idx in reachable_indices])
    
    def min_distance_to_workspace(self, point):
        """
        Find the minimum distance between a point and the reachable workspace.
        
        Args:
            point: [x, y, z] world coordinates
            
        Returns:
            float: Minimum distance to the nearest reachable voxel center.
                   Returns 0.0 if the point is inside a reachable voxel.
                   Returns inf if workspace is empty.
        """
        if self.grid is None:
            raise ValueError("Workspace grid is not initialized.")
        
        point = np.array(point)
        
        # Check if point is already in a reachable voxel
        if self.is_reachable(point):
            return 0.0
        
        # Get all reachable voxel centers
        reachable_centers = self.get_reachable_voxel_centers()
        
        if len(reachable_centers) == 0:
            return np.inf
        
        # Compute distances to all reachable voxel centers
        distances = np.linalg.norm(reachable_centers - point, axis=1)
        
        # Return minimum distance (subtract half voxel diagonal for surface distance)
        # This accounts for the fact that we compute to voxel centers
        min_dist = np.min(distances)
        voxel_half_diagonal = self.resolution * np.sqrt(3) / 2.0
        
        # Return the distance to the voxel surface
        return max(0.0, min_dist - voxel_half_diagonal)
    
    def visualize(self, color=[0, 1, 0], point_size=5, env_param=None):
        """Visualize reachable workspace using mengine debug points."""
        if self.grid is None:
            raise ValueError("Workspace grid is not initialized. Call learn() first.")
        centers = self.get_reachable_voxel_centers()
        if len(centers) > 0:
            DebugPoints(centers, points_rgb=[color] * len(centers), size=point_size, env_param=env_param)
            print(f"Visualized {len(centers)} reachable voxels")
        else:
            print("No reachable voxels to visualize")
    
    def save(self, filename):
        """Save workspace grid to file."""
        filepath = os.path.join(env.asset_dir, filename)
        np.savez_compressed(filepath, 
                          grid=self.grid,
                          bounds=self.bounds,
                          resolution=self.resolution)
        print(f"Workspace saved to {filepath}")
    
    def load(self, filename):
        """Load workspace grid from file and initialize a ManipulationWorkspace object."""
        filepath = os.path.join(env.asset_dir, filename)
        data = np.load(filepath)
        self.grid = data['grid'].astype(bool)
        self.bounds = np.array(data['bounds'], dtype=np.float64)
        self.resolution = float(data['resolution'])
        
        # Recalculate grid_size to ensure consistency
        self.grid_size = np.ceil((self.bounds[3:] - self.bounds[:3]) / self.resolution).astype(int)
        
        print(f"Workspace loaded from {filename}")
        print(f"  Grid shape: {self.grid.shape}")
        print(f"  Bounds: {self.bounds}")
        print(f"  Resolution: {self.resolution}")
        return self

    def learn(self, num_samples=10000):
        
        points = []
        prev_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joints)

        print(f"Sampling {num_samples} random configurations...")
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(f"  Sampled {i + 1}/{num_samples} configurations...")

            # Sample random joint angles within joint limits
            random_joint_angles = []
            for joint_idx in self.robot.controllable_joints:
                lower = self.robot.lower_limits[joint_idx]
                upper = self.robot.upper_limits[joint_idx]
                random_angle = np.random.uniform(lower, upper)
                random_joint_angles.append(random_angle)
            random_joint_angles = np.array(random_joint_angles)
            
            # Set instantly without simulation for speed
            self.robot.control(random_joint_angles, set_instantly=True)
            ee_pos, _ = self.robot.get_link_pos_orient(self.robot.end_effector)

            points.append(ee_pos)
        
        # Restore original configuration
        self.robot.control(prev_joint_angles, set_instantly=True)
        
        points = np.array(points)
        print(f"Captured {len(points)} end-effector positions")
        
        # Re-compute bounds with padding
        min_bounds = np.min(points, axis=0) - self.resolution
        max_bounds = np.max(points, axis=0) + self.resolution
        self.bounds = np.concatenate([min_bounds, max_bounds])
        self.grid_size = np.ceil((self.bounds[3:] - self.bounds[:3]) / self.resolution).astype(int)
        self.grid = np.zeros(self.grid_size, dtype=bool)
        print(f"Auto-computed bounds: {self.bounds}")

        self.mark_reachable(points)

        reachable_volume = self.get_reachable_volume()
        coverage = np.sum(self.grid) / np.prod(self.grid_size) * 100

        print(f"Workspace statistics:")
        print(f"  Reachable volume: {reachable_volume:.4f} m³")
        print(f"  Grid coverage: {coverage:.2f}%")
        print(f"  Reachable voxels: {np.sum(self.grid)}/{np.prod(self.grid_size)}")

        return self

def robot_in_collision(robot: Robot, joint_angles: np.ndarray, obstacles: list = []) -> bool:
    """Returns True if the robot is in collision at the given joint angles (q).
    For simplicity, we only consider robot collision with table and objects.
    Robot self collision or collision with cubes is optional.
    """
    # set robot to joint angles
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(joint_angles, set_instantly=True)

    # robot-obstacle collision
    for obstacle in obstacles:
        if len(robot.get_closest_points(obstacle, distance=0)[-1]) != 0:
            robot.control(prev_joint_angles, set_instantly=True)
            return True

    robot.control(prev_joint_angles, set_instantly=True)
    return False

def moveto(robot: Robot, ee_pose: tuple = None, joint_angles: np.ndarray = None, tolerance: float = 0.03):
    """Move robot to a given ee_pose or joint angles. If both are given, ee_pose is used."""
    if ee_pose is not None:
        joint_angles = robot.ik(robot.end_effector, target_pos=ee_pose[0], target_orient=ee_pose[1],
                                use_current_joint_angles=True)
    if joint_angles is None:
        return

    robot.control(joint_angles, gains=0.05)
    while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - joint_angles) > tolerance:
        env.step_simulation(realtime=True)
    return

