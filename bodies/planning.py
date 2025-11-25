"""
Motion Planning Interface using pybullet_planning.

Provides a simplified interface to pybullet_planning for RRT-based motion planning
with collision checking for robot manipulation tasks.

Usage:
    from bodies.planning import MotionPlanner
    
    # Create planner
    planner = MotionPlanner(robot, obstacles=[table_id, box_id])
    
    # Plan to joint configuration
    path = planner.plan_to_joint_config(target_joints)
    if path:
        planner.execute_path(path)
    
    # Plan to end-effector pose
    success = planner.plan_to_ee_pose(target_position, target_orientation)
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Union
import pybullet as p
from bodies.robot import moveto

try:
    from pybullet_planning import (
        get_joint_positions,
        set_joint_positions,
        get_movable_joints,
        link_from_name,
        rrt,
        birrt,
        MAX_DISTANCE,
        get_sample_fn,
        get_distance_fn,
        get_extend_fn,
        get_collision_fn,
        check_initial_end,
    )
    PLANNING_AVAILABLE = True
except ImportError:
    PLANNING_AVAILABLE = False
    print("WARNING: pybullet_planning not available. Install with: pip install pybullet-planning")


class MotionPlanner:
    """
    High-level interface for RRT-based motion planning.
    
    Wraps pybullet_planning to provide convenient methods for:
    - Joint space planning with collision checking
    - Task space planning (plan to end-effector pose)
    - Path execution with tracking
    """
    
    def __init__(self, 
                 robot,
                 obstacles: List[int] = None,
                 self_collisions: bool = True,
                 disabled_collision_pairs: List[Tuple[int, int]] = None):
        """
        Initialize motion planner.
        
        Args:
            robot: Robot instance with body attribute (PyBullet body ID)
            obstacles: List of PyBullet body IDs to avoid colliding with
            self_collisions: Enable self-collision checking
            disabled_collision_pairs: List of (link_a, link_b) pairs to ignore for self-collisions
        """
        if not PLANNING_AVAILABLE:
            raise ImportError("pybullet_planning is not installed")
        
        self.robot = robot
        
        # Get the integer body ID (robot.body should be an integer)
        self.body_id = robot.body
        if not isinstance(self.body_id, int):
            if hasattr(self.body_id, 'body'):
                self.body_id = self.body_id.body
        
        if not isinstance(self.body_id, int):
            raise TypeError(f"Expected body_id to be int, got {type(self.body_id)}: {self.body_id}")
        
        # Convert obstacles to integer IDs if they are Body objects
        if obstacles:
            self.obstacles = []
            for obs in obstacles:
                if isinstance(obs, int):
                    self.obstacles.append(obs)
                elif hasattr(obs, 'body'):
                    # It's a Body object, extract the integer ID
                    self.obstacles.append(obs.body)
                else:
                    raise TypeError(f"Obstacle must be int or Body object, got {type(obs)}")
        else:
            self.obstacles = []
        self.self_collisions = self_collisions
        self.disabled_collision_pairs = disabled_collision_pairs if disabled_collision_pairs else []
        
        # Get movable joints from pybullet_planning
        self.movable_joints = get_movable_joints(self.body_id)
        
        print(f"MotionPlanner initialized:")
        print(f"  Robot ID: {self.body_id}")
        print(f"  Movable joints: {self.movable_joints}")
        print(f"  Obstacles: {len(self.obstacles)}")
        print(f"  Self-collisions: {self.self_collisions}")

    def _plan_joint_motion(self, body, joints, end_conf, obstacles=[], attachments=[],
                    self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                    weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={}, 
                    diagnosis=False, algorithm='birrt', **kwargs):
        """call birrt to plan a joint trajectory from the robot's **current** conf to ``end_conf``.
        """
        assert len(joints) == len(end_conf)
        sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
        distance_fn = get_distance_fn(body, joints, weights=weights)
        extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
        collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                        disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                        custom_limits=custom_limits, max_distance=max_distance)

        start_conf = get_joint_positions(body, joints)

        if not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=diagnosis):
            return None
        if algorithm == 'birrt':
            return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
        else:
            return rrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
    
    def plan_to_joint_config(self,
                            target_joints: Union[List[float], np.ndarray],
                            joints: List[int] = None,
                            obstacles: List[int] = None,
                            self_collisions: bool = None,
                            algorithm: str = 'rrt',
                            max_distance: float = 0.5,
                            iterations: int = 1000,
                            smooth: int = 50,
                            restarts: int = 2) -> Optional[List[List[float]]]:
        """
        Plan collision-free path to target joint configuration using RRT.
        
        Args:
            target_joints: Target joint angles
            joints: Joint indices to plan for (default: all controllable joints)
            obstacles: Override default obstacles
            self_collisions: Override self-collision checking
            algorithm: 'birrt' (bidirectional) or 'rrt' 
            max_distance: Maximum step size in configuration space
            iterations: Maximum planning iterations
            smooth: Number of smoothing iterations (0 to disable)
            restarts: Number of planning attempts
            
        Returns:
            List of joint configurations forming a path, or None if planning failed
        """
        if joints is None:
            joints = self.robot.controllable_joints
        if obstacles is None:
            obstacles = self.obstacles
        if self_collisions is None:
            self_collisions = self.self_collisions
        
        # Convert to list if numpy array
        if isinstance(target_joints, np.ndarray):
            target_joints = target_joints.tolist()
        
        # Get current joint positions
        current_joints = [p.getJointState(self.body_id, j)[0] for j in joints]
        
        print(f"Planning from {len(current_joints)} joints to target...")
        print(f"  Current: {np.round(current_joints, 3)}")
        print(f"  Target:  {np.round(target_joints, 3)}")
        
        start_time = time.time()
        
        # Try planning with restarts
        for attempt in range(restarts):
            try:
                path = self._plan_joint_motion(
                    self.body_id,
                    joints,
                    target_joints,
                    obstacles=obstacles,
                    self_collisions=self_collisions,
                    disabled_collisions=set(self.disabled_collision_pairs),
                    max_distance=max_distance,
                    algorithm=algorithm
                )
                
                if path is not None:
                    # Smooth the path
                    if smooth > 0:
                        # Simple path smoothing: remove redundant waypoints
                        path = self._smooth_path(path, joints, obstacles, self_collisions, smooth)
                    
                    elapsed = time.time() - start_time
                    print(f"  Planning succeeded in {elapsed:.2f}s (attempt {attempt+1}/{restarts})")
                    print(f"  Path length: {len(path)} waypoints")
                    return path
                
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
        
        elapsed = time.time() - start_time
        print(f"  Planning failed after {elapsed:.2f}s and {restarts} attempts")
        return None
    
    def plan_to_ee_pose(self,
                       target_position: Union[List[float], np.ndarray],
                       target_orientation: Union[List[float], np.ndarray] = None,
                       **planning_kwargs) -> Optional[List[List[float]]]:
        """
        Plan to end-effector pose using IK + joint space planning.
        
        Args:
            target_position: [x, y, z] target position
            target_orientation: [x, y, z, w] quaternion (optional)
            **planning_kwargs: Additional arguments for plan_to_joint_config
            
        Returns:
            Path to target pose, or None if IK or planning failed
        """
        # Convert to lists if numpy arrays
        if isinstance(target_position, np.ndarray):
            target_position = target_position.tolist()
        if target_orientation is not None and isinstance(target_orientation, np.ndarray):
            target_orientation = target_orientation.tolist()
        
        # Compute IK
        print("Computing IK solution...")
        ee_link = self.robot.end_effector
        
        if target_orientation is None:
            # Use current orientation if not specified
            current_ee_state = p.getLinkState(self.body_id, ee_link)
            target_orientation = current_ee_state[1]  # quaternion
        
        # PyBullet IK
        ik_solution = p.calculateInverseKinematics(
            self.body_id,
            ee_link,
            target_position,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # Extract only controllable joints
        target_joints = [ik_solution[j] for j in self.robot.controllable_joints]
        
        print(f"  IK solution: {np.round(target_joints, 3)}")
        
        # Plan to IK solution
        return self.plan_to_joint_config(target_joints, **planning_kwargs)
    
    def execute_path(self,
                    path: List[List[float]],
                    joints: List[int] = None,
                    speed: float = 1.0,
                    tolerance: float = 0.05,
                    timeout_per_waypoint: float = 5.0,
                    gains: float = 0.05,
                    forces: float = 500.0) -> bool:
        """
        Execute a planned path by tracking waypoints.
        
        Args:
            path: List of joint configurations
            joints: Joint indices (default: controllable joints)
            speed: Speed multiplier (< 1 is slower, > 1 is faster)
            tolerance: Joint angle tolerance for waypoint arrival
            timeout_per_waypoint: Maximum time to reach each waypoint
            gains: PD control gains
            forces: Maximum joint forces
            
        Returns:
            True if entire path executed successfully
        """
        if joints is None:
            joints = self.robot.controllable_joints
        
        print(f"Executing path with {len(path)} waypoints...")
        
        # Subsample path based on speed
        step = max(1, int(1.0 / speed))
        subsampled_path = path[::step]
        if path[-1] not in subsampled_path:
            subsampled_path.append(path[-1])  # Always include final waypoint
        
        print(f"  Subsampled to {len(subsampled_path)} waypoints (speed={speed})")
        
        for i, target_config in enumerate(subsampled_path):
            print(f"  Waypoint {i+1}/{len(subsampled_path)}: {np.round(target_config, 3)}")
            
            # Use robot's moveto function
            success = moveto(
                robot=self.robot,
                ee_pose=None,
                joint_angles=target_config,
                tolerance=tolerance,
                timeout=timeout_per_waypoint,
                gains=gains,
                forces=forces
            )
            
            if not success:
                print(f"  Failed to reach waypoint {i+1}")
                return False
        
        print("  Path execution completed successfully")
        return True
    
    def plan_and_execute(self,
                        target_joints: Union[List[float], np.ndarray],
                        **kwargs) -> bool:
        """
        Convenience method: plan and execute in one call.
        
        Args:
            target_joints: Target joint configuration
            **kwargs: Arguments for planning and execution
            
        Returns:
            True if both planning and execution succeeded
        """
        # Separate planning and execution kwargs
        planning_keys = {'joints', 'obstacles', 'self_collisions', 'algorithm', 
                        'max_distance', 'iterations', 'smooth', 'restarts'}
        execution_keys = {'speed', 'tolerance', 'timeout_per_waypoint', 'gains', 'forces'}
        
        planning_kwargs = {k: v for k, v in kwargs.items() if k in planning_keys}
        execution_kwargs = {k: v for k, v in kwargs.items() if k in execution_keys}
        
        # Plan
        path = self.plan_to_joint_config(target_joints, **planning_kwargs)
        if path is None:
            return False
        
        # Execute
        return self.execute_path(path, **execution_kwargs)
    
    def check_collision(self,
                       joint_config: Union[List[float], np.ndarray],
                       joints: List[int] = None) -> bool:
        """
        Check if a joint configuration is in collision.
        
        Args:
            joint_config: Joint angles to test
            joints: Joint indices (default: controllable joints)
            
        Returns:
            True if configuration is collision-free
        """
        if joints is None:
            joints = self.robot.controllable_joints
        
        # Save current state
        current_state = [p.getJointState(self.body_id, j)[0] for j in joints]
        
        # Set test configuration
        for joint, angle in zip(joints, joint_config):
            p.resetJointState(self.body_id, joint, angle)
        
        # Check collisions
        collision_free = True
        
        # Check against obstacles
        for obstacle_id in self.obstacles:
            contacts = p.getClosestPoints(self.body_id, obstacle_id, distance=0.0)
            if len(contacts) > 0:
                collision_free = False
                break
        
        # Check self-collisions if enabled
        if collision_free and self.self_collisions:
            # Get all link pairs
            num_links = p.getNumJoints(self.body_id)
            for i in range(-1, num_links):
                for j in range(i+1, num_links):
                    if (i, j) not in self.disabled_collision_pairs:
                        contacts = p.getClosestPoints(self.body_id, self.body_id, 
                                                     distance=0.0, linkIndexA=i, linkIndexB=j)
                        if len(contacts) > 0:
                            collision_free = False
                            break
                if not collision_free:
                    break
        
        # Restore original state
        for joint, angle in zip(joints, current_state):
            p.resetJointState(self.body_id, joint, angle)
        
        return collision_free
    
    def _smooth_path(self, path, joints, obstacles, self_collisions, iterations):
        """
        Simple path smoothing: try to shortcut between waypoints.
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to connect current to furthest reachable waypoint
            for j in range(len(path) - 1, i, -1):
                if self._is_path_collision_free(path[i], path[j], joints, obstacles, self_collisions):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                # Couldn't skip any, just add next waypoint
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed
    
    def _is_path_collision_free(self, start, end, joints, obstacles, self_collisions, samples=10):
        """
        Check if linear interpolation between two configs is collision-free.
        """
        start = np.array(start)
        end = np.array(end)
        
        for alpha in np.linspace(0, 1, samples):
            config = start + alpha * (end - start)
            if not self.check_collision(config, joints):
                return False
        return True