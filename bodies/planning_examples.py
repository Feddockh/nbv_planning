"""
Examples demonstrating the MotionPlanner interface.

These examples show common usage patterns for motion planning in manipulation tasks.
"""

from bodies.planning import MotionPlanner, smart_moveto
from bodies.robot import Robot
import numpy as np


def example_basic_planning(robot, obstacle_ids):
    """
    Example 1: Basic joint space planning
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Joint Space Planning")
    print("="*80)
    
    # Create planner
    planner = MotionPlanner(robot, obstacles=obstacle_ids)
    
    # Define target configuration
    target_joints = [0.5, -0.3, 0.8, -1.2, 0.0, 1.5, 0.0]
    
    # Plan path
    path = planner.plan_to_joint_config(target_joints)
    
    if path:
        # Execute path
        planner.execute_path(path, speed=0.5)
        return True
    else:
        print("Planning failed!")
        return False


def example_ee_pose_planning(robot, obstacle_ids):
    """
    Example 2: End-effector pose planning
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: End-Effector Pose Planning")
    print("="*80)
    
    planner = MotionPlanner(robot, obstacles=obstacle_ids)
    
    # Define target pose
    target_position = [0.5, 0.2, 0.4]
    target_orientation = [0, 0, 0, 1]  # quaternion [x, y, z, w]
    
    # Plan to pose (IK + joint space planning)
    path = planner.plan_to_ee_pose(
        target_position, 
        target_orientation,
        iterations=2000,  # More iterations for complex scenes
        smooth=100
    )
    
    if path:
        planner.execute_path(path, speed=0.8, tolerance=0.03)
        return True
    return False


def example_multi_waypoint(robot, obstacle_ids):
    """
    Example 3: Planning through multiple waypoints
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-Waypoint Planning")
    print("="*80)
    
    planner = MotionPlanner(robot, obstacles=obstacle_ids)
    
    # Define waypoints
    waypoints = [
        [0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0],
        [0.5, -0.3, 0.8, -1.2, 0.0, 1.5, 0.0],
        [1.0, 0.0, 0.0, -0.8, 0.0, 0.8, 0.0],
    ]
    
    # Plan and execute to each waypoint
    for i, waypoint in enumerate(waypoints):
        print(f"\nPlanning to waypoint {i+1}/{len(waypoints)}...")
        
        path = planner.plan_to_joint_config(waypoint)
        if not path:
            print(f"Failed to plan to waypoint {i+1}")
            return False
        
        success = planner.execute_path(path, speed=1.0)
        if not success:
            print(f"Failed to execute path to waypoint {i+1}")
            return False
    
    print("\n✓ All waypoints reached!")
    return True


def example_collision_checking(robot, obstacle_ids):
    """
    Example 4: Check if configurations are collision-free
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Collision Checking")
    print("="*80)
    
    planner = MotionPlanner(robot, obstacles=obstacle_ids)
    
    # Test several configurations
    test_configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0],
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],  # Likely in collision
    ]
    
    for i, config in enumerate(test_configs):
        is_safe = planner.check_collision(config)
        status = "✓ Safe" if is_safe else "✗ Collision"
        print(f"Config {i+1}: {np.round(config, 2)} - {status}")


def example_smart_moveto(robot, planner, target_joints):
    """
    Example 5: Smart moveto - try direct control, fall back to planning
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Smart Moveto (Direct Control + Planning Fallback)")
    print("="*80)
    
    success = smart_moveto(
        robot, 
        planner, 
        target_joints,
        direct_timeout=3.0,  # Try direct control for 3 seconds
        iterations=1500,     # Planning parameters if needed
        speed=0.8            # Execution speed if planning used
    )
    
    return success


def example_custom_planning_params(robot, obstacle_ids):
    """
    Example 6: Using custom planning parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Planning Parameters")
    print("="*80)
    
    planner = MotionPlanner(
        robot, 
        obstacles=obstacle_ids,
        self_collisions=True  # Enable self-collision checking
    )
    
    target_joints = [0.8, -0.6, 1.0, -1.5, 0.2, 1.3, -0.1]
    
    # Fine-tune planning parameters
    path = planner.plan_to_joint_config(
        target_joints,
        algorithm='birrt',      # Bidirectional RRT (faster than 'rrt')
        max_distance=0.3,       # Smaller steps for cluttered environments
        iterations=2000,        # More iterations for difficult problems
        smooth=100,             # More smoothing for better paths
        restarts=3              # Try up to 3 times if planning fails
    )
    
    if path:
        # Fine-tune execution
        planner.execute_path(
            path,
            speed=0.5,                  # Slower for precision
            tolerance=0.02,             # Tighter tolerance
            timeout_per_waypoint=8.0,   # More time per waypoint
            gains=0.03,                 # Smaller gains for smoother motion
            forces=400.0                # Lower forces
        )
        return True
    return False


def example_plan_and_execute(robot, obstacle_ids):
    """
    Example 7: One-line planning and execution
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Plan and Execute (Convenience Method)")
    print("="*80)
    
    planner = MotionPlanner(robot, obstacles=obstacle_ids)
    
    target_joints = [0.3, -0.4, 0.6, -1.0, 0.1, 1.2, 0.0]
    
    # Plan and execute in one call
    success = planner.plan_and_execute(
        target_joints,
        # Planning params
        iterations=1500,
        smooth=50,
        # Execution params
        speed=0.8,
        tolerance=0.03
    )
    
    return success


def example_incremental_planning(robot, obstacle_ids):
    """
    Example 8: Incremental planning for exploration/NBV
    
    Useful for next-best-view planning where you need to visit
    multiple viewpoints sequentially.
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Incremental Planning for Viewpoint Exploration")
    print("="*80)
    
    planner = MotionPlanner(robot, obstacles=obstacle_ids)
    
    # Simulate a sequence of NBV poses
    viewpoint_configs = [
        [0.2, -0.3, 0.5, -1.1, 0.0, 1.3, 0.0],
        [0.5, -0.5, 0.7, -1.3, 0.2, 1.5, 0.1],
        [0.8, -0.2, 0.4, -0.9, -0.1, 1.1, -0.1],
    ]
    
    total_waypoints = 0
    
    for i, viewpoint in enumerate(viewpoint_configs):
        print(f"\n--- Viewpoint {i+1}/{len(viewpoint_configs)} ---")
        
        # Try smart moveto (direct control + planning fallback)
        success = smart_moveto(
            robot, 
            planner, 
            viewpoint,
            direct_timeout=2.0,
            iterations=1000,
            speed=1.0
        )
        
        if not success:
            print(f"✗ Failed to reach viewpoint {i+1}")
            return False
        
        # Simulate taking observations at this viewpoint
        print(f"  Capturing data at viewpoint {i+1}...")
        # ... your observation code here ...
    
    print(f"\n✓ Visited all {len(viewpoint_configs)} viewpoints successfully!")
    return True


# Quick reference for common patterns
USAGE_PATTERNS = """
QUICK REFERENCE: Common Usage Patterns
=====================================

1. Basic planning:
   planner = MotionPlanner(robot, obstacles=[table_id, box_id])
   path = planner.plan_to_joint_config(target_joints)
   planner.execute_path(path)

2. Plan to end-effector pose:
   path = planner.plan_to_ee_pose([x, y, z], [qx, qy, qz, qw])

3. One-liner:
   planner.plan_and_execute(target_joints, speed=0.8)

4. Smart fallback:
   smart_moveto(robot, planner, target_joints)

5. Check collision:
   is_safe = planner.check_collision(test_config)

6. Custom parameters:
   path = planner.plan_to_joint_config(
       target_joints,
       iterations=2000,
       smooth=100,
       max_distance=0.3
   )

7. Multi-waypoint:
   for waypoint in waypoints:
       path = planner.plan_to_joint_config(waypoint)
       planner.execute_path(path)
"""


if __name__ == "__main__":
    print(USAGE_PATTERNS)
    print("\nTo run examples, import individual example functions.")
    print("Example: from bodies.planning_examples import example_basic_planning")
