# Motion Planning Quick Start Guide

## Overview

The `MotionPlanner` class provides a simplified interface to pybullet_planning for RRT-based collision-free motion planning.

## Installation

pybullet-planning is already installed! ✓

## Basic Usage

```python
from bodies.planning import MotionPlanner

# Create planner with obstacles
planner = MotionPlanner(robot, obstacles=[table_id, box_id])

# Plan to joint configuration
target_joints = [0.5, -0.3, 0.8, -1.2, 0.0, 1.5, 0.0]
path = planner.plan_to_joint_config(target_joints)

if path:
    planner.execute_path(path)
```

## Key Methods

### 1. `plan_to_joint_config(target_joints, **kwargs)`
Plans collision-free path to target joint angles using RRT.

**Parameters:**
- `target_joints`: List/array of target joint angles
- `algorithm`: 'birrt' (default, faster) or 'rrt'
- `iterations`: Max planning iterations (default: 1000)
- `smooth`: Smoothing iterations (default: 50)
- `max_distance`: Step size in config space (default: 0.5)
- `restarts`: Number of planning attempts (default: 2)

**Returns:** List of waypoints or None if planning failed

**Example:**
```python
path = planner.plan_to_joint_config(
    target_joints,
    iterations=2000,
    smooth=100,
    max_distance=0.3
)
```

### 2. `plan_to_ee_pose(target_position, target_orientation, **kwargs)`
Plans to end-effector pose using IK + joint space planning.

**Parameters:**
- `target_position`: [x, y, z] target position
- `target_orientation`: [x, y, z, w] quaternion (optional)
- Plus all `plan_to_joint_config` parameters

**Example:**
```python
path = planner.plan_to_ee_pose(
    target_position=[0.5, 0.2, 0.4],
    target_orientation=[0, 0, 0, 1],
    iterations=1500
)
```

### 3. `execute_path(path, **kwargs)`
Executes planned path by tracking waypoints.

**Parameters:**
- `path`: List of joint configurations from planning
- `speed`: Speed multiplier (default: 1.0, <1 slower, >1 faster)
- `tolerance`: Joint angle tolerance (default: 0.05)
- `timeout_per_waypoint`: Max time per waypoint (default: 5.0s)
- `gains`: PD control gains (default: 0.05)
- `forces`: Max joint forces (default: 500.0)

**Returns:** True if entire path executed successfully

**Example:**
```python
planner.execute_path(path, speed=0.5, tolerance=0.03)
```

### 4. `plan_and_execute(target_joints, **kwargs)`
Convenience method that plans and executes in one call.

**Example:**
```python
success = planner.plan_and_execute(
    target_joints,
    iterations=1500,  # planning param
    speed=0.8        # execution param
)
```

### 5. `check_collision(joint_config)`
Checks if a configuration is collision-free.

**Returns:** True if collision-free, False if in collision

**Example:**
```python
if planner.check_collision(test_config):
    print("Safe configuration!")
```

## Advanced Features

### Smart Moveto (Direct Control + Planning Fallback)

```python
from bodies.planning import smart_moveto

# Try direct control first, fall back to planning if needed
success = smart_moveto(
    robot, 
    planner, 
    target_joints,
    direct_timeout=3.0,  # try direct control for 3 seconds
    iterations=1500      # planning params if needed
)
```

### Custom Collision Pairs

Disable collision checking between specific links:

```python
planner = MotionPlanner(
    robot, 
    obstacles=[table_id],
    self_collisions=True,
    disabled_collision_pairs=[(3, 5), (4, 6)]  # (link_a, link_b)
)
```

### Multi-Waypoint Planning

```python
waypoints = [config1, config2, config3]

for waypoint in waypoints:
    path = planner.plan_to_joint_config(waypoint)
    if path:
        planner.execute_path(path)
```

## Parameter Tuning Guide

### For Cluttered Environments
```python
path = planner.plan_to_joint_config(
    target_joints,
    max_distance=0.2,    # Smaller steps
    iterations=3000,     # More iterations
    smooth=150          # More smoothing
)
```

### For Fast Planning
```python
path = planner.plan_to_joint_config(
    target_joints,
    max_distance=0.8,    # Larger steps
    iterations=500,      # Fewer iterations
    smooth=20           # Less smoothing
)
```

### For Precise Execution
```python
planner.execute_path(
    path,
    speed=0.3,              # Slower
    tolerance=0.01,         # Tighter tolerance
    gains=0.03             # Smaller gains
)
```

## Integration with NBV Planning

```python
# In your NBV loop
for viewpoint in candidate_viewpoints:
    # Convert viewpoint to joint configuration
    target_joints = compute_joint_config_for_viewpoint(viewpoint)
    
    # Try direct control first (fast)
    success = robot.moveto(robot, None, target_joints, timeout=2.0)
    
    # Fall back to planning if direct control fails
    if not success:
        print("Direct control failed, using motion planning...")
        path = planner.plan_to_joint_config(target_joints)
        if path:
            success = planner.execute_path(path)
    
    if success:
        # Capture observations at viewpoint
        update_octomap(viewpoint)
    else:
        print(f"Could not reach viewpoint: {viewpoint}")
```

## Complete Examples

See `bodies/planning_examples.py` for 8 detailed examples:

1. Basic joint space planning
2. End-effector pose planning
3. Multi-waypoint planning
4. Collision checking
5. Smart moveto with fallback
6. Custom planning parameters
7. One-line plan and execute
8. Incremental planning for viewpoint exploration

```python
from bodies.planning_examples import example_basic_planning

example_basic_planning(robot, obstacle_ids)
```

## Troubleshooting

### Planning fails frequently
- Increase `iterations` (try 2000-5000)
- Decrease `max_distance` (try 0.2-0.3)
- Increase `restarts` (try 3-5)
- Check if target configuration is collision-free: `planner.check_collision(target_joints)`

### Execution fails to reach waypoints
- Increase `timeout_per_waypoint`
- Decrease `tolerance`
- Adjust `gains` (lower for smoother, higher for faster)
- Decrease `speed` for better tracking

### Self-collisions detected incorrectly
- Disable specific collision pairs: `disabled_collision_pairs=[(link_a, link_b)]`
- Disable all self-collisions: `self_collisions=False` (not recommended)

## Performance Tips

1. **Use direct control when possible**: Planning has overhead (~0.5-3s)
2. **Cache obstacle lists**: Don't recreate planner for each query
3. **Subsample paths**: Use `speed > 1.0` to skip waypoints
4. **Tune max_distance**: Larger = faster planning but less smooth paths
5. **Use birrt**: Bidirectional RRT is typically faster than unidirectional

## API Summary

```python
# Initialization
planner = MotionPlanner(robot, obstacles, self_collisions, disabled_collision_pairs)

# Planning
path = planner.plan_to_joint_config(target_joints, ...)
path = planner.plan_to_ee_pose(position, orientation, ...)

# Execution  
planner.execute_path(path, speed, tolerance, ...)

# Utilities
is_safe = planner.check_collision(config)
success = planner.plan_and_execute(target_joints, ...)
success = smart_moveto(robot, planner, target_joints, ...)
```
