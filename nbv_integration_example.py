"""
Example integration of motion planning with NBV system.

This shows how to integrate the MotionPlanner into your existing
next-best-view planning workflow.
"""

import numpy as np
from bodies.planning import MotionPlanner, smart_moveto
from viewpoints.viewpoint_selection import compute_information_gain


def integrate_planning_with_nbv_simple(robot, octomap, scene_obstacles):
    """
    Simple integration: Add motion planning to existing NBV loop.
    
    This example shows minimal changes to add planning capability.
    """
    # ONE-TIME SETUP: Create planner at start
    planner = MotionPlanner(
        robot, 
        obstacles=scene_obstacles,
        self_collisions=True
    )
    
    # Your existing NBV loop
    for iteration in range(max_iterations):
        # 1. Generate candidate viewpoints (your existing code)
        candidate_viewpoints = generate_candidate_viewpoints(octomap)
        
        # 2. Score viewpoints (your existing code)
        best_viewpoint = None
        best_score = -np.inf
        
        for viewpoint in candidate_viewpoints:
            score = compute_information_gain(viewpoint, octomap)
            if score > best_score:
                best_score = score
                best_viewpoint = viewpoint
        
        # 3. Move to best viewpoint - USE PLANNING HERE
        target_position = best_viewpoint['position']
        target_orientation = best_viewpoint['orientation']
        
        # Option A: Plan to end-effector pose directly
        path = planner.plan_to_ee_pose(target_position, target_orientation)
        if path:
            planner.execute_path(path)
        
        # Option B: If you compute joint config yourself, use this
        # target_joints = your_ik_solver(target_position, target_orientation)
        # planner.plan_and_execute(target_joints)
        
        # 4. Capture observations (your existing code)
        capture_observations()
        update_octomap(octomap)


def integrate_planning_with_nbv_smart(robot, octomap, scene_obstacles):
    """
    Smart integration: Use direct control + planning fallback.
    
    This is faster because it only uses planning when needed.
    """
    # Setup planner
    planner = MotionPlanner(robot, obstacles=scene_obstacles)
    
    for iteration in range(max_iterations):
        # Generate and score viewpoints
        candidate_viewpoints = generate_candidate_viewpoints(octomap)
        best_viewpoint = select_best_viewpoint(candidate_viewpoints, octomap)
        
        # Compute target joint configuration
        target_joints = compute_joint_config_for_viewpoint(best_viewpoint)
        
        # SMART APPROACH: Try direct control, fall back to planning
        success = smart_moveto(
            robot, 
            planner, 
            target_joints,
            direct_timeout=2.0,      # Try direct control for 2 seconds
            iterations=1500,         # Planning iterations if needed
            speed=0.8               # Execution speed
        )
        
        if success:
            # Capture data at this viewpoint
            observations = capture_observations()
            update_octomap(octomap, observations)
            
            # Check termination criteria
            if is_reconstruction_complete(octomap):
                break
        else:
            print(f"Could not reach viewpoint at iteration {iteration}")


def integrate_planning_with_nbv_advanced(robot, octomap, scene_obstacles):
    """
    Advanced integration: Reachability checking and multi-step planning.
    
    This checks if viewpoints are reachable before committing to them.
    """
    planner = MotionPlanner(robot, obstacles=scene_obstacles)
    
    visited_viewpoints = []
    
    for iteration in range(max_iterations):
        print(f"\n=== NBV Iteration {iteration+1} ===")
        
        # 1. Generate candidates
        candidate_viewpoints = generate_candidate_viewpoints(octomap)
        print(f"Generated {len(candidate_viewpoints)} candidate viewpoints")
        
        # 2. Filter unreachable viewpoints (pre-check with collision detection)
        reachable_candidates = []
        for viewpoint in candidate_viewpoints:
            target_joints = compute_joint_config_for_viewpoint(viewpoint)
            
            # Quick collision check
            if planner.check_collision(target_joints):
                reachable_candidates.append({
                    'viewpoint': viewpoint,
                    'joints': target_joints
                })
        
        print(f"Filtered to {len(reachable_candidates)} reachable candidates")
        
        if not reachable_candidates:
            print("No reachable viewpoints found!")
            break
        
        # 3. Score only reachable viewpoints
        best_candidate = None
        best_score = -np.inf
        
        for candidate in reachable_candidates:
            score = compute_information_gain(candidate['viewpoint'], octomap)
            
            # Penalize viewpoints that are far from current position
            current_joints = np.array([robot.get_joint_position(j) 
                                      for j in robot.controllable_joints])
            distance_penalty = np.linalg.norm(candidate['joints'] - current_joints)
            adjusted_score = score - 0.1 * distance_penalty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_candidate = candidate
        
        print(f"Best viewpoint score: {best_score:.3f}")
        
        # 4. Plan and execute to best viewpoint
        print("Planning path to best viewpoint...")
        path = planner.plan_to_joint_config(
            best_candidate['joints'],
            iterations=2000,
            smooth=100
        )
        
        if path:
            print(f"Path found with {len(path)} waypoints, executing...")
            success = planner.execute_path(path, speed=0.8)
            
            if success:
                # Capture observations
                print("Capturing observations...")
                observations = capture_observations()
                update_octomap(octomap, observations)
                visited_viewpoints.append(best_candidate['viewpoint'])
                
                # Check termination
                coverage = compute_coverage(octomap)
                print(f"Scene coverage: {coverage*100:.1f}%")
                
                if coverage > 0.95:
                    print("Reconstruction complete!")
                    break
            else:
                print("Failed to execute path!")
        else:
            print("Planning failed!")
    
    print(f"\nVisited {len(visited_viewpoints)} viewpoints")


def integrate_with_frontier_exploration(robot, octomap, scene_obstacles):
    """
    Integration with frontier-based exploration.
    
    Shows how to use planning for frontier exploration with clustering.
    """
    from scene.scene_representation import find_frontiers, cluster_frontiers
    
    planner = MotionPlanner(robot, obstacles=scene_obstacles)
    
    while True:
        # 1. Find and cluster frontiers
        frontiers = find_frontiers(octomap)
        
        if len(frontiers) < 10:
            print("Few frontiers remaining, exploration complete!")
            break
        
        # Cluster frontiers
        clusters = cluster_frontiers(
            frontiers, 
            eps=2*octomap.resolution,
            min_samples=5
        )
        
        print(f"Found {len(np.unique(clusters))} frontier clusters")
        
        # 2. Generate viewpoints for each cluster
        candidate_viewpoints = []
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_points = frontiers[clusters == cluster_id]
            cluster_center = np.mean(cluster_points, axis=0)
            
            # Generate viewpoints observing this cluster
            viewpoints = generate_viewpoints_for_target(cluster_center)
            candidate_viewpoints.extend(viewpoints)
        
        # 3. Score and select best viewpoint
        best_viewpoint = None
        best_score = -np.inf
        
        for vp in candidate_viewpoints:
            score = compute_information_gain(vp, octomap)
            if score > best_score:
                best_score = score
                best_viewpoint = vp
        
        # 4. Plan and move to viewpoint
        target_joints = compute_joint_config_for_viewpoint(best_viewpoint)
        
        success = smart_moveto(
            robot, 
            planner, 
            target_joints,
            direct_timeout=2.0
        )
        
        if success:
            capture_observations()
            update_octomap(octomap)
        else:
            print("Could not reach viewpoint, trying next best...")


# ============================================================================
# Helper functions (placeholders for your actual implementations)
# ============================================================================

def generate_candidate_viewpoints(octomap):
    """Generate candidate viewpoints (your implementation)."""
    # Your viewpoint generation code here
    pass

def select_best_viewpoint(candidates, octomap):
    """Select best viewpoint by information gain."""
    # Your viewpoint selection code here
    pass

def compute_joint_config_for_viewpoint(viewpoint):
    """Compute joint configuration for viewpoint (IK)."""
    # Your IK solver here
    pass

def capture_observations():
    """Capture sensor observations at current pose."""
    # Your observation capture code here
    pass

def update_octomap(octomap, observations=None):
    """Update octomap with new observations."""
    # Your octomap update code here
    pass

def is_reconstruction_complete(octomap):
    """Check if scene reconstruction is complete."""
    # Your termination criteria here
    pass

def compute_coverage(octomap):
    """Compute scene coverage metric."""
    # Your coverage computation here
    pass

def generate_viewpoints_for_target(target_position):
    """Generate viewpoints observing a target position."""
    # Your viewpoint generation for target
    pass


# ============================================================================
# Usage example
# ============================================================================

if __name__ == "__main__":
    print("NBV Planning + Motion Planning Integration Examples")
    print("\nAvailable integration patterns:")
    print("  1. integrate_planning_with_nbv_simple() - Basic integration")
    print("  2. integrate_planning_with_nbv_smart() - Smart fallback approach")
    print("  3. integrate_planning_with_nbv_advanced() - Full reachability checking")
    print("  4. integrate_with_frontier_exploration() - Frontier-based exploration")
    print("\nTo use, call the appropriate function with your robot, octomap, and obstacles.")
