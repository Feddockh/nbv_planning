"""
Planar NBV Planning Demo (Baseline)

Baseline demonstration using a simple planar grid of viewpoints.
The robot systematically visits viewpoints on a discretized plane without 
computing information gain or utility. This serves as a baseline for comparison
with volumetric and semantic NBV planners.
"""

import sys
import os
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision import RobotCamera
import env
from bodies.robot import ManipulationWorkspace, moveto, robot_in_collision
from bodies.panda import Panda
from utils import get_quaternion, quat_to_normal
from scene.roi import SphereROI, RectangleROI, ROI
from scene.scene_representation import OctoMap, SemanticOctoMap
from scene.objects import visualize_coordinate_frame, clear_debug_items, DebugPoints, Ground, URDF, load_object, DebugCoordinateFrame
from viewpoints.viewpoint import Viewpoint, visualize_viewpoint, compute_viewpoint_joint_angles
from bodies.planning import MotionPlanner
from detection.fire_blight_detector import FireBlightDetector
from metrics.metrics_logger import MetricsLogger
from metrics.evaluate_semantic_map import SemanticMapDistanceEvaluator


# ===== Configuration =====
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAME = "planar_nbv_default"  # Subfolder name for outputs
GROUND_TRUTH_FILE = os.path.join(repo_dir, "assets", "apple_tree_semantic_ground_truths.json")  # Path to ground truth file for evaluation
LOGGING = True  # Whether to log and evaluate semantic mapping performance
EVAL_DISTANCE_THRESHOLD = 0.1  # Distance threshold for semantic evaluation (in meters)
LEARN_WORKSPACE = False
OCTOMAP_RESOLUTION = 0.02
MAX_ITERATIONS = 30
CAMERA_WIDTH = 1440
CAMERA_HEIGHT = 1080
CAMERA_FOV = 60
MAX_RANGE = 0.5
MIN_RANGE = 0.05
MISMATCH_PENALTY = 0.1
CONFIDENCE_BOOST = 0.05
CONFIDENCE_THRESHOLD = 0.5

# Planar grid configuration
PLANE_HALF_EXTENT = 0.4  # Half-size of the plane grid (meters)
SPATIAL_RESOLUTION = 0.2  # Distance between viewpoints on the plane (meters)

# ===== Main Script =====
# Create environment and ground
nbv_env = env.Env()
ground = Ground(filename=os.path.join(nbv_env.asset_dir, 'dirt_plane', 'dirt_plane.urdf'))
table = URDF(filename=os.path.join(nbv_env.asset_dir, 'table', 'table.urdf'), 
               static=True, position=[1.5, 0, 0], orientation=[0, 0, 0, 1])
obj = load_object("apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8])
obstacles = [obj, table]

# Get the bounding box
b_min, b_max = obj.get_AABB()
size = np.array(b_max) - np.array(b_min)
half_size = size / 2.0
center = (np.array(b_max) + np.array(b_min)) / 2.0
obj_roi = RectangleROI(center=center, half_extents=half_size)
obj_roi.visualize(lines_rgb=[0, 0, 1])

# Let objects settle
env.step_simulation(steps=100, realtime=True)

# Create robot (position it to the side, on the table surface)
robot = Panda(position=[1.0, 0, 0.76], fixed_base=True)

# Learn the workspace for the robotic arm by sampling valid IK configurations
manip_workspace = ManipulationWorkspace(robot, resolution=0.05)
if LEARN_WORKSPACE:
    print("Learning robot workspace...")
    manip_workspace.learn(num_samples=10000000)
    print("Saving workspace...")
    manip_workspace.save("workspace.npz")
else:
    print("Loading known workspace...")
    manip_workspace.load("workspace.npz")

# Create camera and detector
camera_offset_pos = np.array([0, 0, 0.01])
camera_offset_orient = get_quaternion([0, 0, 0])
camera = RobotCamera(robot, robot.end_effector,
                     camera_offset_pos=camera_offset_pos,
                     camera_offset_orient=camera_offset_orient,
                     fov=CAMERA_FOV, camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT)
detector = FireBlightDetector(model_path="best_sim.pt", confidence_threshold=CONFIDENCE_THRESHOLD)

# Initialize octomap
semantic_octomap = SemanticOctoMap(bounds=obj_roi, resolution=OCTOMAP_RESOLUTION)
semantic_octomap.set_class_names({
    -1: "Background",
    0: "Shepherd's Crook",
    1: "Canker"
})

# Initialize metrics logger and evaluator
if LOGGING:
    print("Logging enabled: Semantic mapping performance will be evaluated.")
    logger = MetricsLogger(output_dir="output", experiment_name=EXPERIMENT_NAME)
    if os.path.exists(GROUND_TRUTH_FILE):
        print(f"Ground truth file found: {GROUND_TRUTH_FILE}")
        evaluator = SemanticMapDistanceEvaluator(
            semantic_map=semantic_octomap,
            ground_truth_file=GROUND_TRUTH_FILE,
            distance_threshold=EVAL_DISTANCE_THRESHOLD,
            min_confidence=CONFIDENCE_THRESHOLD
        )
    else:
        print(f"ERROR: Ground truth file not found: {GROUND_TRUTH_FILE}")
        LOGGING = False
else:
    print("Logging disabled.")

# Initial configuration of the robot to view the object
init_position = [0.5, 0, 1.25]
init_orientation = get_quaternion([0, np.pi/2, np.pi])

# ===== Generate Planar Grid of Viewpoints =====
print("\n=== Generating Planar Grid of Viewpoints ===")

# Get basis vectors from the initial orientation
from scipy.spatial.transform import Rotation as R
base_rot = R.from_quat(init_orientation)
R_base = base_rot.as_matrix()

# Extract basis vectors (X-axis right, Y-axis up, Z-axis forward)
plane_x = R_base[:, 0]
plane_y = R_base[:, 1]
normal = R_base[:, 2]

# Generate grid of positions on the plane
planar_viewpoints = []
x_range = np.arange(-PLANE_HALF_EXTENT, PLANE_HALF_EXTENT + SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)
y_range = np.arange(-PLANE_HALF_EXTENT, PLANE_HALF_EXTENT + SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)

for x in x_range:
    for y in y_range:
        # Position on plane
        pos = init_position + x * plane_x + y * plane_y
        
        # Create viewpoint with same orientation as initial
        viewpoint = Viewpoint(
            position=pos,
            orientation=init_orientation,
            target=pos + normal,  # Look-at point is along normal direction
            information_gain=0.0,
            cost=0.0,
            utility=0.0
        )
        planar_viewpoints.append(viewpoint)

print(f"Generated {len(planar_viewpoints)} planar viewpoints")

# Filter viewpoints by robot workspace and collision
filtered_viewpoints = []
for vp in planar_viewpoints:
    if manip_workspace.is_reachable(vp.position):
        # Check if we can compute IK for this viewpoint
        joint_angles = compute_viewpoint_joint_angles(robot, vp, camera)
        if joint_angles is not None:
            # Check collision
            in_collision = robot_in_collision(robot, joint_angles, obstacles)
            if not in_collision:
                filtered_viewpoints.append(vp)

print(f"Filtered to {len(filtered_viewpoints)} reachable viewpoints")

# Limit to MAX_ITERATIONS viewpoints
if len(filtered_viewpoints) > MAX_ITERATIONS:
    # Sample evenly across the filtered viewpoints
    indices = np.linspace(0, len(filtered_viewpoints) - 1, MAX_ITERATIONS, dtype=int)
    filtered_viewpoints = [filtered_viewpoints[i] for i in indices]
    print(f"Sampled {len(filtered_viewpoints)} viewpoints for {MAX_ITERATIONS} iterations")

# Move to initial position
print("\n=== Moving to Initial Position ===")
joint_angles = robot.ik(robot.end_effector, target_pos=init_position, 
    target_orient=init_orientation, use_current_joint_angles=True)
robot.control(joint_angles, set_instantly=True)

# Start the planar NBV planning loop
print("\n=== Starting Planar NBV Planning ===")
# input("Press Enter to begin...")

debug_id_handles = []
for iteration in range(min(len(filtered_viewpoints), MAX_ITERATIONS)):
    clear_debug_items(debug_id_handles)
    debug_id_handles = []
    print(f"\n=== Iteration {iteration + 1}/{MAX_ITERATIONS} ===")

    # Get the current image from the camera
    img, depth, segmentation_mask = camera.get_rgba_depth(flash=True, flash_intensity=2.0, shutter_speed=0.1, max_flash_distance=1.0)
    img_rgb = img[:, :, :3]

    # Detect using RGB image
    detections, annotated_img = detector.detect(img_rgb, visualize=True)
    print(f"Total detections: {len(detections)}")
    
    # Capture point cloud
    points, rgba, valid_mask = camera.get_point_cloud(max_range=MAX_RANGE, pixel_skip=1)
    print(f"Captured {len(points)} total points and {np.sum(valid_mask)} valid points from camera")

    # Integrate into octomap
    if len(points) > 0:
        labels, confidences = SemanticOctoMap.create_semantic_point_cloud_from_detections(
            rgb_image=img_rgb,
            detections=detections,
            background_label=-1,
            background_confidence=CONFIDENCE_THRESHOLD
        )
        stats = semantic_octomap.add_semantic_point_cloud(
            point_cloud=points[valid_mask],
            labels=labels[valid_mask],
            confidences=confidences[valid_mask],
            sensor_origin=camera.camera_pos,
            mismatch_penalty=MISMATCH_PENALTY,
            confidence_boost=CONFIDENCE_BOOST
        )
        handles = semantic_octomap.visualize_semantic(
            min_confidence=CONFIDENCE_THRESHOLD,
            colors=[[0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 1]],  # Gray, Red, Blue
            point_size=5.0,
            max_points=50000,
            visualize_free=True
        )
        debug_id_handles.extend(handles)

    # Logging and evaluation
    if LOGGING:
        # Compute coverage statistics
        coverage_stats = semantic_octomap.compute_coverage()
        print(f"Coverage: {coverage_stats['coverage_percent']:.2f}% ({coverage_stats['known_voxels']}/{coverage_stats['total_voxels']} voxels)")
        
        # Evaluate semantic map quality
        eval_results = evaluator.evaluate(verbose=False)
        
        # Log iteration metrics
        logger.log_iteration(
            iteration=iteration + 1,
            # Coverage metrics
            total_voxels=coverage_stats['total_voxels'],
            known_voxels=coverage_stats['known_voxels'],
            occupied_voxels=coverage_stats['occupied_voxels'],
            free_voxels=coverage_stats['free_voxels'],
            unknown_voxels=coverage_stats['unknown_voxels'],
            coverage_percent=coverage_stats['coverage_percent'],
            occupied_percent=coverage_stats['occupied_percent'],
            # Semantic evaluation metrics
            total_predictions=eval_results.get('total_predictions', None),
            true_positives=eval_results.get('num_TP', None),
            false_positives=eval_results.get('num_FP', None),
            false_negatives=eval_results.get('num_FN', None),
            hit_rate=eval_results.get('hit_rate', None),
            tp_avg_distance=eval_results.get('TP_avg_distance', None),
            fp_avg_distance=eval_results.get('FP_avg_distance', None),
            tp_avg_confidence=eval_results.get('TP_avg_confidence', None),
            fp_avg_confidence=eval_results.get('FP_avg_confidence', None),
            tp_max_confidence=eval_results.get('TP_max_confidence', None),
            fp_max_confidence=eval_results.get('FP_max_confidence', None),
        )
        
        # Update plots and save data
        logger.save_data()
        logger.plot_metrics(save=True, show=False)
    
    # Move to next viewpoint in the sequence (if not last iteration)
    if iteration < len(filtered_viewpoints) - 1:
        next_vp = filtered_viewpoints[iteration + 1]
        print(f"\nMoving to viewpoint {iteration + 2}/{len(filtered_viewpoints)} at position {next_vp.position}")
        
        # Compute joint angles for next viewpoint
        joint_angles = compute_viewpoint_joint_angles(robot, next_vp, camera)
        if joint_angles is not None:
            robot.control(joint_angles, set_instantly=True)
        else:
            print(f"WARNING: Could not compute IK for viewpoint {iteration + 2}")

print("\nPlanar NBV planning demo complete.")

# Save final octomap and generate summary
if LOGGING:
    # Save final octomap to output directory
    octomap_points_path = os.path.join(logger.data_dir, "planar_octomap_points.npz")
    octomap_labels_path = os.path.join(logger.data_dir, "planar_octomap_labels.npz")
    semantic_octomap.save_semantic(octomap_points_path, octomap_labels_path)
    
    # Generate final summary and plots
    logger.print_summary()
    logger.plot_combined_metrics({
        'Coverage Metrics': ['known_voxels', 'unknown_voxels'],
        'Detection Quality': ['true_positives', 'false_positives', 'false_negatives'],
    }, save=True, show=False)

# Keep running for visualization
print("Press Ctrl+C to exit")
try:
    while True:
        env.step_simulation(steps=1, realtime=True)
except KeyboardInterrupt:
    print("\nExiting...")

env.disconnect()
