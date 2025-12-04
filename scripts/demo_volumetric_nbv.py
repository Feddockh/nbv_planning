"""
Next-Best-View Planning Demo

Simple demonstration of active perception using octomap for volumetric mapping.
The robot explores the scene by moving its camera to maximize information gain.
"""

import sys
import os
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import heapq
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
from viewpoints.viewpoint_proposal import generate_planar_spherical_cap_candidates, sample_views_from_hemisphere
from viewpoints.viewpoint import Viewpoint, visualize_viewpoint, compute_viewpoint_joint_angles
from viewpoints.viewpoint_selection import compute_information_gain
from bodies.planning import MotionPlanner
from detection.fire_blight_detector import FireBlightDetector


# ===== Configuration =====
LEARN_WORKSPACE = False
OCTOMAP_RESOLUTION = 0.04
NUM_SAMPLES_PER_FRONTIER = 5
MAX_ITERATIONS = 30
CAMERA_WIDTH = 1440
CAMERA_HEIGHT = 1080
CAMERA_FOV = 60
NUM_RAYS = 50
MAX_RANGE = 0.5
MIN_RANGE = 0.05
MISMATCH_PENALTY = 0.1
CONFIDENCE_BOOST = 0.05
ALPHA = 0.1 # Cost weight for utility calculation
MIN_INFORMATION_GAIN = 0.1 # Minimum information gain to continue planning
CONFIDENCE_THRESHOLD = 0.1

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
detector = FireBlightDetector(model_path=os.path.join("detection", "models", "best_sim.pt"), confidence_threshold=CONFIDENCE_THRESHOLD)

# Initialize octomap
semantic_octomap = SemanticOctoMap(bounds=obj_roi, resolution=OCTOMAP_RESOLUTION)
semantic_octomap.set_class_names({
    -1: "Background",
    0: "Shepherd's Crook",
    1: "Canker"
})

# Initial configuration of the robot to view the object
init_position = [0.5, 0, 1.25]
init_orientation = get_quaternion([0, np.pi/2, np.pi])
joint_angles = robot.ik(robot.end_effector, target_pos=init_position, 
    target_orient=init_orientation, use_current_joint_angles=True)
robot.control(joint_angles, set_instantly=True)

# Visualize the camera frame
camera_pos, camera_orient = camera.get_camera_pose()

# Start the NBV planning loop
print("\n=== Starting NBV Planning ===")
input("Press Enter to begin...")

# NBV planning iterations
best_information_gain = np.inf
debug_id_handles = []
for iteration in range(MAX_ITERATIONS):
    clear_debug_items(debug_id_handles)
    debug_id_handles = []
    print(f"\n=== Iteration {iteration + 1}/{MAX_ITERATIONS} ===")

    # Get the current image from the "camera" and display it
    img, depth, segmentation_mask = camera.get_rgba_depth(flash=True, flash_intensity=2.0, shutter_speed=0.1, max_flash_distance=1.0)
    img_rgb = img[:, :, :3]
    
    detections, annotated_img = detector.detect(img_rgb, visualize=True)
    print(f"Total detections: {len(detections)}")
    # plt.imshow(annotated_img)
    # plt.show()
    
    # Capture point cloud
    points, rgba, valid_mask = camera.get_point_cloud(max_range=MAX_RANGE, pixel_skip=1)
    print(f"Captured {len(points)} total points and {np.sum(valid_mask)} valid points from camera")

    # # Display the point cloud and hide the object
    # if np.sum(valid_mask) > 0:
    #     valid_points_debug_marker_ids = DebugPoints(points[valid_mask], points_rgb=rgba[valid_mask, :3], size=5)
    #     obj.change_visual(link=obj.base, rgba=[1, 1, 1, 0.25])

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
            colors=[[0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 1]],  # Gray, Red, Blue for different classes
            point_size=5.0,
            max_points=50000,
            visualize_free=True
        )
        debug_id_handles.extend(handles)

    # Compute the viewpoints for the next best view
    frontiers = semantic_octomap.find_frontiers(min_unknown_neighbors=1)
    print(f"Found {len(frontiers)} frontiers")
    # frontiers_debug_ids = semantic_octomap.visualize_frontiers(frontiers, point_size=10.0)
    # Cluster the frontiers
    clustered_frontiers = semantic_octomap.cluster_frontiers(frontiers, algorithm='kmeans', min_samples=3, eps=OCTOMAP_RESOLUTION,
                                                             n_clusters=max(1, len(frontiers) // 40 if len(frontiers) // 40 <= 40 else 40))
    print(f"Clustered frontiers into {len(clustered_frontiers['cluster_centers'])} groups")
    # clustered_frontiers_debug_ids = semantic_octomap.visualize_frontier_clusters(clustered_frontiers, point_size=10)
    # Filter out the clusters that are too far to reach
    reachable_cluster_centers = []
    for cluster_center in clustered_frontiers["cluster_centers"]:
        if manip_workspace.min_distance_to_workspace(cluster_center) <= MAX_RANGE:
            reachable_cluster_centers.append(cluster_center)
    print(f"Reachable cluster centers: {len(reachable_cluster_centers)}/{len(clustered_frontiers['cluster_centers'])}")
    # Create a partial Fibboni sphere around each cluster centroid and sample viewpoints
    viewpoint_candidates: List[Viewpoint] = []
    for cluster_center in reachable_cluster_centers:
        new_viewpoint_candidates = sample_views_from_hemisphere(
            center=cluster_center,
            base_orientation=init_orientation,
            min_radius=MIN_RANGE,
            max_radius=MAX_RANGE,
            num_samples=NUM_SAMPLES_PER_FRONTIER,
            use_positive_z=False,
            z_bias_sigma=np.pi/3,
            min_distance=0.05,
            max_attempts=1000
        )
        viewpoint_candidates.extend(new_viewpoint_candidates)
    print(f"Generated {len(viewpoint_candidates)} viewpoint candidates from frontier clusters")
    # Get viewpoints from planar spherical cap sampling too
    viewpoint_candidates += generate_planar_spherical_cap_candidates(
        position=init_position,
        orientation=init_orientation,
        half_extent=0.4,
        spatial_resolution=0.2,
        max_theta=50.0,
        angular_resolution=30.0
    )
    print(f"Total viewpoint candidates after adding planar spherical cap: {len(viewpoint_candidates)}")
    # Filter viewpoints by robot workspace
    filtered_viewpoint_candidates: List[Viewpoint] = []
    for vp in viewpoint_candidates:
        if manip_workspace.is_reachable(vp.position) and not semantic_octomap.is_occupied(vp.position):
            filtered_viewpoint_candidates.append(vp)
            # visualize_viewpoint(vp)
    print(f"Filtered to {len(filtered_viewpoint_candidates)} / {len(viewpoint_candidates)} viewpoints")
    # input("Press Enter to compute information gain for viewpoints...")

    # Compute information gain for each viewpoint
    for vp in filtered_viewpoint_candidates:
        vp.information_gain = compute_information_gain(
            viewpoint=vp, 
            scene_representation=semantic_octomap, 
            fov=CAMERA_FOV, 
            width=CAMERA_WIDTH, 
            height=CAMERA_HEIGHT, 
            max_range=MAX_RANGE, 
            resolution=OCTOMAP_RESOLUTION, 
            num_rays=NUM_RAYS, 
            roi=obj_roi
        )
           
    # Compute the utility for each viewpoint
    camera_pos, _ = camera.get_camera_pose()
    for vp in filtered_viewpoint_candidates:
        distance_cost = np.linalg.norm(np.array(vp.position) - np.array(camera_pos))
        vp.cost = distance_cost
        vp.utility = vp.information_gain - ALPHA * vp.cost

    # Select the best viewpoint
    heap = filtered_viewpoint_candidates.copy()
    heapq.heapify(heap)
    print(f"Created a heap with {len(heap)} viewpoints ordered by utility")

    # Get the top viewpoint
    print("Evaluating viewpoints for NBV:")
    while heap:
        # Pop the best viewpoint from the heap
        best_vp = heapq.heappop(heap)
        print(f"  Best viewpoint at {best_vp.position} with utility = {best_vp.utility:.4f}, IG = {best_vp.information_gain:.4f}, cost = {best_vp.cost:.4f}")
        # Update the best information gain seen so far
        if best_vp.information_gain < best_information_gain:
            best_information_gain = best_vp.information_gain
        # Check if information gain is too low
        if best_information_gain < MIN_INFORMATION_GAIN:
            print(f"  Information gain too low ({best_information_gain:.4f} < {MIN_INFORMATION_GAIN}), stopping exploration")
            print("\nNBV planning complete (scene sufficiently explored)")
            break
        # Try to compute the IK for the best viewpoint
        best_joint_angles = compute_viewpoint_joint_angles(robot, best_vp, camera)
        if best_joint_angles is not None:
            in_collision = robot_in_collision(robot, best_joint_angles, obstacles)
            if in_collision:
                print(f"  Best viewpoint is in collision, trying next...")
                continue
            print(f"  Computed IK for best viewpoint")
            break
        else:
            print(f"  Could not compute IK for best viewpoint, trying next...")
    # Break the main loop if no viewpoint found with sufficient information gain
    if best_information_gain < MIN_INFORMATION_GAIN or best_joint_angles is None:
        break
    # Move the robot to the best viewpoint if there is one
    print(f"\nMoving to best joint angles: {best_joint_angles}")
    robot.control(best_joint_angles, set_instantly=True)
    # visualize_viewpoint(best_vp)
    # input("Press Enter to continue to next iteration...")    

print("\nNBV planning demo complete.")
semantic_octomap.save_semantic("volumetric_octomap_points_sim.npz", "volumetric_octomap_labels_sim.npz")

# Keep running for visualization
print("Press Ctrl+C to exit")
try:
    while True:
        env.step_simulation(steps=1, realtime=True)
except KeyboardInterrupt:
    print("\nExiting...")

env.disconnect()
