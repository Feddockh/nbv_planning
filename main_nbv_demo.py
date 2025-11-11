"""
Next-Best-View Planning Demo

Simple demonstration of active perception using octomap for volumetric mapping.
The robot explores the scene by moving its camera to maximize information gain.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

from vision import RobotCamera
import env
from bodies.robot import ManipulationWorkspace, moveto, robot_in_collision
from bodies.panda import Panda
from utils import get_quaternion, quat_to_normal
from scene.roi import SphereROI, RectangleROI, ROI
from scene.scene_representation import OctoMap
from scene.objects import visualize_coordinate_frame, clear_debug_items, DebugPoints, Ground, URDF, load_object, DebugCoordinateFrame
from viewpoints.viewpoint_proposal import generate_planar_spherical_candidates
from viewpoints.viewpoint import Viewpoint, visualize_viewpoint, compute_viewpoint_joint_angles
from viewpoints.viewpoint_selection import compute_information_gain


# ===== Configuration =====
LEARN_WORKSPACE = False  # Whether to learn the robot workspace or load a known one
OCTOMAP_RESOLUTION = 0.04  # 2cm voxels
NUM_SAMPLES_PER_FRONTIER = 10  # Viewpoints to sample per frontier
MAX_ITERATIONS = 3  # Maximum NBV iterations
CAMERA_WIDTH = 1440
CAMERA_HEIGHT = 1080
CAMERA_FOV = 60
NUM_RAYS = 20
MAX_RANGE = 2.0  # Max sensor range in meters
ALPHA = 0.1  # Cost weight for utility computation
MIN_INFORMATION_GAIN = 0.1  # Minimum IG to continue exploration

# ===== Main Script =====
# Create environment and ground
nbv_env = env.Env()
ground = Ground()
visualize_coordinate_frame()

# Create table and object
table = URDF(filename=os.path.join(nbv_env.asset_dir, 'table', 'table.urdf'), 
               static=True, position=[1.5, 0, 0], orientation=[0, 0, 0, 1])

# Load object - choose one:
# Object options: "apple_tree", "cheezit", "spam", "mustard", "tomato_soup_can", "mug"
obj = load_object("apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8])
obstacles = [obj, table]

# Get the bounding box of the mustard bottle
b_min, b_max = obj.get_AABB()
size = np.array(b_max) - np.array(b_min)
half_size = size / 2.0
center = (np.array(b_max) + np.array(b_min)) / 2.0

# Re-define ROI around the object
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
    manip_workspace.learn(num_samples=100000)
    print("Saving workspace...")
    manip_workspace.save("workspace.npz")
else:
    print("Loading known workspace...")
    manip_workspace.load("workspace.npz")

# Create camera
camera_offset_pos = np.array([0, 0, 0.01])
camera_offset_orient = get_quaternion([0, 0, 0])
camera = RobotCamera(robot, robot.end_effector,
                     camera_offset_pos=camera_offset_pos,
                     camera_offset_orient=camera_offset_orient,
                     fov=CAMERA_FOV, camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT)

# Create detector
from detection.fire_blight_detector import FireBlightDetector
detector = FireBlightDetector(confidence_threshold=0.1)

# Initialize octomap
octomap = OctoMap(bounds=obj_roi, resolution=OCTOMAP_RESOLUTION)

# Determine initial configuration with position slightly above the base and orientation facing -x axis
pos = [0.75, 0, 1.25]
orientation = get_quaternion([0, np.pi / 2, np.pi])
joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orientation,
                                use_current_joint_angles=True)
robot.control(joint_angles, set_instantly=True)

# Visualize the camera frame
init_camera_pos, init_camera_orient = camera.get_camera_pose()
# visualize_coordinate_frame(init_camera_pos, init_camera_orient)

# input("Press Enter to start NBV planning...")

# Start the NBV planning loop
print("\n=== Starting NBV Planning ===")

# NBV planning iterations
best_information_gain = np.inf
for iteration in range(MAX_ITERATIONS):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration + 1}/{MAX_ITERATIONS}")
    print(f"{'='*60}")

    # Get the current image from the "camera" and display it
    img, depth, segmentation_mask = camera.get_rgba_depth(flash=True, flash_intensity=2.5, shutter_speed=0.1, max_flash_distance=1.5)
    # plt.imshow(img)
    # plt.show()

    # Convert RGBA to RGB for YOLO (remove alpha channel)
    img_rgb = img[:, :, :3]
    
    # Run fire blight detection
    print("Running fire blight detection...")
    detections, annotated_img = detector.detect(img_rgb, visualize=True)

    # Print detection results
    print(f"\nDetection Results:")
    print(f"  Total detections: {len(detections)}")
    plt.imshow(annotated_img)
    plt.show()
    

    # Step 1: Capture point cloud
    print("Step 1: Capturing point cloud...")
    points, rgba, valid_mask = camera.get_point_cloud(max_range=MAX_RANGE, pixel_skip=1)
    print(f"  Captured {len(points)} total points and {np.sum(valid_mask)} valid points from camera")

    # Display the point cloud and hide the object
    if np.sum(valid_mask) > 0:
        valid_points_debug_marker_ids = DebugPoints(points[valid_mask], points_rgb=rgba[valid_mask, :3], size=5)
        obj.change_visual(link=obj.base, rgba=[1, 1, 1, 0.25])

    # # Wait for user to press Enter
    # input("Press Enter to continue...")

    # Step 2: Integrate into octomap
    print("Step 2: Integrating point cloud into octomap...")
    if len(points) > 0:
        origin = np.array(camera.camera_pos, dtype=np.float64)
        points_array = np.asarray(points, dtype=np.float64)
        # success_count = octree.insertPointCloudRaysFast(points_array, origin, lazy_eval=True) # This function doesn't filter by max range properly
        success_count = octomap.add_point_cloud(points_array, origin, max_range=MAX_RANGE, lazy_eval=False, discretize=True)
        # octree.updateInnerOccupancy()
        print(f"  Integrated {success_count}/{len(points)} points into octomap")

        # Visualize octomap update (show free spaces as green and occupied as yellow)
        print("  Visualizing octomap...")
        octomap.update_stats(verbose=True, max_points=100000)
        octomap_debug_ids = octomap.visualize(max_points=100000, point_size=5)

    # # Wait for user to press Enter
    # input("Press Enter to continue...")

    # Step 3: Find ROI frontiers
    print("\nStep 3: Finding ROI frontiers...")
    frontiers = octomap.find_frontiers(min_unknown_neighbors=1)
    print(f"  Found {len(frontiers)} frontiers") 

    # Visualize frontiers
    frontiers_debug_ids = octomap.visualize_frontiers(frontiers, point_size=10) # Have to make bigger to see past octomap points

    # # Wait for user to press Enter
    # input("Press Enter to continue...")

    # Step 4: Generate the viewpoint candidates
    print("\nStep 4: Generating viewpoint candidates...")

    viewpoint_candidates = generate_planar_spherical_candidates(
        position=init_camera_pos,
        orientation=init_camera_orient,
        half_extent=0.4,
        spatial_resolution=0.2,
        max_theta=50.0,
        angular_resolution=25.0
    )
    print(f"  Generated {len(viewpoint_candidates)} viewpoint candidates on plane above object")

    # # Visualize the viewpoints
    # viewpoints_debug_idxs = []
    # for vp in viewpoint_candidates:
    #     idx = visualize_viewpoint(vp)
    #     viewpoints_debug_idxs.append(idx)

    # Step 5: Filter viewpoints by robot workspace
    print("Step 5: Filtering viewpoints by robot workspace...")
    filtered_viewpoints: list[Viewpoint] = []
    for vp in viewpoint_candidates:
        if manip_workspace.is_reachable(vp.position):
            filtered_viewpoints.append(vp)

    print(f"    {len(filtered_viewpoints)} viewpoints remain after filtering by robot workspace")

    # Step 6: Compute information gain for each viewpoint
    print("\nStep 6: Computing information gain for each viewpoint...")
    for vp in filtered_viewpoints:
        vp.information_gain = compute_information_gain(vp, octomap, fov=CAMERA_FOV, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, max_range=MAX_RANGE, resolution=OCTOMAP_RESOLUTION, num_rays=NUM_RAYS, roi=obj_roi)
        # print(f"  Viewpoint at {vp.position} has IG = {vp.information_gain:.4f}")

    # Step 7: Compute the utility for each viewpoint
    print("\nStep 7: Computing viewpoint utilities...")
    for vp in filtered_viewpoints:
        distance_cost = np.linalg.norm(np.array(vp.position) - np.array(init_camera_pos))
        vp.cost = distance_cost
        vp.utility = vp.information_gain - ALPHA * vp.cost
        # print(f"  Viewpoint at {vp.position} has utility = {vp.utility:.4f}")

    # Step 8: Select the best viewpoint
    print("\nStep 8: Selecting the best viewpoint...")
    import heapq
    heap = filtered_viewpoints.copy()
    heapq.heapify(heap)
    print(f"  Created a heap with {len(heap)} viewpoints ordered by utility")

    # Get the top viewpoint
    
    while heap:
        best_vp = heapq.heappop(heap)
        print(f"  Best viewpoint at {best_vp.position} with utility = {best_vp.utility:.4f}, IG = {best_vp.information_gain:.4f}, cost = {best_vp.cost:.4f}")

        if best_vp.information_gain < best_information_gain:
            best_information_gain = best_vp.information_gain
        
        # Check if information gain is too low (scene mostly explored)
        if best_information_gain < MIN_INFORMATION_GAIN:
            print(f"  Information gain too low ({best_information_gain:.4f} < {MIN_INFORMATION_GAIN}), stopping exploration")
            print("\nNBV planning complete - scene sufficiently explored")
            break
        
        vp_debug_ids = visualize_viewpoint(best_vp)

        # Try to compute the IK for the best viewpoint
        best_joint_angles = compute_viewpoint_joint_angles(robot, best_vp, camera)
        in_collision = robot_in_collision(robot, best_joint_angles, obstacles)
        if best_joint_angles is not None and not in_collision:
            print(f"  Computed IK for best viewpoint")
            break
        else:
            print(f"  Could not compute IK for best viewpoint, trying next...")
    
    # Check if we found a valid viewpoint
    if best_joint_angles is None:
        print(f"  ERROR: No valid viewpoint with IK solution found")
        print("\nNBV planning stopped - no reachable viewpoints")
        break

    if best_information_gain < MIN_INFORMATION_GAIN:
        break

    print(f"\nMoving to best joint angles: {best_joint_angles}")
    moveto(robot, joint_angles=best_joint_angles, tolerance=0.1)

    # Clear the visualizations from this iteration
    print("Clearing debug visualizations...")
    if np.sum(valid_mask) > 0 > 0:
        clear_debug_items(valid_points_debug_marker_ids)
    if len(points) > 0:
        clear_debug_items(octomap_debug_ids)
        clear_debug_items(frontiers_debug_ids)
    clear_debug_items(vp_debug_ids)
    obj.change_visual(link=obj.base, rgba=[1, 1, 1, 1])

print("\nNBV planning demo complete.")

# Keep running for visualization
print("Press Ctrl+C to exit")
try:
    while True:
        env.step_simulation(steps=1, realtime=True)
except KeyboardInterrupt:
    print("\nExiting...")

env.disconnect()
