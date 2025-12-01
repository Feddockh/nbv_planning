"""
This code demonstrates the semantic octomap implementation.

This is demonstrated by the following steps:
1. Capture image and point cloud from the robot camera
2. Run object detection on the image to get semantic labels
3. Create a semantic point cloud by associating labels with points
4. Add the semantic point cloud to the octomap
5. Visualize the semantic map with different colors for each class
6. Show uncertain regions that need more observation
"""

import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

import env
from utils import get_quaternion
from scene.roi import RectangleROI
from scene.objects import URDF, Ground, load_object, DebugCoordinateFrame, clear_debug_items
from bodies.panda import Panda
from scene.scene_representation import SemanticOctoMap
from vision import RobotCamera
from detection.fire_blight_detector import FireBlightDetector


def create_semantic_point_cloud(rgb_image, point_cloud, detections, background_label=-1, background_confidence=0.5):
    """
    Create a semantic point cloud by associating detection labels with 3D points.
    
    Args:
        rgb_image: RGB image (H, W, 3)
        point_cloud: Point cloud (N, 3) corresponding to image pixels
        detections: List of detection dicts from FireBlightDetector
        background_label: Label for points not in any detection box
        background_confidence: Confidence for points not in any detection box
    
    Returns:
        labels: Array of semantic labels (N,)
        confidences: Array of confidence scores (N,)
    """
    height, width = rgb_image.shape[:2]
    
    # Initialize all points with default label and confidence
    labels = np.full(len(point_cloud), background_label, dtype=np.int32)
    confidences = np.full(len(point_cloud), background_confidence, dtype=np.float32)
    
    # Create a pixel-to-point mapping
    # Assuming point cloud is ordered row-major (same as image)
    if len(point_cloud) == height * width:
        # For each detection, assign its label to points in the bounding box
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            
            # Create mask for bounding box using vectorized operations
            # Create grid of v, u coordinates
            v_indices, u_indices = np.meshgrid(np.arange(y1, y2+1), np.arange(x1, x2+1), indexing='ij')
            
            # Flatten and compute linear indices
            bbox_indices = v_indices.ravel() * width + u_indices.ravel()
            
            # Filter valid indices and assign labels
            valid = bbox_indices < len(labels)
            labels[bbox_indices[valid]] = det['class_id']
            confidences[bbox_indices[valid]] = det['confidence']
    
    return labels, confidences


def main():
    # Initialize environment
    print("=== Initializing NBV Environment ===")
    nbv_env = env.Env()
    ground = Ground(filename=os.path.join(nbv_env.asset_dir, 'dirt_plane', 'dirt_plane.urdf'))
    
    # Create table and object
    table = URDF(filename=os.path.join(nbv_env.asset_dir, 'table', 'table.urdf'), 
                   static=True, position=[1.5, 0, 0], orientation=[0, 0, 0, 1])
    
    # Load object - apple tree with disease
    print("Loading apple tree object...")
    obj = load_object("apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8])
    
    # Get the bounding box for ROI
    b_min, b_max = obj.get_AABB()
    size = np.array(b_max) - np.array(b_min)
    half_size = size / 2.0
    center = (np.array(b_max) + np.array(b_min)) / 2.0
    
    # Define ROI around the object
    obj_roi = RectangleROI(center=center, half_extents=half_size)
    roi_handles = obj_roi.visualize(lines_rgb=[0, 0, 1])
    
    # Let objects settle
    env.step_simulation(steps=100, realtime=True)
    
    # Create robot
    print("Creating robot...")
    robot = Panda(position=[1.0, 0, 0.76], fixed_base=True)
    
    # Create camera attached to robot end effector
    camera_offset_pos = np.array([0, 0, 0.01])
    camera_offset_orient = get_quaternion([0, 0, -np.pi/2])
    camera = RobotCamera(robot, robot.end_effector,
                         camera_offset_pos=camera_offset_pos,
                         camera_offset_orient=camera_offset_orient,
                         fov=60, camera_width=640, camera_height=480)
    
    # Initialize semantic octomap
    RESOLUTION = 0.01  # 1cm resolution (higher than paper's 3mm but good for demo)
    print(f"Initializing Semantic OctoMap with resolution: {RESOLUTION}m")
    semantic_map = SemanticOctoMap(bounds=obj_roi, resolution=RESOLUTION)
    
    # Set class names for visualization
    semantic_map.set_class_names({
       -1: "Background",
        0: "Shepherd's Crook",
        1: "Canker"
    })
    
    # Initialize fire blight detector (optional - will use dummy detections if not available)
    print("Loading Fire Blight detector...")
    detector = FireBlightDetector(confidence_threshold=0.3)
    
    # Determine initial configuration with position slightly above the base and orientation facing -x axis
    pos = [0.5, 0, 1.25]
    orientation = get_quaternion([0, np.pi / 2, np.pi])
    joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orientation,
                                    use_current_joint_angles=True)
    robot.control(joint_angles, set_instantly=True)
    camera.update_camera_pose()
    
    # Get the camera pose
    init_camera_pos, init_camera_orient = camera.get_camera_pose(update=False)

    # Capture the RGB image and point cloud from the initial viewpoint
    rgb_img, depth, segmentation_mask = camera.get_rgba_depth(flash=True, flash_intensity=2.5, shutter_speed=0.1, max_flash_distance=1.5)
    rgb_img = rgb_img[:, :, :3]  # Remove alpha channel
    points, rgba, valid_mask = camera.get_point_cloud(max_range=0.3, pixel_skip=1)
    
    # Run fire blight detection
    print("Running fire blight detection...")
    detections, annotated_img = detector.detect(rgb_img, visualize=True)
    print(f"\nDetection Results:")
    print(f"  Total detections: {len(detections)}")
    plt.imshow(annotated_img)
    plt.show()

    # Create semantic point cloud
    labels, confidences = create_semantic_point_cloud(rgb_img, points, detections)

    # Convert back to an image and overlay semantic labels for visualization
    # Reshape labels to image dimensions
    labels_img = labels.reshape((rgb_img.shape[0], rgb_img.shape[1]))
    
    # Create overlay image
    overlay_img = rgb_img.copy()
    
    # Create a blue overlay where labels are not background (-1)
    mask = labels_img != -1
    overlay_img[mask] = (overlay_img[mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
    
    # Draw bounding boxes on the overlay
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(overlay_img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original RGB Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay_img)
    plt.title("Semantic Overlay (Blue = Detected Objects)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Add to semantic octomap
    print("Adding semantic point cloud to octomap...")
    stats = semantic_map.add_semantic_point_cloud(
        point_cloud=points,
        labels=labels,
        confidences=confidences,
        sensor_origin=init_camera_pos,
        max_range=0.3,
        mismatch_penalty=0.1  # 10% penalty as described in paper
    )
    print(f"Update stats: {stats}")

    # Visualize the semantic map
    print("Visualizing semantic octomap...")
    debug_handles = []
    handles = semantic_map.visualize_semantic(
        label=None,  # Visualize all labels
        min_confidence=0.3,
        colors=[[0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 1]],  # Gray, Red, Blue for different classes
        point_size=5.0,
        max_points=50000
    )
    debug_handles.extend(handles)




    




    # # Capture and process from multiple viewpoints
    # for i, viewpoint in enumerate(viewpoints):
    #     print(f"--- Viewpoint {i+1}/{len(viewpoints)} ---")
        
    #     # Move robot to viewpoint
    #     pos = viewpoint['pos']
    #     orient = viewpoint['orient']
    #     joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orient,
    #                             max_iterations=100, use_current_joint_angles=True)
        
    #     if joint_angles is not None:
    #         robot.move_to_joint_angles(joint_angles)
    #         env.step_simulation(steps=50, realtime=True)
            
    #         # Get camera pose
    #         cam_pos, cam_orient = camera.get_camera_pose()
    #         print(f"Camera position: {cam_pos}")
            
    #         # Capture RGB image
    #         rgb_img, depth, seg_mask = camera.get_rgba_depth()
    #         rgb_img = rgb_img[:, :, :3]  # Remove alpha channel
            
    #         # Capture point cloud
    #         points, colors = camera.get_point_cloud(max_range=2.0)
    #         print(f"Captured {len(points)} points")
            
    #         if len(points) == 0:
    #             print("Warning: No points captured, skipping viewpoint")
    #             continue
            
    #         # Run semantic detection
    #         if use_detector and detector is not None:
    #             detections = detector.detect(rgb_img)
    #             print(f"Detected {len(detections)} objects")
    #             for det in detections:
    #                 print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    #         else:
    #             # Use dummy detections for demonstration
    #             # Simulate detecting fire blight in center region
    #             h, w = rgb_img.shape[:2]
    #             detections = [
    #                 {
    #                     'bbox': [w*0.3, h*0.3, w*0.7, h*0.7],
    #                     'class_id': 1,  # Fire blight
    #                     'class_name': 'fire_blight',
    #                     'confidence': 0.8
    #                 }
    #             ]
    #             print(f"Using dummy detection: {detections}")
            
    #         # Create semantic point cloud
    #         labels, confidences = create_semantic_point_cloud(
    #             rgb_img, points, detections, 
    #             default_label=0, default_confidence=0.5
    #         )
            
    #         # Add to semantic octomap
    #         stats = semantic_map.add_semantic_point_cloud(
    #             point_cloud=points,
    #             labels=labels,
    #             confidences=confidences,
    #             sensor_origin=cam_pos,
    #             max_range=2.0,
    #             mismatch_penalty=0.1  # 10% penalty as described in paper
    #         )
            
    #         print(f"Update stats: {stats}")
            
    #         # Small delay
    #         time.sleep(0.5)
    #     else:
    #         print(f"Warning: Could not reach viewpoint {i+1}")
    
    # # Update and print statistics
    # print("\n=== Final Semantic OctoMap Statistics ===")
    # semantic_map.update_stats(verbose=True, max_points=50000)
    
    # # Find frontiers
    # print("\n=== Finding Frontiers ===")
    # frontiers = semantic_map.find_frontiers(min_unknown_neighbors=1)
    # print(f"Found {len(frontiers)} frontier voxels")
    
    # # Cluster frontiers
    # if len(frontiers) > 0:
    #     print("\n=== Clustering Frontiers ===")
    #     cluster_result = semantic_map.cluster_frontiers(
    #         eps=RESOLUTION * 2,  # Adjacent voxels
    #         min_samples=5,
    #         algorithm='dbscan'
    #     )
    #     print(f"Found {cluster_result['n_clusters']} frontier clusters")
    #     print(f"Cluster sizes: {cluster_result['cluster_sizes']}")
    
    # # Visualization
    # print("\n=== Visualization ===")
    
    # # Clear previous debug items
    # clear_debug_items(debug_handles)
    # debug_handles = []
    
    # # Visualize the base occupancy map
    # print("Visualizing occupancy map (free=green, occupied=yellow)...")
    # handles = semantic_map.visualize(
    #     free_color=[0, 1, 0],
    #     occupied_color=[1, 1, 0],
    #     point_size=3.0,
    #     max_points=20000
    # )
    # debug_handles.extend(handles)
    
    # # Visualize semantic information
    # print("Visualizing semantic labels with distinct colors...")
    # handles = semantic_map.visualize_semantic(
    #     label=None,  # Visualize all labels
    #     min_confidence=0.3,
    #     point_size=5.0,
    #     max_points=20000
    # )
    # debug_handles.extend(handles)
    
    # # Visualize uncertain regions
    # print("Visualizing uncertain regions (orange)...")
    # handles = semantic_map.visualize_uncertainty(
    #     max_confidence=0.6,
    #     color=[1, 0.5, 0],
    #     point_size=6.0
    # )
    # debug_handles.extend(handles)
    
    # # Visualize frontiers
    # if len(frontiers) > 0:
    #     print("Visualizing frontier clusters...")
    #     handles = semantic_map.visualize_frontier_clusters(
    #         cluster_result=cluster_result,
    #         point_size=7.0,
    #         show_noise=True
    #     )
    #     debug_handles.extend(handles)
    
    # print("\n=== Demo Complete ===")
    # print("The semantic map shows:")
    # print("  - Green: Free space")
    # print("  - Yellow: Occupied space")
    # print("  - Various colors: Different semantic classes")
    # print("  - Orange: Uncertain regions (low confidence)")
    # print("  - Clustered colors: Frontier regions (unexplored boundaries)")
    # print("\nPress Ctrl+C to exit")
    
    # Keep simulation running
    try:
        while True:
            env.step_simulation(steps=1, realtime=True)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()



