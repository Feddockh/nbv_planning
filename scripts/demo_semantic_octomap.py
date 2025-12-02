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
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from viewpoints.viewpoint import Viewpoint
from utils import get_quaternion
from scene.roi import RectangleROI
from scene.objects import URDF, Ground, load_object, DebugCoordinateFrame, clear_debug_items
from bodies.panda import Panda
from scene.scene_representation import SemanticOctoMap
from vision import RobotCamera
from detection.fire_blight_detector import FireBlightDetector


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
    detector = FireBlightDetector(confidence_threshold=0.1)

    # Create the views to capture
    views = [
        Viewpoint(position=[0.5, 0, 1.25], orientation=get_quaternion([0, np.pi/2, np.pi])),
        Viewpoint(position=[0.6, 0, 1.3], orientation=get_quaternion([0, np.pi/2, np.pi])),
    ]

    for view in views:
        pos = view.position
        orientation = view.orientation
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
        for det in detections:
            print(f"    - {det['class_name']}: {det['confidence']:.2f}")
        plt.imshow(annotated_img)
        plt.show()

        # Create semantic point cloud using the SemanticOctoMap method
        labels, confidences = SemanticOctoMap.create_semantic_point_cloud_from_detections(
            rgb_image=rgb_img, 
            point_cloud=points, 
            detections=detections,
            background_label=-1,
            background_confidence=0.5
        )

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


        # Go to next position
        input("Press Enter to continue to next viewpoint...")
        clear_debug_items(debug_handles)

    # Keep simulation running
    try:
        while True:
            env.step_simulation(steps=1, realtime=True)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()



