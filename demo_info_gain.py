import os
import numpy as np

import env
from utils import get_quaternion
from scene.roi import RectangleROI
from scene.objects import URDF, Ground, load_object, DebugCoordinateFrame, DebugPoints
from bodies.panda import Panda
from scene.scene_representation import OctoMap
from vision import RobotCamera
from viewpoints.viewpoint import Viewpoint, visualize_viewpoint


nbv_env = env.Env()
ground = Ground(filename=os.path.join(nbv_env.asset_dir, 'dirt_plane', 'dirt_plane.urdf'))
# visualize_coordinate_frame()

# Create table and object
table = URDF(filename=os.path.join(nbv_env.asset_dir, 'table', 'table.urdf'), 
               static=True, position=[1.5, 0, 0], orientation=[0, 0, 0, 1])

# Load object - choose one:
# Object options: "apple_tree", "cheezit", "spam", "mustard", "tomato_soup_can", "mug"
obj = load_object("apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8])
obstacles = [obj, table]

# Get the bounding box
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

# Create camera
camera_offset_pos = np.array([0, 0, 0.01])
camera_offset_orient = get_quaternion([0, 0, -np.pi/2])
camera = RobotCamera(robot, robot.end_effector,
                     camera_offset_pos=camera_offset_pos,
                     camera_offset_orient=camera_offset_orient,
                     fov=60, camera_width=1440, camera_height=1080)

# Initialize octomap
octomap = OctoMap(bounds=obj_roi, resolution=0.03)

# Determine initial configuration with position slightly above the base and orientation facing -x axis
pos = [0.75, 0, 1.25]
orientation = get_quaternion([0, np.pi/2, np.pi])
joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orientation,
                                use_current_joint_angles=True)
robot.control(joint_angles, set_instantly=True)

# Visualize the camera frame
init_camera_pos, init_camera_orient = camera.get_camera_pose()

# Compute the current viewpoint
init_viewpoint = Viewpoint(position=init_camera_pos, orientation=init_camera_orient)
visualize_viewpoint(init_viewpoint, coordinate_frame=True)

# Compute the rotation matrix
x, y, z, w = init_viewpoint.orientation
xx, yy, zz = x*x, y*y, z*z
xy, xz, yz = x*y, x*z, y*z
wx, wy, wz = w*x, w*y, w*z
R_wc = np.array([
    [1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy)],
    [2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx)],
    [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy)]
], dtype=np.float64)

# Information gain parameters
hfov = np.deg2rad(float(60))
focal = (1440 / 2.0) / np.tan(hfov / 2.0)
cx = (1440  - 1) * 0.5
cy = (1080 - 1) * 0.5

# Octomap access
max_depth = int(octomap.octree.getTreeDepth())

# March across all pixels with a stride of 200
total_unknown = 0
num_rays_cast = 0
for v in range(0, 1080, 200):
    for u in range(0, 1440, 200):
        # Ray in camera frame (z forward, x right, y down in image coords)
        x = (u - cx) / focal
        y = (v - cy) / focal
        ray_cam = np.array([x, y, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)

        # Transform to world frame
        ray_world = R_wc @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        unknown_count = 0

        # March along the ray
        prev_point_in_roi = False
        for i in range(0, int(2.0 / 0.03)):
            t = i * 0.03
            point = init_viewpoint.position + t * ray_world

            # End ray if the current point is outside the ROI and the previous point was inside (single convex ROI)
            if not obj_roi.contains(point) and prev_point_in_roi:
                break
            prev_point_in_roi = obj_roi.contains(point)

            # ROI gating: skip points outside ROI
            if not obj_roi.contains(point):
                DebugPoints([point], [[0.2, 0.2, 0.2]], 5.0)
                continue
            else:
                DebugPoints([point], [[1.0, 0, 0]], 5.0)

            # Check voxel state
            node = octomap.octree.search(point, depth=max_depth)
            if node is None:
                # Unknown voxel - increment counter
                unknown_count += 1
            else:
                # Known voxel: if occupied, stop the ray
                if octomap.octree.isNodeOccupied(node):
                    break
                # If free, continue marching

        total_unknown += unknown_count
        num_rays_cast += 1
        print(f"Ray ({u}, {v}): {unknown_count} unknown voxels")

# Return average unknown voxels per ray
ig = float(total_unknown / num_rays_cast) if num_rays_cast > 0 else 0.0
print(f"Information Gain from initial viewpoint: {ig:.2f} unknown voxels per ray")





# Keep running for visualization
print("Press Ctrl+C to exit")
try:
    while True:
        env.step_simulation(steps=1, realtime=True)
except KeyboardInterrupt:
    print("\nExiting...")

env.disconnect()