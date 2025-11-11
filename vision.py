import sys
import os
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

import env
from utils import multiply_transforms, get_quaternion


class Camera:
    def __init__(self, camera_pos=[0.5, -0.5, 1.5], look_at_pos=[0, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4, local_env=None):
        self.env = local_env if local_env is not None else env.envir
        self.fov = fov
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_pos, look_at_pos, [0, 0, 1], physicsClientId=self.env.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.camera_width / self.camera_height, 0.01, 100, physicsClientId=self.env.id)

    def set_camera_rpy(self, look_at_pos=[0, 0, 0.75], distance=1.5, rpy=[0, -35, 40]):
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(look_at_pos, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.env.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.camera_width / self.camera_height, 0.01, 100, physicsClientId=self.env.id)

    def __flash_image(self, img, depth, flash_intensity=1.5, shutter_speed=0.5, max_flash_distance=1.0):
        """
        Simulate flash camera effect with tunable parameters.
        
        Args:
            img: Input image array (H, W, 4)
            depth: Depth buffer (H, W) with values in [0, 1]
            flash_intensity: Multiplier for flash brightness on foreground (default 1.5)
                           - 1.0 = no flash effect
                           - >1.0 = brighter flash (try 1.5-3.0)
            shutter_speed: Multiplier for overall image brightness (default 0.5)
                          - Simulates faster shutter speed with flash
                          - 1.0 = no darkening
                          - <1.0 = darker background (try 0.3-0.7)
            max_flash_distance: Maximum distance for flash effect (default 1.0)
        
        Returns:
            Flash-processed image
        """
        # Linearize depth buffer (PyBullet uses non-linear depth)
        near = 0.01
        far = 20.0
        linear_depth = far * near / (far - depth * (far - near))
        
        # Normalize depth to [0, 1] range (0 = near, 1 = far)
        depth_normalized = np.clip(linear_depth / max_flash_distance, 0, 1)
        
        # Calculate flash falloff based on distance
        # Use exponential falloff so distant objects truly go to zero
        # e^(-k*depth) ensures falloff reaches near-zero for distant objects
        flash_falloff = np.exp(-6.0 * depth_normalized**2)
        
        # Apply flash effect:
        # 1. Base shutter speed effect (darkens everything)
        # 2. Add flash contribution based on distance
        flash_contribution = flash_intensity * flash_falloff
        brightness_factor = shutter_speed + flash_contribution
        
        flash_img = img.copy()
        flash_img[:, :, :3] = np.clip(img[:, :, :3] * brightness_factor[:, :, None], 0, 255).astype(np.uint8)
        
        return flash_img

    def get_rgba_depth(self, light_pos=[0, -3, 1], shadow=False, ambient=0.8, diffuse=0.3, specular=0.1, flash=False, flash_intensity=1.5, shutter_speed=0.5, max_flash_distance=1.0):
        w, h, img, depth, segmentation_mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, lightDirection=light_pos, shadow=shadow, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.env.id)
        img = np.reshape(img, (h, w, 4))
        depth = np.reshape(depth, (h, w))
        segmentation_mask = np.reshape(segmentation_mask, (h, w))
        if flash:
            img = self.__flash_image(img, depth, flash_intensity=flash_intensity, shutter_speed=shutter_speed, max_flash_distance=max_flash_distance)
        return img, depth, segmentation_mask

    def get_point_cloud(self, body=None):
        # get a depth image
        rgba, depth, segmentation_mask = self.get_rgba_depth()
        rgba = rgba.reshape((-1, 4))
        depth = depth.flatten()
        segmentation_mask = segmentation_mask.flatten()

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.camera_height, -1:1:2 / self.camera_width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)

        # Filter point cloud to only include points on the target body
        if body is not None:
            pixels = pixels[segmentation_mask == body.body]
            z = z[segmentation_mask == body.body]
            rgba = rgba[segmentation_mask == body.body]

        # filter out "infinite" depths
        pixels = pixels[z < 20]
        rgba = rgba[z < 20]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points, rgba/255

class RobotCamera(Camera):
    def __init__(self, robot, end_effector_link, 
                 camera_offset_pos=[0, 0, 0], camera_offset_orient=[0, 0, 0, 1],
                 fov=60, camera_width=640, camera_height=480):
        """
        Camera mounted on a robot's end effector.
        
        Args:
            robot: mengine Robot instance
            end_effector_link: link index of the end effector where the camera is mounted
            camera_offset_pos: position offset of the camera relative to the end effector
            camera_offset_orient: orientation offset (quaternion) of the camera relative to the end effector
            fov: field of view in degrees
            camera_width: image width in pixels
            camera_height: image height in pixels
        """
        self.robot = robot
        self.end_effector_link = end_effector_link
        self.camera_offset_pos = np.array(camera_offset_pos)
        self.camera_offset_orient = np.array(camera_offset_orient)
        
        # Initialize the base Camera with dummy position/orientation; will update later
        super().__init__(camera_pos=[0,0,0], look_at_pos=[0,0,1], 
                         fov=fov, camera_width=camera_width, camera_height=camera_height)
    
    def update_camera_pose(self):
        """
        Update the camera's global position and orientation based on the robot's end effector.
        
        This also updates the view matrix and projection matrix, which are required for
        correct point cloud generation in get_point_cloud().
        """
        # Get end effector pose
        ee_pos, ee_orient = self.robot.get_link_pos_orient(self.end_effector_link)
        
        # Transform camera offset from end effector frame to world frame
        # Using PyBullet's multiplyTransforms (available via mengine)
        cam_pos, cam_orient = multiply_transforms(ee_pos, ee_orient,
                                                     self.camera_offset_pos, 
                                                     self.camera_offset_orient)
        
        # Compute look-at position (camera looks along its local +Z axis in mengine convention)
        # Transform a point 1 meter along local +Z axis to get the look direction
        look_at_offset = [0, 0, 1]  # Local +Z direction
        look_at_pos, _ = multiply_transforms(cam_pos, cam_orient, 
                                               look_at_offset, [0, 0, 0, 1])
        
        # Update view matrix
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=look_at_pos,
            cameraUpVector=[0, 0, 1],  # Z-up convention
            physicsClientId=self.env.id
        )
        
        # Projection matrix doesn't change unless FOV or aspect ratio changes
        # but we can recompute it to be safe
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.01,
            farVal=100,
            physicsClientId=self.env.id
        )
        
        # Store these for reference (optional, but useful for debugging)
        self.camera_pos = np.array(cam_pos)
        self.look_at_pos = np.array(look_at_pos)
        self.camera_orient = np.array(cam_orient)

    def get_camera_pose(self):
        """
        Get the current camera pose (position and orientation).
        
        Returns:
            camera_pos: 3D position of the camera in world frame
            camera_orient: orientation of the camera as a quaternion [x, y, z, w]
        """
        self.update_camera_pose()
        return self.camera_pos, self.camera_orient

    def get_point_cloud(self, body=None, max_range=2.0, pixel_skip=1, **kwargs):
        """
        Get point cloud from the current camera pose.
        
        This automatically updates the camera pose before capturing the point cloud.
        
        Args:
            body: Optional body to filter points by
            max_range: Maximum depth range (meters)
            pixel_skip: Sample every Nth pixel (1=all pixels, 2=every other pixel, etc.)
                       Higher values = faster but less dense point cloud
            **kwargs: Additional arguments passed to get_rgba_depth()
            
        Returns:
            points: Nx3 array of 3D points
            rgba: Nx4 array of RGBA colors (0-1 range)
        """
        self.update_camera_pose()

        # get a depth image
        rgba, depth, segmentation_mask = self.get_rgba_depth(**kwargs)
        
        # Reshape to 2D for pixel skipping
        rgba_2d = rgba.copy() # uint array of shape (H, W, 4) with values [0, 255]
        depth_2d = depth.copy() # float array of shape (H, W) with depth values in [0, 1]
        segmentation_2d = segmentation_mask.copy() # int array of shape (H, W) with body indices
        
        # Skip pixels if requested (subsample)
        if pixel_skip > 1:
            rgba_2d = rgba_2d[::pixel_skip, ::pixel_skip]
            depth_2d = depth_2d[::pixel_skip, ::pixel_skip]
            segmentation_2d = segmentation_2d[::pixel_skip, ::pixel_skip]
            
            # Update effective camera dimensions
            eff_height = rgba_2d.shape[0]
            eff_width = rgba_2d.shape[1]
        else:
            eff_height = self.camera_height
            eff_width = self.camera_width
        
        # Flatten after subsampling
        rgba = rgba_2d.reshape((-1, 4))
        depth = depth_2d.flatten()
        segmentation_mask = segmentation_2d.flatten()

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        # Adjust grid to account for subsampling
        y, x = np.mgrid[-1:1:2 / eff_height, -1:1:2 / eff_width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth
        h = np.ones_like(z)

        # DON'T clamp z values - keep original depths for correct 3D transformation
        pixels = np.stack([x, y, z, h], axis=1)

        # Filter point cloud to only include points on the target body
        if body is not None:
            body_mask = segmentation_mask == body.body
            pixels = pixels[body_mask]
            z = z[body_mask]
            rgba = rgba[body_mask]
            segmentation_mask = segmentation_mask[body_mask]

        # Transform to normalized device coordinates (NDC) for proper projection
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        # Compute distances from camera
        distances = np.linalg.norm(points - self.camera_pos, axis=1)
        points = points
        rgba = rgba
        valid_mask = distances <= max_range

        return points, rgba/255, valid_mask

    def get_rgba_depth(self, **kwargs):
        """
        Get RGBA image and depth map from the current camera pose.
        
        This automatically updates the camera pose before capturing the image.
        """
        self.update_camera_pose()
        return super().get_rgba_depth(**kwargs)
    
def compute_lookat_quaternion(eye: np.ndarray, target: np.ndarray, 
                              up: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """
    Compute quaternion for a camera looking from eye to target.
    
    Args:
        eye: Camera position
        target: Point to look at
        up: Up vector (default Z-up)
        
    Returns:
        Quaternion [x, y, z, w]
    """
    # Use PyBullet's computeViewMatrix and extract orientation
    # This is more robust than manually constructing rotation matrices
    
    # Compute view matrix
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist()
    )
    
    # Extract rotation matrix from view matrix (first 3x3 block, transposed)
    # View matrix is column-major, we need row-major for rotation
    view_matrix_np = np.array(view_matrix).reshape(4, 4, order='F')
    rot_matrix = view_matrix_np[:3, :3].T  # Transpose to get world-to-camera as camera-to-world
    
    # Invert to get camera orientation in world frame
    rot_matrix = rot_matrix.T
    
    # Convert to quaternion using scipy
    try:
        rotation = R.from_matrix(rot_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
    except ValueError:
        # Fallback: use simple forward-looking orientation
        # Just look along the direction vector
        direction = target - eye
        direction = direction / np.linalg.norm(direction)
        
        # Use mengine's quaternion utilities
        # Default orientation [0,0,0,1] looks along +Z
        # We need to rotate to look along 'direction'
        
        # Simple approach: use atan2 to get yaw and pitch
        yaw = np.arctan2(direction[1], direction[0])
        pitch = np.arcsin(-direction[2])
        
        # Convert to quaternion via euler angles
        quat = get_quaternion([0, pitch, yaw])
    
    return quat