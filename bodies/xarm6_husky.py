"""
xArm6 + Husky Robot Definition

This is a placeholder for your custom xArm6 mounted on Husky mobile base.
Follow this pattern when you add your URDF file.

To add your robot:
1. Place your URDF file in: project/nbv_planning/assets/xarm6_husky/xarm6_husky.urdf
2. Place mesh files in: project/nbv_planning/assets/xarm6_husky/meshes/
3. Update the controllable_joints list with your robot's joint indices
4. Update end_effector with the link index for your end effector
5. Update gripper_joints if you have a gripper
"""

import os
import pybullet as p
from bodies.robot import Robot


class Xarm6Husky(Robot):
    def __init__(self, env, position=[0, 0, 0], orientation=[0, 0, 0, 1], 
                 controllable_joints=None, fixed_base=False):
        """
        xArm6 mounted on Husky mobile base robot.
        
        Args:
            env: mengine environment
            position: base position [x, y, z]
            orientation: base orientation quaternion [x, y, z, w]
            controllable_joints: list of joint indices to control (None = auto-detect)
            fixed_base: whether to fix the base (False for mobile base)
        """
        # TODO: Update these values based on your actual robot URDF
        # For xArm6 (6 DOF arm) + Husky (4 wheels), you might have:
        # - 4 wheel joints (indices 0-3) 
        # - 6 arm joints (indices 4-9)
        # - gripper joints if applicable
        
        # Example joint configuration (UPDATE THESE):
        if controllable_joints is None:
            # Typically: [wheel joints, arm joints]
            controllable_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        end_effector = 10  # UPDATE: Link index for end effector
        gripper_joints = []  # UPDATE: Add gripper joint indices if you have a gripper
        
        # Path to your URDF file
        urdf_path = os.path.join(os.path.dirname(__file__), 
                                 '../assets/xarm6_husky/xarm6_husky.urdf')
        
        # Check if URDF exists, otherwise provide helpful error
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"URDF file not found at: {urdf_path}\n"
                f"Please add your xArm6+Husky URDF file to:\n"
                f"  project/nbv_planning/assets/xarm6_husky/xarm6_husky.urdf\n"
                f"And mesh files to:\n"
                f"  project/nbv_planning/assets/xarm6_husky/meshes/"
            )
        
        # Load the URDF
        body = p.loadURDF(urdf_path, 
                         useFixedBase=fixed_base, 
                         basePosition=position, 
                         baseOrientation=orientation,
                         physicsClientId=env.id)
        
        super().__init__(body, env, controllable_joints, end_effector, gripper_joints)
        
        # Set initial joint configuration (UPDATE THESE VALUES)
        # Example for 6-DOF arm in a neutral pose
        initial_angles = [0] * len(controllable_joints)
        # You might want something like:
        # initial_angles = [0, 0, 0, 0,  # wheels
        #                   0, -0.5, 0, 1.57, 0, 0]  # arm joints
        
        try:
            self.set_joint_angles(initial_angles)
        except:
            print(f"Warning: Could not set initial joint angles. "
                  f"Verify controllable_joints list matches your URDF.")
        
        # Initialize gripper if present
        if gripper_joints:
            self.set_gripper_position([0]*len(gripper_joints), set_instantly=True)
    
    def get_camera_mount_link(self):
        """
        Returns the link index where the camera should be mounted.
        Typically this is the end effector link.
        """
        return self.end_effector
