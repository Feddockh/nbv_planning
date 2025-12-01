"""
Interactive Ground Truth Point Placement Tool

This script allows you to place ground truth points in the simulation for
evaluating semantic occupancy map accuracy.

Controls:
- Arrow Keys (UP/DOWN/LEFT/RIGHT): Move point in X/Y plane
- I/K: Move point higher/lower (Z-axis)
- ENTER: Save current point and move to next
- Q: Quit and save all points
- R: Reset current point position
- SPACE: Toggle fine adjustment mode (smaller steps)

The points are saved to a JSON file for later evaluation.
"""

import os
import json
import numpy as np
import pybullet as p
from datetime import datetime

import env
from utils import get_quaternion
from scene.roi import RectangleROI
from scene.objects import URDF, Ground, load_object, DebugCoordinateFrame, clear_debug_items, Points
from bodies.panda import Panda
from scene.scene_representation import SemanticOctoMap
from vision import RobotCamera


class PointPlacer:
    def __init__(self, initial_position=[0, 0, 1.0], step_size=0.01, physics_client_id=0):
        """
        Initialize the point placer.
        
        Args:
            initial_position: Starting position [x, y, z]
            step_size: Movement step size in meters
            physics_client_id: PyBullet physics client ID
        """
        self.position = np.array(initial_position, dtype=float)
        self.step_size = step_size
        self.fine_mode = False
        self.fine_step_multiplier = 0.1
        self.physics_client_id = physics_client_id
        
        self.saved_points = []
        self.current_point_index = 0
        
        # Debug visualization - use visual sphere instead of debug items
        self.marker_point = None  # Current movable marker
        self.saved_point_markers = []  # List of saved point markers
        self.local_env = None
        
        # Create initial visualization
        self.update_visualization()
        
        print("\n" + "="*60)
        print("POINT PLACEMENT TOOL")
        print("="*60)
        print("Controls:")
        print("  Arrow Keys: Move in X/Y plane")
        print("    UP/DOWN: Forward/backward (X-axis)")
        print("    LEFT/RIGHT: Left/right (Y-axis)")
        print("  I/K: Move higher/lower (Z-axis)")
        print("  ENTER: Save point and move to next")
        print("  Q: Quit and save all points")
        print("  R: Reset current point position")
        print("  SPACE: Toggle fine adjustment mode")
        print("="*60)
        print(f"Current step size: {self.step_size}m")
        print(f"Point {self.current_point_index + 1}")
        print(f"Position: [{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}]")
        print()
    
    def get_step_size(self):
        """Get current step size based on fine mode."""
        if self.fine_mode:
            return self.step_size * self.fine_step_multiplier
        return self.step_size
    
    def move(self, direction):
        """
        Move the point in the specified direction.
        
        Args:
            direction: numpy array [dx, dy, dz]
        """
        step = self.get_step_size()
        self.position += direction * step
        self.update_visualization()
        self.print_status()
    
    def reset_position(self, position=None):
        """Reset position to default or specified location."""
        if position is None:
            position = [0, 0, 1.0]
        self.position = np.array(position, dtype=float)
        self.update_visualization()
        self.print_status()
    
    def toggle_fine_mode(self):
        """Toggle fine adjustment mode."""
        self.fine_mode = not self.fine_mode
        mode_str = "FINE" if self.fine_mode else "NORMAL"
        step = self.get_step_size()
        print(f"\n>>> Mode: {mode_str} (step size: {step:.4f}m)")
        self.print_status()
    
    def save_current_point(self, label=None, class_name=None):
        """Save the current point and prepare for next one."""
        point_data = {
            'index': self.current_point_index,
            'position': self.position.tolist(),
            'label': label,
            'class_name': class_name,
            'timestamp': datetime.now().isoformat()
        }
        self.saved_points.append(point_data)
        
        # Auto-save after each point
        self.save_to_file(self.get_autosave_filename())
        
        # Create a permanent marker at the saved position (blue color)
        saved_marker = Points([self.position.copy()], rgba=[0, 0, 1, 1], radius=0.015, local_env=self.local_env)
        self.saved_point_markers.append(saved_marker)
        
        print(f"\n>>> SAVED Point {self.current_point_index + 1}")
        print(f"    Position: [{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}]")
        if class_name:
            print(f"    Class: {class_name} (label {label})")
        
        # Move to next point
        self.current_point_index += 1
        
        # Keep current position as starting point for next placement
        self.update_visualization()
        print(f"\n>>> Ready for Point {self.current_point_index + 1}")
        self.print_status()
    
    def update_visualization(self):
        """Update the debug visualization of the current point."""
        # Color changes based on mode (green for normal, red for fine)
        color = [1, 0, 0, 1] if self.fine_mode else [0, 1, 0, 1]
        
        # Always update the marker point position and color
        # Note: Points function updates position if replace_points is provided
        self.marker_point = Points([self.position], rgba=color, radius=0.01, 
                                  replace_points=self.marker_point, local_env=self.local_env)
    
    def print_status(self):
        """Print current status."""
        mode_str = " [FINE]" if self.fine_mode else ""
        print(f"Point {self.current_point_index + 1}{mode_str}: "
              f"[{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}]", 
              end='\r')
    
    def save_to_file(self, filename=None):
        """Save all points to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ground_truth_points_{timestamp}.json"
        
        data = {
            'num_points': len(self.saved_points),
            'step_size': self.step_size,
            'points': self.saved_points
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n>>> Saved {len(self.saved_points)} points to {filename}")
        return filename
    
    def get_autosave_filename(self):
        """Get the autosave filename."""
        return "ground_truth_points_autosave.json"


def main():
    # Initialize environment
    print("Initializing environment...")
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
    env.step_simulation(steps=100, realtime=False)
    
    # Initialize point placer at center of ROI
    initial_pos = center.copy()
    placer = PointPlacer(initial_position=initial_pos, step_size=0.01, physics_client_id=nbv_env.id)
    placer.local_env = nbv_env  # Set the environment for Points visualization
    
    # Ask user for class labels
    print("\nDo you want to assign semantic labels to points? (y/n): ", end='')
    use_labels_input = input().strip().lower()
    use_labels = use_labels_input == 'y'
    
    class_labels = {}
    if use_labels:
        print("\nDefine your class labels (press Enter with empty name to finish):")
        label_id = 0
        while True:
            class_name = input(f"Class {label_id} name: ").strip()
            if not class_name:
                break
            class_labels[str(label_id)] = class_name
            label_id += 1
        print(f"\nDefined {len(class_labels)} classes: {class_labels}")
    
    print("\n" + "="*60)
    print("READY! Click on the PyBullet window and use keyboard controls")
    print("="*60)
    
    # Main loop
    running = True
    last_key_state = {}  # Track last state to detect new presses
    
    while running:
        try:
            # Step simulation
            env.step_simulation(steps=1, realtime=True)
            
            # Check for keyboard events
            keys = p.getKeyboardEvents(physicsClientId=nbv_env.id)
        except (p.error, Exception) as e:
            print(f"\n\n!!! Physics server disconnected or error occurred: {e}")
            print("Saving points before exit...")
            running = False
            break
        
        for key, state in keys.items():
            # Check if this is a new key press (wasn't pressed before or state changed to pressed)
            # State 3 = KEY_IS_DOWN | KEY_WAS_TRIGGERED (new press)
            # State 1 = KEY_IS_DOWN (held down)
            is_new_press = (state & p.KEY_WAS_TRIGGERED) or (key not in last_key_state and state == p.KEY_IS_DOWN)
            
            if not is_new_press:
                continue
            
            # Movement controls - using arrow keys and I/K to avoid PyBullet conflicts
            if key == p.B3G_UP_ARROW:
                placer.move(np.array([1, 0, 0]))  # Forward (X+)
            elif key == p.B3G_DOWN_ARROW:
                placer.move(np.array([-1, 0, 0]))  # Backward (X-)
            elif key == p.B3G_LEFT_ARROW:
                placer.move(np.array([0, 1, 0]))  # Left (Y+)
            elif key == p.B3G_RIGHT_ARROW:
                placer.move(np.array([0, -1, 0]))  # Right (Y-)
            elif key == ord('i'):
                placer.move(np.array([0, 0, 1]))  # Higher (Z+)
            elif key == ord('k'):
                placer.move(np.array([0, 0, -1]))  # Lower (Z-)
            
            # Mode toggles
            elif key == ord(' '):  # Space bar
                placer.toggle_fine_mode()
            
            # Reset
            elif key == ord('r'):
                print("\n>>> Resetting to center position")
                placer.reset_position(center)
            
            # Save point
            elif key == p.B3G_RETURN:  # Enter key
                label = None
                class_name = None
                
                if use_labels:
                    print("\nSelect class for this point:")
                    for lbl, name in class_labels.items():
                        print(f"  {lbl}: {name}")
                    print("Class ID: ", end='')
                    label_input = input().strip()
                    
                    if label_input in class_labels:
                        label = int(label_input)
                        class_name = class_labels[label_input]
                    else:
                        print(f"Invalid class ID, saving without label")
                
                placer.save_current_point(label=label, class_name=class_name)
            
            # Quit
            elif key == ord('q'):
                print("\n\n>>> Quitting...")
                running = False
                break
        
        # Update last key state
        last_key_state = keys.copy()
    
    # Save points to file with timestamp
    if len(placer.saved_points) > 0:
        # Save final version with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ground_truth_points_{timestamp}.json"
        placer.save_to_file(filename)
        
        print(f"\nSummary:")
        print(f"  Total points placed: {len(placer.saved_points)}")
        print(f"  Saved to: {filename}")
        print(f"  Also saved to: {placer.get_autosave_filename()}")
        
        # Print all points
        print("\nAll points:")
        for pt in placer.saved_points:
            pos = pt['position']
            label_str = f" - {pt['class_name']} ({pt['label']})" if pt['class_name'] else ""
            print(f"  Point {pt['index'] + 1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]{label_str}")
    else:
        print("\nNo points were saved.")
    
    print("\nExiting...")


if __name__ == "__main__":
    main()
