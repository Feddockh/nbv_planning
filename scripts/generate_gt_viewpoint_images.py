"""
Generate RGB images from viewpoints sampled around ground truth points.

This script:
1. Builds the environment with the tree
2. Loads ground truth points
3. Samples viewpoints from hemispheres around each GT point
4. Captures RGB images from each viewpoint
5. Saves images organized by GT point index
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from scene.objects import URDF, Ground, load_object
from vision import Camera, compute_lookat_quaternion
from viewpoints.viewpoint_proposal import sample_views_from_hemisphere
from viewpoints.viewpoint import Viewpoint
from utils import get_quaternion


def load_ground_truth(filename: str) -> Dict:
    """Load ground truth points from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Loaded {data['num_points']} ground truth points from {filename}")
    return data


def setup_environment():
    """Setup PyBullet environment with tree object."""
    print("=== Setting up environment ===")
    nbv_env = env.Env()
    
    # Ground plane
    ground = Ground(filename=os.path.join(nbv_env.asset_dir, 'dirt_plane', 'dirt_plane.urdf'))
    
    # Load apple tree with disease
    print("Loading apple tree object...")
    obj = load_object("apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8])
    
    # Let objects settle
    env.step_simulation(steps=100, realtime=False)
    
    return nbv_env, ground, obj


def sample_viewpoints_for_gt_point(
    gt_point: Dict,
    base_orientation: np.ndarray,
    hemisphere_params: Dict
) -> List[Viewpoint]:
    """
    Sample viewpoints from a hemisphere around a ground truth point.
    
    Args:
        gt_point: Ground truth point dict with 'position' key
        base_orientation: Base orientation quaternion for the hemisphere
        hemisphere_params: Parameters for hemisphere sampling
        
    Returns:
        List of Viewpoint objects
    """
    center = np.array(gt_point['position'])
    
    viewpoints = sample_views_from_hemisphere(
        center=center,
        base_orientation=base_orientation,
        min_radius=hemisphere_params['min_radius'],
        max_radius=hemisphere_params.get('max_radius', None),
        num_samples=hemisphere_params['num_samples'],
        use_positive_z=hemisphere_params.get('use_positive_z', False),
        z_bias_sigma=hemisphere_params.get('z_bias_sigma', 0.3),
        min_distance=hemisphere_params.get('min_distance', 0.1),
        max_attempts=hemisphere_params.get('max_attempts', 1000)
    )
    
    return viewpoints


def capture_image_from_viewpoint(
    viewpoint: Viewpoint,
    camera_params: Dict,
    nbv_env
) -> np.ndarray:
    """
    Capture RGB image from a viewpoint.
    
    Args:
        viewpoint: Viewpoint to capture from
        camera_params: Camera parameters (fov, width, height, flash settings)
        nbv_env: Environment instance
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Create camera at viewpoint
    camera = Camera(
        camera_pos=viewpoint.position.tolist(),
        look_at_pos=viewpoint.target.tolist(),
        fov=camera_params.get('fov', 60),
        camera_width=camera_params.get('width', 640),
        camera_height=camera_params.get('height', 480),
        local_env=nbv_env
    )
    
    # Capture image
    rgba, depth, segmentation = camera.get_rgba_depth(
        shadow=camera_params.get('shadow', False),
        ambient=camera_params.get('ambient', 0.8),
        diffuse=camera_params.get('diffuse', 0.3),
        specular=camera_params.get('specular', 0.1),
        flash=camera_params.get('flash', False),
        flash_intensity=camera_params.get('flash_intensity', 1.5),
        shutter_speed=camera_params.get('shutter_speed', 0.5),
        max_flash_distance=camera_params.get('max_flash_distance', 1.0)
    )
    
    # Convert to RGB (drop alpha channel)
    rgb = rgba[:, :, :3]
    
    return rgb


def save_image(image: np.ndarray, filepath: str):
    """Save RGB image to file."""
    from PIL import Image
    img = Image.fromarray(image.astype(np.uint8))
    img.save(filepath)


def main():
    parser = argparse.ArgumentParser(
        description='Generate RGB images from viewpoints around ground truth points'
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='Path to ground truth points JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='gt_viewpoint_images',
        help='Output directory for images'
    )
    parser.add_argument(
        '--num_views_per_point',
        type=int,
        default=10,
        help='Number of viewpoints to sample around each GT point'
    )
    parser.add_argument(
        '--min_hemisphere_radius',
        type=float,
        default=0.05,
        help='Minimum radius of hemisphere around each GT point (meters)'
    )
    parser.add_argument(
        '--max_hemisphere_radius',
        type=float,
        default=0.5,
        help='Maximum radius of hemisphere around each GT point (meters)'
    )
    parser.add_argument(
        '--min_distance',
        type=float,
        default=0.05,
        help='Minimum distance between sampled viewpoints (meters)'
    )
    parser.add_argument(
        '--camera_width',
        type=int,
        default=1440,
        help='Camera image width'
    )
    parser.add_argument(
        '--camera_height',
        type=int,
        default=1080,
        help='Camera image height'
    )
    parser.add_argument(
        '--fov',
        type=float,
        default=60,
        help='Camera field of view (degrees)'
    )
    parser.add_argument(
        '--flash',
        action='store_true',
        help='Enable flash effect (high ambient/diffuse lighting)'
    )
    parser.add_argument(
        '--flash_intensity',
        type=float,
        default=2.0,
        help='Flash intensity multiplier'
    )
    parser.add_argument(
        '--shutter_speed',
        type=float,
        default=0.1,
        help='Camera shutter speed for flash effect'
    )
    parser.add_argument(
        '--max_flash_distance',
        type=float,
        default=1.0,
        help='Maximum effective distance for flash (meters)'
    )
    parser.add_argument(
        '--flat_structure',
        action='store_true',
        help='Store all images in a single folder instead of subdirectories per GT point'
    )
    parser.add_argument(
        '--hemisphere_euler',
        type=float,
        nargs=3,
        default=None,
        metavar=('ROLL', 'PITCH', 'YAW'),
        help='Hemisphere orientation as Euler angles in radians (roll, pitch, yaw). '
             'Example: 0 1.5708 3.1416 for [0, pi/2, pi]. '
             'If not provided, defaults to looking downward (180° around X-axis)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup environment
    nbv_env, ground, tree_obj = setup_environment()
    
    # Load ground truth points
    gt_data = load_ground_truth(args.ground_truth)
    gt_points = gt_data['points']
    
    # Camera parameters
    if args.flash:
        # Flash effect: enable flash with tunable parameters
        camera_params = {
            'fov': args.fov,
            'width': args.camera_width,
            'height': args.camera_height,
            'shadow': False,
            'ambient': 0.8,
            'diffuse': 0.3,
            'specular': 0.1,
            'flash': True,
            'flash_intensity': args.flash_intensity,
            'shutter_speed': args.shutter_speed,
            'max_flash_distance': args.max_flash_distance
        }
    else:
        # Normal lighting
        camera_params = {
            'fov': args.fov,
            'width': args.camera_width,
            'height': args.camera_height,
            'shadow': False,
            'ambient': 0.8,
            'diffuse': 0.3,
            'specular': 0.1,
            'flash': False
        }
    
    # Hemisphere sampling parameters
    hemisphere_params = {
        'min_radius': args.min_hemisphere_radius,
        'max_radius': args.max_hemisphere_radius,  # Sample on surface
        'num_samples': args.num_views_per_point,
        'use_positive_z': False,  # Use -Z hemisphere (looking down/inward)
        'z_bias_sigma': 0.3,
        'min_distance': args.min_distance,
        'max_attempts': 1000
    }
    
    # Base orientation: hemisphere direction
    if args.hemisphere_euler is not None:
        # Use custom Euler angles provided by user
        base_orientation = get_quaternion(args.hemisphere_euler)
        print(f"Using custom hemisphere orientation: Euler {args.hemisphere_euler} -> Quat {base_orientation}")
    else:
        # Default: facing away from Z-axis (looking downward in Z-up convention)
        # This is a 180-degree rotation around X-axis to flip from +Z to -Z
        base_orientation = R.from_euler('x', 180, degrees=True).as_quat()  # [x, y, z, w]
        print(f"Using default hemisphere orientation (looking downward)")
    
    # Process each ground truth point
    total_images = 0
    all_metadata = []  # For flat structure, collect all metadata
    
    for gt_idx, gt_point in enumerate(gt_points):
        print(f"\n=== Processing GT point {gt_idx + 1}/{len(gt_points)} ===")
        print(f"Position: {gt_point['position']}")
        
        # Create subdirectory only if not using flat structure
        if args.flat_structure:
            gt_dir = output_dir
        else:
            gt_dir = output_dir / f"gt_point_{gt_idx:03d}"
            gt_dir.mkdir(exist_ok=True)
        
        # Sample viewpoints around this GT point
        viewpoints = sample_viewpoints_for_gt_point(
            gt_point,
            base_orientation,
            hemisphere_params
        )
        
        print(f"Sampled {len(viewpoints)} viewpoints")
        
        # Capture and save image from each viewpoint
        viewpoint_metadata = []
        for view_idx, viewpoint in enumerate(viewpoints):
            # Capture image
            rgb_image = capture_image_from_viewpoint(
                viewpoint,
                camera_params,
                nbv_env
            )
            
            # Save image with appropriate naming
            if args.flat_structure:
                image_filename = gt_dir / f"gt{gt_idx:03d}_view{view_idx:03d}.png"
            else:
                image_filename = gt_dir / f"view_{view_idx:03d}.png"
            save_image(rgb_image, str(image_filename))
            
            # Store viewpoint metadata
            viewpoint_metadata.append({
                'index': view_idx,
                'position': viewpoint.position.tolist(),
                'orientation': viewpoint.orientation.tolist(),
                'target': viewpoint.target.tolist()
            })
            
            total_images += 1
            
            if (view_idx + 1) % 5 == 0:
                print(f"  Captured {view_idx + 1}/{len(viewpoints)} views")
        
        # Save or collect metadata
        gt_metadata = {
            'gt_point_index': gt_idx,
            'gt_point': gt_point,
            'num_viewpoints': len(viewpoints),
            'base_orientation': base_orientation.tolist(),
            'viewpoints': viewpoint_metadata
        }
        
        if args.flat_structure:
            all_metadata.append(gt_metadata)
        else:
            metadata_file = gt_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(gt_metadata, f, indent=2)
    
    # Save combined metadata for flat structure
    if args.flat_structure:
        metadata_file = output_dir / 'all_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'num_gt_points': len(gt_points),
                'total_images': total_images,
                'gt_points_metadata': all_metadata
            }, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(gt_points)} ground truth points")
    print(f"Generated {total_images} total images")
    print(f"Images saved to: {output_dir}")
    
    # Disconnect environment
    env.disconnect()


if __name__ == '__main__':
    main()
