"""
Display Semantic OctoMap

Script to load and visualize a saved semantic octomap from .npz files.
Loads both the octree structure and semantic information, then displays it in PyBullet.

Usage:
    python display_semantic_octomap.py <octomap_points_path> <octomap_labels_path>
    
Example:
    python display_semantic_octomap.py output/semantic_nbv_default/data/semantic_octomap_points.npz \
                                       output/semantic_nbv_default/data/semantic_octomap_labels.npz
"""

import sys
import os
import argparse
import numpy as np
import pybullet as p
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.scene_representation import SemanticOctoMap
from scene.objects import Ground, load_object
import env


def display_semantic_octomap(octomap_points_path: str, octomap_labels_path: str,
                             label_to_display: int = None,
                             min_confidence: float = 0.0,
                             point_size: float = 5.0,
                             visualize_free: bool = True,
                             scale_by_confidence: bool = False):
    """
    Load and display a semantic octomap from saved files.
    
    Args:
        octomap_points_path: Path to the octree binary file (.npz)
        octomap_labels_path: Path to the semantic labels file (.npz)
        label_to_display: If specified, only show voxels with this label (None = all)
        min_confidence: Minimum confidence threshold for visualization
        point_size: Size of visualization points
        visualize_free: Whether to visualize free (non-semantic) voxels
        scale_by_confidence: Whether to scale point size by confidence
    """
    # Check if files exist
    if not os.path.exists(octomap_points_path):
        print(f"Error: Octree file not found: {octomap_points_path}")
        return
    
    if not os.path.exists(octomap_labels_path):
        print(f"Error: Semantic labels file not found: {octomap_labels_path}")
        return
    
    # Initialize PyBullet
    print("Initializing PyBullet...")
    nbv_env = env.Env()
    ground = Ground(filename=os.path.join(nbv_env.asset_dir, 'dirt_plane', 'dirt_plane.urdf'))
    obj = load_object("apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8])
    obj.change_visual(link=obj.base, rgba=[1, 1, 1, 0.5])
    
    # Set camera view for better visualization
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    # Create a semantic octomap instance
    print("\nLoading semantic octomap...")
    semantic_octomap = SemanticOctoMap(resolution=0.02)  # Resolution will be set from loaded file
    
    # Load the semantic octomap
    semantic_octomap.load_semantic(octomap_points_path, octomap_labels_path)
    
    # Print statistics
    print("\n" + "="*60)
    print("SEMANTIC OCTOMAP STATISTICS")
    print("="*60)
    
    # Get octree statistics
    print(f"\nOctree Information:")
    print(f"  Resolution: {semantic_octomap.octree.getResolution():.4f} m")
    print(f"  Tree depth: {semantic_octomap.octree.getTreeDepth()}")
    print(f"  Total nodes: {semantic_octomap.octree.size()}")
    
    # Get semantic statistics
    print(f"\nSemantic Information:")
    print(f"  Total semantic voxels: {len(semantic_octomap.semantic_map)}")
    
    if semantic_octomap.class_names:
        print(f"\n  Class names:")
        for label, name in semantic_octomap.class_names.items():
            count = sum(1 for v in semantic_octomap.semantic_map.values() if v['label'] == label)
            print(f"    {label}: {name} ({count} voxels)")
    
    # Count voxels per label
    label_counts = {}
    confidence_stats = {}
    
    for voxel_key, semantic_info in semantic_octomap.semantic_map.items():
        label = semantic_info['label']
        confidence = semantic_info['confidence']
        
        if label not in label_counts:
            label_counts[label] = 0
            confidence_stats[label] = []
        
        label_counts[label] += 1
        confidence_stats[label].append(confidence)
    
    print(f"\n  Voxels per label:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        avg_conf = np.mean(confidence_stats[label])
        max_conf = np.max(confidence_stats[label])
        min_conf = np.min(confidence_stats[label])
        class_name = semantic_octomap.class_names.get(label, "Unknown")
        print(f"    Label {label} ({class_name}): {count} voxels")
        print(f"      Confidence - avg: {avg_conf:.3f}, min: {min_conf:.3f}, max: {max_conf:.3f}")
    
    # Update stats to cache free/occupied points
    print("\nUpdating octree statistics...")
    semantic_octomap.update_stats(verbose=True)
    
    # Visualize the semantic octomap
    print("\nVisualizing semantic octomap...")
    print(f"  Display settings:")
    print(f"    Label filter: {label_to_display if label_to_display is not None else 'All labels'}")
    print(f"    Min confidence: {min_confidence:.2f}")
    print(f"    Point size: {point_size}")
    print(f"    Show free voxels: {visualize_free}")
    print(f"    Scale by confidence: {scale_by_confidence}")
    
    debug_handles = semantic_octomap.visualize_semantic(
        label=label_to_display,
        min_confidence=min_confidence,
        point_size=point_size,
        visualize_free=visualize_free,
        scale_by_confidence=scale_by_confidence,
        colors=[[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # Gray, Red, Blue for different classes
    )
    
    print(f"\nVisualization complete. Created {len(debug_handles)} debug items.")
    print("\n" + "="*60)
    print("Press Ctrl+C to exit")
    print("="*60)
    
    # Keep the visualization open
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        p.disconnect()


def main():
    """Main function to parse arguments and display semantic octomap."""
    parser = argparse.ArgumentParser(
        description="Load and visualize a semantic octomap from .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display all semantic voxels
  python display_semantic_octomap.py output/data/semantic_octomap_points.npz output/data/semantic_octomap_labels.npz
  
  # Display only label 0 with minimum confidence 0.5
  python display_semantic_octomap.py points.npz labels.npz --label 0 --min-confidence 0.5
  
  # Display with larger points, scaled by confidence
  python display_semantic_octomap.py points.npz labels.npz --point-size 10 --scale-confidence
        """
    )
    
    parser.add_argument(
        'octomap_points',
        type=str,
        help='Path to the octree points file (semantic_octomap_points.npz)'
    )
    
    parser.add_argument(
        'octomap_labels',
        type=str,
        help='Path to the semantic labels file (semantic_octomap_labels.npz)'
    )
    
    parser.add_argument(
        '--label', '-l',
        type=int,
        default=None,
        help='Only display voxels with this label (default: all labels)'
    )
    
    parser.add_argument(
        '--min-confidence', '-c',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0.0-1.0, default: 0.0)'
    )
    
    parser.add_argument(
        '--point-size', '-s',
        type=float,
        default=5.0,
        help='Size of visualization points (default: 5.0)'
    )
    
    parser.add_argument(
        '--no-free',
        action='store_true',
        help='Do not visualize free (non-semantic) voxels'
    )
    
    parser.add_argument(
        '--scale-confidence',
        action='store_true',
        help='Scale point size by confidence (0.5x to 2x)'
    )
    
    args = parser.parse_args()
    
    # Display the semantic octomap
    display_semantic_octomap(
        octomap_points_path=args.octomap_points,
        octomap_labels_path=args.octomap_labels,
        label_to_display=args.label,
        min_confidence=args.min_confidence,
        point_size=args.point_size,
        visualize_free=not args.no_free,
        scale_by_confidence=args.scale_confidence
    )


if __name__ == "__main__":
    main()
