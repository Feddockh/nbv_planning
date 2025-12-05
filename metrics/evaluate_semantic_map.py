"""
Semantic OctoMap Distance Evaluator (3D only)

For each ground-truth point:
  - Find nearest semantic voxel of the same label.
  - Compute Euclidean distance.
  - If distance <= threshold -> hit, else miss.

3D visualization:
  - Blue spheres  : ground-truth lesion points
  - Green spheres : matched semantic voxels (hits)
  - Red spheres   : nearest semantic voxels that are misses
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import env
from scene.objects import URDF, Ground, load_object, DebugPoints
from scene.scene_representation import SemanticOctoMap


class SemanticMapDistanceEvaluator:
    def __init__(
        self,
        semantic_map: SemanticOctoMap,
        ground_truth_file: str,
        distance_threshold: float = 0.05,
        score_radius: Optional[float] = None,
        min_confidence: float = 0.0,
    ):
        """
        Args:
            semantic_map: SemanticOctoMap instance (already built or loaded).
            ground_truth_file: Path to JSON file with ground truth points.
            distance_threshold: Max distance (m) for hit vs miss.
            score_radius: Radius (m) for distance-weighted confidence score.
                          If None, defaults to distance_threshold.
            min_confidence: Minimum semantic confidence for voxels to be considered.
        """
        self.semantic_map = semantic_map
        self.distance_threshold = float(distance_threshold)
        self.score_radius = float(score_radius) if score_radius is not None else float(
            distance_threshold
        )
        self.min_confidence = float(min_confidence)

        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.results: List[Dict] = []

    def _load_ground_truth(self, filename: str) -> Dict:
        with open(filename, "r") as f:
            data = json.load(f)
        print(f"Loaded {data['num_points']} ground truth points from {filename}")
        return data
    
    def _get_all_predictions(self) -> List[Tuple[np.ndarray, float, int]]:
        """
        Returns:
            List of (coord, confidence, label) for all *semantic prediction voxels*.
            Free / unknown / unlabeled voxels are excluded.
        """
        preds = []
        for (x, y, z), info in self.semantic_map.semantic_map.items():

            # Skip voxels with no semantic class assigned
            if info["label"] is None or info["label"] < 0:
                continue

            # Skip voxels below minimum confidence
            if info["confidence"] < self.min_confidence:
                continue

            coord = np.array([x, y, z], dtype=float)
            conf = float(info["confidence"])
            label = int(info["label"])
            preds.append((coord, conf, label))

        return preds
    
    def _get_predictions_with_label(self, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            coords: (N,3) voxel centers for this label
            confs:  (N,)  confidences for those voxels
        """
        coords = []
        confs = []
        for (x, y, z), info in self.semantic_map.semantic_map.items():
            if info["label"] == label and info["confidence"] >= self.min_confidence:
                coords.append([x, y, z])
                confs.append(info["confidence"])
        if not coords:
            return np.zeros((0, 3)), np.zeros((0,), dtype=float)
        return np.array(coords, dtype=float), np.array(confs, dtype=float)
    
    def _lookup_match_coord(self, matches: List[Tuple[Dict, np.ndarray]], coord: np.ndarray) -> int:
        for i, (matched_pt, matched_coord) in enumerate(matches):
            if np.array_equal(matched_coord, coord):
                return i
        return -1
    
    def _lookup_match_gt(self, matches: List[Tuple[Dict, np.ndarray]], gt_pt: Dict) -> int:
        for i, (matched_pt, matched_coord) in enumerate(matches):
            if matched_pt["index"] == gt_pt["index"]:
                return i
        return -1

    def evaluate(self, verbose: bool = True) -> Dict:
        """
        For each labeled GT point:
          - Get voxels of same label with confidences.
          - Compute nearest distance (for hit/miss).
          - Compute distance-weighted confidence score S(g).

        Args:
            verbose: If True, print results to console

        Returns:
            metrics dict with hit_rate, distances, semantic scores, etc.
        """

        results: Dict[str, Any] = {}
        results["TPs"] = []
        results["FPs"] = []
        results["FNs"] = []
        results["TP_distances"] = []
        results["FP_distances"] = []
        results["TP_confidences"] = []
        results["FP_confidences"] = []

        # Pre-filter labeled GTs
        labeled_gts = [
            pt for pt in self.ground_truth["points"]
            if pt.get("label", None) is not None
        ]

        # Iterate through all predictions once
        all_preds = self._get_all_predictions()
        for coord, conf, pred_label in all_preds:
            # GTs with the same label as this prediction
            same_label_gts = [
                pt for pt in labeled_gts
                if pt.get("label", None) == pred_label
            ]

            if same_label_gts:
                gt_positions = np.array(
                    [pt["position"] for pt in same_label_gts],
                    dtype=float,
                )
                dists = np.linalg.norm(gt_positions - coord[None, :], axis=1)
                min_idx = int(np.argmin(dists))
                d = float(dists[min_idx])
                closest_gt = same_label_gts[min_idx]
            else:
                # No GT of this label
                d = float("nan")
                closest_gt = None

            if closest_gt is not None and d <= self.distance_threshold:
                # True Positive: prediction close to a GT of same label
                results["TPs"].append((closest_gt, coord))
                results["TP_distances"].append(d)
                results["TP_confidences"].append(conf)
            else:
                # False Positive: no close GT of same label
                results["FPs"].append(coord)
                results["FP_distances"].append(d)
                results["FP_confidences"].append(conf)

        # Finally, compute the ground truth points that were not matched at all (FN)
        for gt_pt in self.ground_truth["points"]:
            gt_label = gt_pt.get("label", None)
            # Skip unlabeled points
            if gt_label is None:
                continue
            # Check if this GT point was matched
            matched_idx = self._lookup_match_gt(results["TPs"], gt_pt)
            if matched_idx == -1:
                # Not matched, add to FN list
                results["FNs"].append(gt_pt)

        # Compute metrics
        num_TP = len(results["TPs"])
        num_FP = len(results["FPs"])
        num_FN = len(results["FNs"])
        total_predictions = num_TP + num_FP
        hit_rate = num_TP / total_predictions if total_predictions > 0 else 0.0

        # Compute the average TP and FP distances
        results["TP_avg_distance"] = np.mean(results["TP_distances"]) if results["TP_distances"] else float('nan')
        results["FP_avg_distance"] = np.mean(results["FP_distances"]) if results["FP_distances"] else float('nan')

        # Compute the average TP and FP confidences
        results["TP_avg_confidence"] = np.mean(results["TP_confidences"]) if results["TP_confidences"] else float('nan')
        results["FP_avg_confidence"] = np.mean(results["FP_confidences"]) if results["FP_confidences"] else float('nan')

        # Compute the max TP and FP confidences
        results["TP_max_confidence"] = max(results["TP_confidences"]) if results["TP_confidences"] else float('nan')
        results["FP_max_confidence"] = max(results["FP_confidences"]) if results["FP_confidences"] else float('nan')
        
        # Add summary metrics
        results["total_predictions"] = total_predictions
        results["num_TP"] = num_TP
        results["num_FP"] = num_FP
        results["num_FN"] = num_FN
        results["hit_rate"] = hit_rate
        
        self.results = results
        if verbose:
            self._print_results(results)
        return results
    
    def _print_results(self, results: Dict):
        num_TP = results["num_TP"]
        num_FP = results["num_FP"]
        num_FN = results["num_FN"]
        total_predictions = results["total_predictions"]
        labeled_gts = [pt for pt in self.ground_truth["points"] if pt.get("label", None) is not None]
        total_gt = len(labeled_gts)
        hit_rate = results["hit_rate"]

        print("\nSemantic Map Distance Evaluation Results:")
        print(f"  Total ground-truth points : {total_gt}")
        print(f"  Total predictions         : {total_predictions}")
        print(f"  True Positives (hits)     : {num_TP}")
        print(f"  False Positives (misses)  : {num_FP}")
        print(f"  False Negatives           : {num_FN}")
        print(f"  Hit Rate                  : {hit_rate*100:.2f}%")
        print(f"  Avg TP Distance (m)       : {results['TP_avg_distance']:.4f}" if not np.isnan(results['TP_avg_distance']) else "  Avg TP Distance (m)      : N/A")
        print(f"  Avg FP Distance (m)       : {results['FP_avg_distance']:.4f}" if not np.isnan(results['FP_avg_distance']) else "  Avg FP Distance (m)      : N/A")
        print(f"  Avg TP Confidence         : {results['TP_avg_confidence']:.4f}" if not np.isnan(results['TP_avg_confidence']) else "  Avg TP Confidence        : N/A")
        print(f"  Avg FP Confidence         : {results['FP_avg_confidence']:.4f}" if not np.isnan(results['FP_avg_confidence']) else "  Avg FP Confidence        : N/A")
        print(f"  Max TP Confidence         : {results['TP_max_confidence']:.4f}" if not np.isnan(results['TP_max_confidence']) else "  Max TP Confidence        : N/A")
        print(f"  Max FP Confidence         : {results['FP_max_confidence']:.4f}" if not np.isnan(results['FP_max_confidence']) else "  Max FP Confidence        : N/A")

    def visualize_matches_3d(
        self,
        gt_color=(0.0, 0.0, 1.0),
        hit_color=(0.0, 1.0, 0.0),
        miss_color=(1.0, 0.0, 0.0),
        gt_size=0.01,
        voxel_size=0.01,
    ):
        """
        Visualize in 3D (PyBullet) using DebugPoints:

        - Ground-truth lesion points: blue spheres
        - Matched semantic voxels (hits): green spheres
        - Unmatched prediction voxels (FPs): red spheres
        """
        if not self.results:
            print("No evaluation results found. Run evaluate() first.")
            return

        # Labeled GT points
        gt_pts = [
            pt for pt in self.ground_truth["points"]
            if pt.get("label", None) is not None
        ]
        if not gt_pts:
            print("No labeled ground truth points to visualize.")
            return

        gt_positions = np.array([pt["position"] for pt in gt_pts], dtype=float)

        # Hit voxels (TPs store (gt_pt, coord))
        hit_voxels = [coord for (gt_pt, coord) in self.results.get("TPs", [])]
        hit_voxels = np.array(hit_voxels, dtype=float) if hit_voxels else np.zeros((0, 3))

        # Miss voxels (FPs store just coord)
        miss_voxels = list(self.results.get("FPs", []))
        miss_voxels = np.array(miss_voxels, dtype=float) if miss_voxels else np.zeros((0, 3))

        debug_handles = []

        # 1) Ground-truth lesion points (blue)
        if gt_positions.size > 0:
            h = DebugPoints(
                gt_positions,
                points_rgb=list(gt_color),
                size=gt_size,
            )
            if isinstance(h, list):
                debug_handles.extend(h)
            else:
                debug_handles.append(h)

        # 2) Hit voxels (green)
        if hit_voxels.size > 0:
            h = DebugPoints(
                hit_voxels,
                points_rgb=list(hit_color),
                size=voxel_size,
            )
            if isinstance(h, list):
                debug_handles.extend(h)
            else:
                debug_handles.append(h)

        # 3) Miss voxels (red)
        if miss_voxels.size > 0:
            h = DebugPoints(
                miss_voxels,
                points_rgb=list(miss_color),
                size=voxel_size,
            )
            if isinstance(h, list):
                debug_handles.extend(h)
            else:
                debug_handles.append(h)

        print(
            f"Visualized {gt_positions.shape[0]} GT points (blue), "
            f"{hit_voxels.shape[0]} hit voxels (green), "
            f"{miss_voxels.shape[0]} miss voxels (red)."
        )

        return debug_handles


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate semantic octomap distances")
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth points JSON file",
    )
    parser.add_argument(
        "--octree",
        type=str,
        required=True,
        help="Path to saved OctoMap binary (.bt or similar)",
    )
    parser.add_argument(
        "--semantic",
        type=str,
        required=True,
        help="Path to saved semantic map .npz file",
    )
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=0.10,
        help="Distance threshold (m) for hit vs miss",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.0,
        help="Minimum semantic confidence for voxels",
    )

    args = parser.parse_args()

    # Bring up environment and tree object for visual in PyBullet
    nbv_env = env.Env()
    ground = Ground(
        filename=os.path.join(nbv_env.asset_dir, "dirt_plane", "dirt_plane.urdf")
    )
    obj = load_object(
        "apple_tree_crook_canker", obj_position=[0, 0, 0], scale=[0.8, 0.8, 0.8]
    )
    obstacles = [obj, ground]  # in case you need them later

    # Load semantic octomap
    semantic_map = SemanticOctoMap()
    semantic_map.load_semantic(args.octree, args.semantic)

    # Make sure class names are set (if not already in the npz)
    if not semantic_map.class_names:
        semantic_map.set_class_names({0: "Crook", 1: "Canker"})

    # Evaluate
    evaluator = SemanticMapDistanceEvaluator(
        semantic_map,
        args.ground_truth,
        distance_threshold=args.dist_thresh,
        score_radius=None,
        min_confidence=args.min_conf,
    )
    metrics = evaluator.evaluate()

    # 3D visualization: GT (blue), hit voxels (green), miss voxels (red)
    evaluator.visualize_matches_3d(
        gt_color=(0.0, 0.0, 1.0),
        hit_color=(0.0, 1.0, 0.0),
        miss_color=(1.0, 0.0, 0.0),
        gt_size=10.0,
        voxel_size=10.0,
    )

    # keep the window open depending on how env.Env is implemented
    print("Semantic evaluation complete. Inspect the scene in PyBullet.")

    # Keep running for visualization
    print("Press Ctrl+C to exit")
    try:
        while True:
            env.step_simulation(steps=1, realtime=True)
    except KeyboardInterrupt:
        print("\nExiting...")

    env.disconnect()


if __name__ == "__main__":
    main()
