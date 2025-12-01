"""
Evaluate Semantic OctoMap Accuracy

This script loads ground truth points and compares them against a semantic octomap
to measure accuracy metrics such as:
- Label accuracy (correct class prediction)
- Confidence scores for correct/incorrect predictions
- Confusion matrix
- Per-class precision, recall, F1-score
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple

import env
from utils import get_quaternion
from scene.roi import RectangleROI
from scene.objects import URDF, Ground, load_object, DebugCoordinateFrame
from scene.scene_representation import SemanticOctoMap


class SemanticMapEvaluator:
    def __init__(self, semantic_map: SemanticOctoMap, ground_truth_file: str):
        """
        Initialize evaluator with a semantic map and ground truth points.
        
        Args:
            semantic_map: SemanticOctoMap instance to evaluate
            ground_truth_file: Path to JSON file with ground truth points
        """
        self.semantic_map = semantic_map
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        self.results = []
        
    def load_ground_truth(self, filename: str) -> Dict:
        """Load ground truth points from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded {data['num_points']} ground truth points from {filename}")
        return data
    
    def evaluate(self) -> Dict:
        """
        Evaluate the semantic map against ground truth points.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING SEMANTIC MAP")
        print("="*60)
        
        results = []
        
        for pt_data in self.ground_truth['points']:
            position = np.array(pt_data['position'])
            gt_label = pt_data.get('label', None)
            gt_class = pt_data.get('class_name', None)
            
            # Query semantic map
            semantic_info = self.semantic_map.get_semantic_info(position)
            
            if semantic_info is None:
                pred_label = None
                pred_confidence = 0.0
                pred_class = "unknown"
            else:
                pred_label = semantic_info['label']
                pred_confidence = semantic_info['confidence']
                pred_class = self.semantic_map.class_names.get(pred_label, f"class_{pred_label}")
            
            # Check occupancy
            is_occupied = self.semantic_map.is_occupied(position)
            
            result = {
                'index': pt_data['index'],
                'position': position.tolist(),
                'gt_label': gt_label,
                'gt_class': gt_class,
                'pred_label': pred_label,
                'pred_class': pred_class,
                'pred_confidence': pred_confidence,
                'is_occupied': is_occupied,
                'correct': (gt_label == pred_label) if gt_label is not None else None
            }
            
            results.append(result)
            
            # Print result
            status = "✓" if result['correct'] else "✗" if result['correct'] is not None else "?"
            print(f"{status} Point {pt_data['index'] + 1}: "
                  f"GT={gt_class}({gt_label}) | "
                  f"Pred={pred_class}({pred_label}, conf={pred_confidence:.2f}) | "
                  f"Occupied={is_occupied}")
        
        self.results = results
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        return metrics
    
    def compute_metrics(self) -> Dict:
        """Compute evaluation metrics from results."""
        print("\n" + "="*60)
        print("METRICS")
        print("="*60)
        
        # Filter results with ground truth labels
        labeled_results = [r for r in self.results if r['gt_label'] is not None]
        
        if len(labeled_results) == 0:
            print("No labeled ground truth points to evaluate")
            return {}
        
        # Overall accuracy
        correct = sum(1 for r in labeled_results if r['correct'])
        total = len(labeled_results)
        accuracy = correct / total if total > 0 else 0.0
        
        # Points with predictions
        predicted = [r for r in labeled_results if r['pred_label'] is not None]
        unknown = [r for r in labeled_results if r['pred_label'] is None]
        
        # Average confidence for correct/incorrect
        correct_preds = [r for r in predicted if r['correct']]
        incorrect_preds = [r for r in predicted if not r['correct']]
        
        avg_conf_correct = np.mean([r['pred_confidence'] for r in correct_preds]) if correct_preds else 0.0
        avg_conf_incorrect = np.mean([r['pred_confidence'] for r in incorrect_preds]) if incorrect_preds else 0.0
        
        # Per-class metrics
        class_metrics = self.compute_per_class_metrics(labeled_results)
        
        metrics = {
            'total_points': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'predicted_points': len(predicted),
            'unknown_points': len(unknown),
            'avg_confidence_correct': avg_conf_correct,
            'avg_confidence_incorrect': avg_conf_incorrect,
            'class_metrics': class_metrics
        }
        
        # Print summary
        print(f"Total points: {total}")
        print(f"Correct predictions: {correct}/{total} ({accuracy*100:.1f}%)")
        print(f"Points with predictions: {len(predicted)}")
        print(f"Points without predictions (unknown): {len(unknown)}")
        print(f"Avg confidence (correct): {avg_conf_correct:.3f}")
        print(f"Avg confidence (incorrect): {avg_conf_incorrect:.3f}")
        
        print("\nPer-class metrics:")
        for class_name, metrics_dict in class_metrics.items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics_dict['precision']:.3f}")
            print(f"    Recall: {metrics_dict['recall']:.3f}")
            print(f"    F1-score: {metrics_dict['f1']:.3f}")
        
        return metrics
    
    def compute_per_class_metrics(self, results: List[Dict]) -> Dict:
        """Compute precision, recall, F1 per class."""
        class_metrics = {}
        
        # Get all unique classes
        all_classes = set()
        for r in results:
            if r['gt_class']:
                all_classes.add(r['gt_class'])
            if r['pred_class'] and r['pred_class'] != 'unknown':
                all_classes.add(r['pred_class'])
        
        for class_name in all_classes:
            # True positives: predicted this class and was correct
            tp = sum(1 for r in results if r['pred_class'] == class_name and r['gt_class'] == class_name)
            
            # False positives: predicted this class but was wrong
            fp = sum(1 for r in results if r['pred_class'] == class_name and r['gt_class'] != class_name)
            
            # False negatives: should have predicted this class but didn't
            fn = sum(1 for r in results if r['gt_class'] == class_name and r['pred_class'] != class_name)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return class_metrics
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix."""
        # Get labeled results
        labeled_results = [r for r in self.results if r['gt_label'] is not None]
        
        if len(labeled_results) == 0:
            print("No labeled data to plot confusion matrix")
            return
        
        # Get class names
        all_classes = sorted(set(
            [r['gt_class'] for r in labeled_results] +
            [r['pred_class'] for r in labeled_results if r['pred_class'] != 'unknown']
        ))
        
        # Add 'unknown' for predictions without semantic info
        all_classes_with_unknown = all_classes + ['unknown']
        n_classes = len(all_classes_with_unknown)
        
        # Build confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        class_to_idx = {cls: i for i, cls in enumerate(all_classes_with_unknown)}
        
        for r in labeled_results:
            gt_idx = class_to_idx[r['gt_class']]
            pred_class = r['pred_class'] if r['pred_class'] else 'unknown'
            pred_idx = class_to_idx[pred_class]
            cm[gt_idx, pred_idx] += 1
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_classes_with_unknown,
                    yticklabels=all_classes_with_unknown,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Class')
        plt.ylabel('Ground Truth Class')
        plt.title('Semantic Map Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(self, save_path: str = None):
        """Plot confidence score distributions."""
        labeled_results = [r for r in self.results if r['gt_label'] is not None and r['pred_label'] is not None]
        
        if len(labeled_results) == 0:
            print("No predictions to plot confidence distribution")
            return
        
        correct_conf = [r['pred_confidence'] for r in labeled_results if r['correct']]
        incorrect_conf = [r['pred_confidence'] for r in labeled_results if not r['correct']]
        
        plt.figure(figsize=(10, 6))
        
        if correct_conf:
            plt.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
        if incorrect_conf:
            plt.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confidence distribution to {save_path}")
        
        plt.show()
    
    def save_results(self, filename: str):
        """Save evaluation results to JSON."""
        data = {
            'ground_truth_file': self.ground_truth,
            'results': self.results,
            'metrics': self.compute_metrics()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSaved evaluation results to {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate semantic octomap accuracy')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth points JSON file')
    parser.add_argument('--semantic_map', type=str, default=None,
                        help='Path to saved semantic map (octree + semantic files)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualizations (confusion matrix, confidence distribution)')
    
    args = parser.parse_args()
    
    # Load or create semantic map
    # For now, this is a placeholder - you'll need to load your actual semantic map
    print("Note: This script expects you to provide a semantic map.")
    print("You can either:")
    print("  1. Load a saved semantic map using --semantic_map")
    print("  2. Modify this script to build the semantic map from scratch")
    print("\nFor demonstration, exiting...")
    
    # TODO: Load semantic map
    # if args.semantic_map:
    #     semantic_map = SemanticOctoMap(...)
    #     semantic_map.load_semantic(...)
    # else:
    #     # Build semantic map from demo
    #     pass
    
    # evaluator = SemanticMapEvaluator(semantic_map, args.ground_truth)
    # metrics = evaluator.evaluate()
    
    # if args.visualize:
    #     evaluator.plot_confusion_matrix()
    #     evaluator.plot_confidence_distribution()
    
    # evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
