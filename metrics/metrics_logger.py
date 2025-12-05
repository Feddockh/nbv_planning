"""
Metrics Logger for NBV Planning

Logs metrics, creates/updates plots, and saves data to .npz files.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path


class MetricsLogger:
    """
    Logger for NBV planning metrics.
    
    Tracks metrics across iterations, generates plots, and saves data.
    """
    
    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        """
        Initialize the metrics logger.
        
        Args:
            output_dir: Base output directory (e.g., "output/")
            experiment_name: Name of the experiment subfolder
        """
        self.output_dir = Path(output_dir) / experiment_name
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.metrics: Dict[str, List[Any]] = {
            'iteration': [],
            # Coverage metrics
            'total_voxels': [],
            'known_voxels': [],
            'occupied_voxels': [],
            'free_voxels': [],
            'unknown_voxels': [],
            'coverage_percent': [],
            'occupied_percent': [],
            # Semantic evaluation metrics
            'total_predictions': [],
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
            'hit_rate': [],
            'tp_avg_distance': [],
            'fp_avg_distance': [],
            'tp_avg_confidence': [],
            'fp_avg_confidence': [],
            'tp_max_confidence': [],
            'fp_max_confidence': [],
        }
        
        # Custom metrics that can be added dynamically
        self.custom_metrics: Dict[str, List[Any]] = {}
        
        print(f"MetricsLogger initialized. Output directory: {self.output_dir}")
    
    def log_iteration(self, iteration: int, **kwargs):
        """
        Log metrics for a single iteration.
        
        Args:
            iteration: Current iteration number
            **kwargs: Metric values to log (must match metric names)
        """
        self.metrics['iteration'].append(iteration)
        
        # Log standard metrics
        for key in self.metrics.keys():
            if key == 'iteration':
                continue
            value = kwargs.get(key, None)
            self.metrics[key].append(value)
        
        # Log custom metrics
        for key, value in kwargs.items():
            if key not in self.metrics:
                if key not in self.custom_metrics:
                    self.custom_metrics[key] = []
                self.custom_metrics[key].append(value)
    
    def add_custom_metric(self, name: str, value: Any):
        """
        Add a custom metric to the most recent iteration.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)
    
    def save_data(self, filename: str = "metrics.npz"):
        """
        Save all metrics to a .npz file.
        
        Args:
            filename: Name of the output file
        """
        filepath = self.data_dir / filename
        
        # Combine standard and custom metrics
        all_metrics = {**self.metrics, **self.custom_metrics}
        
        # Convert lists to arrays where possible
        save_dict = {}
        for key, values in all_metrics.items():
            if len(values) == 0:
                continue
            try:
                # Try to convert to numpy array
                save_dict[key] = np.array(values)
            except (ValueError, TypeError):
                # If conversion fails, save as object array
                save_dict[key] = np.array(values, dtype=object)
        
        np.savez(filepath, **save_dict)
        print(f"Metrics saved to {filepath}")
    
    def load_data(self, filename: str = "metrics.npz"):
        """
        Load metrics from a .npz file.
        
        Args:
            filename: Name of the file to load
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File {filepath} does not exist")
            return
        
        data = np.load(filepath, allow_pickle=True)
        
        # Load into metrics
        for key in data.files:
            if key in self.metrics:
                self.metrics[key] = data[key].tolist()
            else:
                self.custom_metrics[key] = data[key].tolist()
        
        print(f"Metrics loaded from {filepath}")
    
    def plot_metrics(self, metrics_to_plot: Optional[List[str]] = None, 
                     save: bool = True, show: bool = False):
        """
        Generate plots for specified metrics.
        
        Args:
            metrics_to_plot: List of metric names to plot. If None, plots all.
            save: Whether to save plots to disk
            show: Whether to display plots interactively
        """
        if len(self.metrics['iteration']) == 0:
            print("No data to plot")
            return
        
        iterations = self.metrics['iteration']
        
        # Determine which metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = [k for k in self.metrics.keys() 
                             if k != 'iteration' and len(self.metrics[k]) > 0]
            metrics_to_plot += list(self.custom_metrics.keys())
        
        # Create individual plots for each metric
        for metric_name in metrics_to_plot:
            if metric_name in self.metrics:
                values = self.metrics[metric_name]
            elif metric_name in self.custom_metrics:
                values = self.custom_metrics[metric_name]
            else:
                continue
            
            # Skip if no data
            if len(values) == 0 or all(v is None for v in values):
                continue
            
            # Filter out None values
            valid_iterations = [it for it, v in zip(iterations, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if len(valid_values) == 0:
                continue
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(valid_iterations, valid_values, 'b-o', linewidth=2, markersize=6)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            plt.title(f'{metric_name.replace("_", " ").title()} vs Iteration', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save:
                plot_path = self.plots_dir / f"{metric_name}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved: {plot_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
    
    def plot_combined_metrics(self, metric_groups: Dict[str, List[str]], 
                             save: bool = True, show: bool = False):
        """
        Generate combined plots for groups of related metrics.
        
        Args:
            metric_groups: Dict mapping plot titles to lists of metric names
            save: Whether to save plots to disk
            show: Whether to display plots interactively
        """
        if len(self.metrics['iteration']) == 0:
            print("No data to plot")
            return
        
        iterations = self.metrics['iteration']
        
        for group_name, metric_names in metric_groups.items():
            plt.figure(figsize=(12, 6))
            
            for metric_name in metric_names:
                if metric_name in self.metrics:
                    values = self.metrics[metric_name]
                elif metric_name in self.custom_metrics:
                    values = self.custom_metrics[metric_name]
                else:
                    continue
                
                # Filter out None values
                valid_iterations = [it for it, v in zip(iterations, values) if v is not None]
                valid_values = [v for v in values if v is not None]
                
                if len(valid_values) > 0:
                    plt.plot(valid_iterations, valid_values, '-o', 
                           label=metric_name.replace('_', ' ').title(),
                           linewidth=2, markersize=6)
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.title(group_name, fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save:
                plot_path = self.plots_dir / f"{group_name.replace(' ', '_').lower()}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Combined plot saved: {plot_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_iterations': len(self.metrics['iteration']),
            'final_coverage_percent': None,
            'final_known_voxels': None,
            'final_true_positives': None,
            'final_false_positives': None,
            'final_false_negatives': None,
            'final_hit_rate': None,
            'avg_hit_rate': None,
        }
        
        # Get final coverage
        valid_coverage = [v for v in self.metrics['coverage_percent'] if v is not None]
        if valid_coverage:
            summary['final_coverage_percent'] = valid_coverage[-1]
        
        valid_known = [v for v in self.metrics['known_voxels'] if v is not None]
        if valid_known:
            summary['final_known_voxels'] = valid_known[-1]
        
        # Get final semantic evaluation metrics
        valid_tp = [v for v in self.metrics['true_positives'] if v is not None]
        if valid_tp:
            summary['final_true_positives'] = valid_tp[-1]
        
        valid_fp = [v for v in self.metrics['false_positives'] if v is not None]
        if valid_fp:
            summary['final_false_positives'] = valid_fp[-1]
        
        valid_fn = [v for v in self.metrics['false_negatives'] if v is not None]
        if valid_fn:
            summary['final_false_negatives'] = valid_fn[-1]
        
        valid_hit_rate = [v for v in self.metrics['hit_rate'] if v is not None]
        if valid_hit_rate:
            summary['final_hit_rate'] = valid_hit_rate[-1]
            summary['avg_hit_rate'] = np.mean(valid_hit_rate)
        
        return summary
    
    def print_summary(self):
        """Print a summary of the logged metrics."""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("NBV Planning Metrics Summary")
        print("="*60)
        print(f"Total iterations: {summary['total_iterations']}")
        if summary['final_coverage_percent'] is not None:
            print(f"Final coverage: {summary['final_coverage_percent']:.2f}%")
        if summary['final_known_voxels'] is not None:
            print(f"Final known voxels: {summary['final_known_voxels']}")
        if summary['final_true_positives'] is not None:
            print(f"Final true positives: {summary['final_true_positives']}")
        if summary['final_false_positives'] is not None:
            print(f"Final false positives: {summary['final_false_positives']}")
        if summary['final_false_negatives'] is not None:
            print(f"Final false negatives: {summary['final_false_negatives']}")
        if summary['final_hit_rate'] is not None:
            print(f"Final hit rate: {summary['final_hit_rate']*100:.2f}%")
        if summary['avg_hit_rate'] is not None:
            print(f"Avg hit rate: {summary['avg_hit_rate']*100:.2f}%")
        print("="*60 + "\n")
