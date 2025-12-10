"""
Plot Metrics from NBV Planning Runs

Script to load and visualize metrics from multiple .npz files,
allowing comparison across different runs and experiments.

Usage:
    python plot_metrics.py <npz_file1> [npz_file2 ...] --metrics <metric1> [metric2 ...]
    
Examples:
    # Plot coverage_percent from a single run
    python plot_metrics.py output/run1/data/metrics.npz --metrics coverage_percent
    
    # Compare coverage between two runs
    python plot_metrics.py output/run1/data/metrics.npz output/run2/data/metrics.npz \
                           --metrics coverage_percent --labels "Run 1" "Run 2"
    
    # Plot multiple metrics from multiple runs
    python plot_metrics.py run1/metrics.npz run2/metrics.npz \
                           --metrics coverage_percent hit_rate true_positives \
                           --labels "Volumetric NBV" "Semantic NBV"
    
    # Create separate plots for each metric
    python plot_metrics.py run1/metrics.npz run2/metrics.npz \
                           --metrics coverage_percent hit_rate \
                           --separate
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path


def load_metrics_from_npz(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load metrics from a .npz file.
    
    Args:
        filepath: Path to the .npz file
        
    Returns:
        Dictionary of metric name -> values
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = np.load(filepath, allow_pickle=True)
    metrics = {}
    
    for key in data.files:
        metrics[key] = data[key]
    
    return metrics


def get_available_metrics(npz_files: List[str]) -> List[str]:
    """
    Get all available metrics across all provided .npz files.
    
    Args:
        npz_files: List of paths to .npz files
        
    Returns:
        Sorted list of unique metric names
    """
    all_metrics = set()
    
    for filepath in npz_files:
        try:
            metrics = load_metrics_from_npz(filepath)
            all_metrics.update(metrics.keys())
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    # Remove 'iteration' from the list since it's the x-axis
    all_metrics.discard('iteration')
    
    return sorted(all_metrics)


def plot_single_metric(npz_files: List[str], 
                       metric_name: str,
                       labels: Optional[List[str]] = None,
                       title: Optional[str] = None,
                       xlabel: str = 'Iteration',
                       ylabel: Optional[str] = None,
                       save_path: Optional[str] = None,
                       show: bool = True,
                       figsize: Tuple[int, int] = (12, 6),
                       style: str = '-o',
                       linewidth: float = 2,
                       markersize: float = 6) -> plt.Figure:
    """
    Plot a single metric from one or more .npz files.
    
    Args:
        npz_files: List of paths to .npz files
        metric_name: Name of the metric to plot
        labels: Optional labels for each run (defaults to filenames)
        title: Plot title (defaults to metric name)
        xlabel: X-axis label
        ylabel: Y-axis label (defaults to metric name)
        save_path: Path to save the plot (None = don't save)
        show: Whether to display the plot
        figsize: Figure size (width, height)
        style: Line style (e.g., '-o', '--', '-')
        linewidth: Line width
        markersize: Marker size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate default labels if not provided
    if labels is None:
        labels = [Path(f).stem for f in npz_files]
    elif len(labels) != len(npz_files):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of files ({len(npz_files)})")
    
    # Plot each run
    for filepath, label in zip(npz_files, labels):
        try:
            metrics = load_metrics_from_npz(filepath)
            
            # Check if metric exists
            if metric_name not in metrics:
                print(f"Warning: Metric '{metric_name}' not found in {filepath}")
                continue
            
            # Get iterations and metric values
            iterations = metrics.get('iteration', np.arange(len(metrics[metric_name])))
            values = metrics[metric_name]
            
            # Filter out None values
            valid_mask = np.array([v is not None for v in values])
            valid_iterations = iterations[valid_mask]
            valid_values = values[valid_mask]
            
            if len(valid_values) == 0:
                print(f"Warning: No valid data for '{metric_name}' in {filepath}")
                continue
            
            # Plot
            ax.plot(valid_iterations, valid_values, style, 
                   label=label, linewidth=linewidth, markersize=markersize)
            
        except Exception as e:
            print(f"Error plotting {filepath}: {e}")
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel or metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'{metric_name.replace("_", " ").title()} vs {xlabel}', fontsize=14)
    
    # Add legend if multiple runs
    if len(npz_files) > 1:
        ax.legend(fontsize=10, loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_multiple_metrics(npz_files: List[str],
                          metric_names: List[str],
                          labels: Optional[List[str]] = None,
                          title: Optional[str] = None,
                          xlabel: str = 'Iteration',
                          ylabel: str = 'Value',
                          save_path: Optional[str] = None,
                          show: bool = True,
                          figsize: Tuple[int, int] = (14, 7),
                          separate_plots: bool = False,
                          output_dir: Optional[str] = None) -> List[plt.Figure]:
    """
    Plot multiple metrics from one or more .npz files.
    
    Args:
        npz_files: List of paths to .npz files
        metric_names: List of metric names to plot
        labels: Optional labels for each run
        title: Overall plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the combined plot (only used if separate_plots=False)
        show: Whether to display plots
        figsize: Figure size
        separate_plots: If True, create separate plots for each metric
        output_dir: Directory to save plots (only used if separate_plots=True)
        
    Returns:
        List of matplotlib Figure objects
    """
    figures = []
    
    if separate_plots:
        # Create separate plot for each metric
        for metric_name in metric_names:
            save_path_metric = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path_metric = os.path.join(output_dir, f"{metric_name}.png")
            
            fig = plot_single_metric(
                npz_files=npz_files,
                metric_name=metric_name,
                labels=labels,
                save_path=save_path_metric,
                show=show,
                figsize=figsize
            )
            figures.append(fig)
    else:
        # Create combined plot with all metrics
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate default labels if not provided
        if labels is None:
            labels = [Path(f).stem for f in npz_files]
        
        # Color palette for different runs
        colors = plt.cm.tab10(np.linspace(0, 1, len(npz_files)))
        
        # Line styles for different metrics
        line_styles = ['-', '--', '-.', ':']
        
        # Plot each combination of run and metric
        for run_idx, (filepath, label) in enumerate(zip(npz_files, labels)):
            try:
                metrics = load_metrics_from_npz(filepath)
                iterations = metrics.get('iteration', None)
                
                for metric_idx, metric_name in enumerate(metric_names):
                    if metric_name not in metrics:
                        continue
                    
                    values = metrics[metric_name]
                    
                    # Use iterations if available, otherwise use index
                    if iterations is not None:
                        x_values = iterations
                    else:
                        x_values = np.arange(len(values))
                    
                    # Filter out None values
                    valid_mask = np.array([v is not None for v in values])
                    valid_x = x_values[valid_mask]
                    valid_y = values[valid_mask]
                    
                    if len(valid_y) == 0:
                        continue
                    
                    # Create label combining run and metric
                    plot_label = f"{label} - {metric_name.replace('_', ' ')}"
                    
                    # Use different line styles for different metrics
                    style = line_styles[metric_idx % len(line_styles)]
                    
                    ax.plot(valid_x, valid_y, style, 
                           color=colors[run_idx],
                           label=plot_label,
                           linewidth=2,
                           markersize=6,
                           marker='o')
                
            except Exception as e:
                print(f"Error plotting {filepath}: {e}")
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title or 'Metrics Comparison', fontsize=14)
        ax.legend(fontsize=9, loc='best', ncol=min(2, len(metric_names)))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Combined plot saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        figures.append(fig)
    
    return figures


def print_metrics_summary(npz_files: List[str], metric_names: Optional[List[str]] = None):
    """
    Print a summary of metrics from the .npz files.
    
    Args:
        npz_files: List of paths to .npz files
        metric_names: Optional list of specific metrics to summarize (None = all)
    """
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    for filepath in npz_files:
        print(f"\nFile: {filepath}")
        print("-" * 80)
        
        try:
            metrics = load_metrics_from_npz(filepath)
            
            # Determine which metrics to display
            if metric_names:
                display_metrics = [m for m in metric_names if m in metrics]
            else:
                display_metrics = [k for k in sorted(metrics.keys()) if k != 'iteration']
            
            # Print iterations info
            if 'iteration' in metrics:
                print(f"  Iterations: {len(metrics['iteration'])} total")
            
            # Print each metric
            for metric_name in display_metrics:
                values = metrics[metric_name]
                
                # Filter out None values
                valid_values = [v for v in values if v is not None]
                
                if len(valid_values) == 0:
                    print(f"  {metric_name}: No valid data")
                    continue
                
                # Compute statistics
                valid_array = np.array(valid_values)
                final_value = valid_values[-1]
                mean_value = np.mean(valid_array)
                max_value = np.max(valid_array)
                min_value = np.min(valid_array)
                
                print(f"  {metric_name}:")
                print(f"    Final: {final_value:.4f}")
                print(f"    Mean:  {mean_value:.4f}")
                print(f"    Min:   {min_value:.4f}")
                print(f"    Max:   {max_value:.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print("="*80 + "\n")


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description="Plot metrics from NBV planning .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'npz_files',
        nargs='+',
        type=str,
        help='Path(s) to .npz metric files'
    )
    
    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        type=str,
        default=None,
        help='Metric name(s) to plot (if not specified, will list available metrics)'
    )
    
    parser.add_argument(
        '--labels', '-l',
        nargs='+',
        type=str,
        default=None,
        help='Labels for each run (must match number of files)'
    )
    
    parser.add_argument(
        '--title', '-t',
        type=str,
        default=None,
        help='Plot title'
    )
    
    parser.add_argument(
        '--xlabel',
        type=str,
        default='Iteration',
        help='X-axis label (default: Iteration)'
    )
    
    parser.add_argument(
        '--ylabel',
        type=str,
        default=None,
        help='Y-axis label (default: based on metric name)'
    )
    
    parser.add_argument(
        '--separate', '-s',
        action='store_true',
        help='Create separate plots for each metric'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory or file path to save plots'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots interactively'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[12, 6],
        help='Figure size (width height), default: 12 6'
    )
    
    parser.add_argument(
        '--list-metrics',
        action='store_true',
        help='List all available metrics in the files and exit'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics for metrics'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    for filepath in args.npz_files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    # List available metrics if requested
    if args.list_metrics:
        available_metrics = get_available_metrics(args.npz_files)
        print("\nAvailable metrics:")
        for metric in available_metrics:
            print(f"  - {metric}")
        print(f"\nTotal: {len(available_metrics)} metrics\n")
        sys.exit(0)
    
    # Print summary if requested
    if args.summary:
        print_metrics_summary(args.npz_files, args.metrics)
        if not args.metrics:
            sys.exit(0)
    
    # Determine metrics to plot
    if args.metrics is None:
        available_metrics = get_available_metrics(args.npz_files)
        print("\nNo metrics specified. Available metrics:")
        for metric in available_metrics:
            print(f"  - {metric}")
        print("\nPlease specify metrics using --metrics flag")
        print("Example: --metrics coverage_percent hit_rate\n")
        sys.exit(1)
    
    # Generate plots
    print(f"\nPlotting {len(args.metrics)} metric(s) from {len(args.npz_files)} file(s)...")
    
    if len(args.metrics) == 1 and not args.separate:
        # Single metric plot
        save_path = args.output
        plot_single_metric(
            npz_files=args.npz_files,
            metric_name=args.metrics[0],
            labels=args.labels,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            save_path=save_path,
            show=not args.no_show,
            figsize=tuple(args.figsize)
        )
    else:
        # Multiple metrics
        output_dir = args.output if args.separate else None
        save_path = args.output if not args.separate else None
        
        plot_multiple_metrics(
            npz_files=args.npz_files,
            metric_names=args.metrics,
            labels=args.labels,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel or 'Value',
            save_path=save_path,
            show=not args.no_show,
            figsize=tuple(args.figsize),
            separate_plots=args.separate,
            output_dir=output_dir
        )
    
    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
