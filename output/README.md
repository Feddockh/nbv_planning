# Output Directory

This directory contains the outputs from NBV planning experiments.

## Structure

Each experiment creates a subdirectory named according to the `EXPERIMENT_NAME` configuration parameter. For example:
- `volumetric_nbv_default/` - Default volumetric NBV planning results
- `semantic_nbv_default/` - Default semantic NBV planning results

## Subdirectory Contents

Each experiment subdirectory contains:

### `data/`
- `metrics.npz` - Logged metrics from all iterations (numpy compressed format)
- `volumetric_octomap_points.npz` or `semantic_octomap_points.npz` - Final octomap point data
- `volumetric_octomap_labels.npz` or `semantic_octomap_labels.npz` - Final octomap semantic labels

### `plots/`
Individual metric plots (PNG format):
- `num_detections.png` - Number of detections per iteration
- `num_frontiers.png` - Number of frontiers per iteration
- `num_clusters.png` - Number of frontier clusters per iteration
- `best_information_gain.png` - Information gain of selected viewpoint per iteration
- `best_utility.png` - Utility of selected viewpoint per iteration
- `best_cost.png` - Cost of selected viewpoint per iteration
- And more...

Combined plots:
- `viewpoint_generation.png` - Combined plot of frontier/cluster/viewpoint metrics
- `information_metrics.png` - Combined plot of information gain, utility, and cost
- `detection_and_points.png` - Combined plot of detections and point cloud sizes

## Usage

The logging system is automatically initialized in the NBV planning scripts:

```python
from logging.metrics_logger import MetricsLogger

# Initialize logger
logger = MetricsLogger(output_dir="output", experiment_name="my_experiment")

# Log metrics during iterations
logger.log_iteration(
    iteration=i,
    num_detections=len(detections),
    best_information_gain=ig,
    # ... more metrics
)

# Save data and generate plots
logger.save_data()
logger.plot_metrics(save=True, show=False)

# Generate summary at end
logger.print_summary()
```

## Notes

- Output files are excluded from version control (see `.gitignore`)
- Plots and data are updated after each iteration
- Only the `.gitkeep` file is tracked in git to preserve directory structure
