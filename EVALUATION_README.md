# Semantic OctoMap Accuracy Evaluation Tools

This directory contains tools for evaluating the accuracy of semantic occupancy maps by comparing them against manually placed ground truth points.

## Tools

### 1. `place_ground_truth_points.py`
Interactive tool for placing ground truth points in the simulation.

**Usage:**
```bash
python place_ground_truth_points.py
```

**Controls:**
- `Arrow Keys`: Move point in X/Y plane
  - `UP/DOWN`: Forward/backward (X-axis)
  - `LEFT/RIGHT`: Left/right (Y-axis)
- `I/K`: Move point higher/lower (Z-axis)
- `SPACE`: Toggle fine adjustment mode (10x smaller steps)
- `R`: Reset point to center position
- `ENTER`: Save current point and prepare for next one
- `Q`: Quit and save all points to JSON file

**Features:**
- Visual feedback with colored sphere (green=normal, red=fine mode)
- Optional semantic label assignment for each point
- Saves points with timestamps to JSON file
- Real-time position display

**Output:**
Creates a JSON file named `ground_truth_points_YYYYMMDD_HHMMSS.json` containing:
```json
{
  "num_points": 10,
  "step_size": 0.01,
  "points": [
    {
      "index": 0,
      "position": [0.123, 0.456, 0.789],
      "label": 1,
      "class_name": "fire_blight",
      "timestamp": "2025-11-30T12:34:56.789"
    },
    ...
  ]
}
```

### 2. `evaluate_semantic_map.py`
Evaluates semantic map accuracy using ground truth points.

**Usage:**
```bash
python evaluate_semantic_map.py --ground_truth ground_truth_points.json --visualize
```

**Arguments:**
- `--ground_truth`: Path to ground truth JSON file (required)
- `--semantic_map`: Path to saved semantic map files (optional)
- `--output`: Output file for results (default: evaluation_results.json)
- `--visualize`: Show plots (confusion matrix, confidence distribution)

**Metrics Computed:**
- Overall accuracy (% correct predictions)
- Per-class precision, recall, F1-score
- Average confidence for correct vs incorrect predictions
- Confusion matrix
- Coverage (% of points with predictions)

**Output:**
Creates a JSON file with detailed results:
```json
{
  "total_points": 50,
  "accuracy": 0.85,
  "correct_predictions": 42,
  "predicted_points": 48,
  "unknown_points": 2,
  "avg_confidence_correct": 0.78,
  "avg_confidence_incorrect": 0.42,
  "class_metrics": {
    "fire_blight": {
      "precision": 0.88,
      "recall": 0.85,
      "f1": 0.86
    }
  }
}
```

## Workflow

### Step 1: Create Semantic Map
Run your semantic mapping pipeline (e.g., `demo_semantic_octomap.py`) to build the semantic map.

### Step 2: Place Ground Truth Points
```bash
python place_ground_truth_points.py
```

1. At startup, choose whether to assign semantic labels
2. If yes, define your class labels (e.g., 0=background, 1=fire_blight, 2=canker)
3. Navigate to interesting points in the scene using Arrow Keys and I/K controls
4. Press ENTER to save each point
5. If using labels, select the appropriate class for each point
6. Press Q when finished

**Tips:**
- Use SPACE to toggle fine mode for precise placement
- Arrow keys move in the X/Y plane, I/K move up/down
- Place points at boundaries between different semantic regions
- Include points in all semantic classes for balanced evaluation
- Place some points in uncertain/ambiguous regions

### Step 3: Evaluate Accuracy
```bash
python evaluate_semantic_map.py \
  --ground_truth ground_truth_points_20251130_123456.json \
  --visualize
```

This will:
1. Load ground truth points
2. Query the semantic map at each point
3. Compare predictions vs ground truth
4. Compute accuracy metrics
5. Generate visualizations (if --visualize flag used)

### Step 4: Analyze Results
Review the metrics to identify:
- Which classes are well-predicted vs poorly-predicted
- Whether confidence scores correlate with accuracy
- Regions of the map with missing predictions (unknown)
- Confusion between similar classes

## Integration with Your Code

To evaluate your semantic map programmatically:

```python
from scene.scene_representation import SemanticOctoMap
from evaluate_semantic_map import SemanticMapEvaluator

# Load or create your semantic map
semantic_map = SemanticOctoMap(bounds=roi, resolution=0.01)
# ... build the map ...

# Evaluate
evaluator = SemanticMapEvaluator(semantic_map, "ground_truth_points.json")
metrics = evaluator.evaluate()

# Visualize
evaluator.plot_confusion_matrix(save_path="confusion_matrix.png")
evaluator.plot_confidence_distribution(save_path="confidence_dist.png")

# Save results
evaluator.save_results("evaluation_results.json")

print(f"Accuracy: {metrics['accuracy']*100:.1f}%")
```

## Dependencies

- numpy
- matplotlib
- seaborn (optional, for nicer confusion matrix plots)
- pybullet (for interactive placement)

Install with:
```bash
pip install numpy matplotlib seaborn
```
