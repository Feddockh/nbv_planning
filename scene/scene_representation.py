import numpy as np
from typing import List, Optional, Tuple, Union, Dict
import pyoctomap
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans

from scene.objects import DebugPoints
from scene.roi import RectangleROI


def _generate_distinct_colors(n: int) -> List[List[float]]:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            hue = i / max(n, 1)
            # Convert HSV to RGB (simple approximation)
            h = hue * 6
            x = 1 - abs((h % 2) - 1)
            if h < 1:
                rgb = [1, x, 0]
            elif h < 2:
                rgb = [x, 1, 0]
            elif h < 3:
                rgb = [0, 1, x]
            elif h < 4:
                rgb = [0, x, 1]
            elif h < 5:
                rgb = [x, 0, 1]
            else:
                rgb = [1, 0, x]
            colors.append(rgb)
        return colors

class SceneRepresentation:
    """Base class for object 3D representations."""
    def __init__(self, bounds: Optional[Union[np.ndarray, RectangleROI]] = None, resolution: float = 0.05):
        self.bounds: np.ndarray = None
        if bounds is not None:
            if isinstance(bounds, np.ndarray):
                self.bounds = bounds
            else:
                self.bounds = bounds.to_bounds()
        self.resolution = resolution
        self.frontiers: np.ndarray = np.empty((0, 3))
        self.free_points: np.ndarray = np.empty((0, 3))
        self.occupied_points: np.ndarray = np.empty((0, 3))

    def add_point_cloud(self):
        raise NotImplementedError("Subclasses must implement add_point_cloud()")
    
    def update_stats(self):
        raise NotImplementedError("Subclasses must implement update_stats()")

    def visualize(self, free_color: List[float] = [0, 1, 0], occupied_color: List[float] = [1, 1, 0],
                  point_size: float = 5.0, max_points: int = 100000):
        """Visualize the object representation using cached free/occupied points."""
        if self.free_points.shape[0] > max_points:
            free_points = self.free_points[:max_points]
            print(f"WARN: Limiting free points visualization to {max_points} points.")
        else:
            free_points = self.free_points

        if self.occupied_points.shape[0] > max_points:
            occupied_points = self.occupied_points[:max_points]
            print(f"WARN: Limiting occupied points visualization to {max_points} points.")
        else:
            occupied_points = self.occupied_points

        debug_handles = []
        
        # Visualize free voxels
        if len(free_points) > 0:
            handles = DebugPoints(free_points, points_rgb=free_color, size=point_size)
            # DebugPoints returns a list of debug IDs, extend instead of append
            if isinstance(handles, list):
                debug_handles.extend(handles)
            else:
                debug_handles.append(handles)
        
        # Visualize occupied voxels
        if len(occupied_points) > 0:
            handles = DebugPoints(occupied_points, points_rgb=occupied_color, size=point_size)
            # DebugPoints returns a list of debug IDs, extend instead of append
            if isinstance(handles, list):
                debug_handles.extend(handles)
            else:
                debug_handles.append(handles)

        return debug_handles

    def find_frontiers(self):
        raise NotImplementedError("Subclasses must implement find_frontiers()")
    
    def visualize_frontiers(self, frontiers: np.ndarray = None,
                           color: List[float] = [1, 0, 0],
                           point_size: float = 5.0):
        """Visualize frontier voxels."""
        if frontiers is None:
            frontiers = self.frontiers
        debug_handles = DebugPoints(frontiers, points_rgb=color, size=point_size)
        print(f"Visualized {len(frontiers)} frontier voxels")
        return debug_handles


class OctoMap(SceneRepresentation):
    """Object representation using OctoMap (octree-based 3D occupancy mapping)."""
    
    def __init__(self, bounds: Optional[Union[np.ndarray, RectangleROI]] = None, resolution: float = 0.05):
        super().__init__(bounds, resolution)
        self.octree = pyoctomap.OcTree(resolution)
    
    def add_point_cloud(self, point_cloud: np.ndarray, sensor_origin: np.ndarray, max_range: float = -1.0, lazy_eval: bool = False, discretize: bool = True):
        """
        Add a point cloud to the octree.
        
        Args:
            point_cloud: Nx3 array of 3D points
            sensor_origin: 3D point representing sensor position for ray casting.
            max_range: Maximum range for points (-1.0 = no limit)
            lazy_eval: If True, delay updates to inner nodes for efficiency
            discretize: If True, discretize points to voxel centers before insertion

        Returns:
            success_count: Number of points successfully integrated into the octree
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float64)
        
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError(f"Point cloud must be Nx3, got shape {point_cloud.shape}")

        success_count = self.octree.insertPointCloud(point_cloud, sensor_origin=sensor_origin, max_range=max_range, lazy_eval=lazy_eval, discretize=discretize)
        return success_count

    def update_stats(self, verbose: bool = False, max_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute statistics about the octree occupancy.
        
        Args:
            verbose: If True, print statistics to console
            max_points: Maximum number of voxel points to consider for stats
            
        Returns:
            free_points: Nx3 array of free voxel positions
            occupied_points: Mx3 array of occupied voxel positions
        """

        free_points = []
        occupied_points = []
        point_count = 0

        # Iterate through leaf nodes within ROI if specified
        if self.bounds is not None:
            roi_min, roi_max = np.array(self.bounds[:3]), np.array(self.bounds[3:])
            leaf_iterator = self.octree.begin_leafs_bbx(roi_min, roi_max)
        else:
            leaf_iterator = self.octree.begin_leafs()
        
        # Iterate through all leaf nodes
        for leaf_it in leaf_iterator:
            if point_count >= max_points:
                print(f"WARN: Reached max_points limit ({max_points}). Increase limit to see more voxels.")
                break
            
            try:
                coord = leaf_it.getCoordinate()
                point = np.array(coord, dtype=np.float64)
                
                # Classify as occupied or free
                if self.octree.isNodeOccupied(leaf_it):
                    occupied_points.append(point)
                else:
                    free_points.append(point)
                
                point_count += 1
            except Exception as e:
                continue
        
        # Print statistics
        if verbose:
            print(f"Free voxels: {len(free_points)}")
            print(f"Occupied voxels: {len(occupied_points)}")
            print(f"Total visualized: {len(free_points) + len(occupied_points)}")
            print(f"Total tree size: {self.octree.size()}")

        # Convert to numpy arrays
        free_points = np.array(free_points) if free_points else np.zeros((0, 3))
        occupied_points = np.array(occupied_points) if occupied_points else np.zeros((0, 3))

        # Save to internal variables
        self.free_points = free_points
        self.occupied_points = occupied_points

        return free_points, occupied_points

    def find_frontiers(self, min_unknown_neighbors: int = 1) -> np.ndarray:
        """
        Find frontier voxels - free voxels near the boundary of unknown space.
        
        Args:
            min_unknown_neighbors: Minimum unknown neighbors to qualify as frontier
            
        Returns:
            frontiers (np.ndarray): List of frontier positions (Nx3 numpy array)
        """
        frontiers = []
        
        if self.bounds is None:
            # Use entire octree
            leaf_iterator = self.octree.begin_leafs()
        else:
            roi_min, roi_max = np.array(self.bounds[:3]), np.array(self.bounds[3:])
            leaf_iterator = self.octree.begin_leafs_bbx(roi_min, roi_max)
        
        # Iterate through leaves and check for frontiers in one pass
        for leaf_it in leaf_iterator:
            # Only consider free voxels
            if self.octree.isNodeOccupied(leaf_it):
                continue
            
            voxel = np.array(leaf_it.getCoordinate(), dtype=np.float64)
            
            # Generate 6-neighborhood (connectivity)
            neighbors = [
                voxel + np.array([self.resolution, 0, 0]),
                voxel + np.array([-self.resolution, 0, 0]),
                voxel + np.array([0, self.resolution, 0]),
                voxel + np.array([0, -self.resolution, 0]),
                voxel + np.array([0, 0, self.resolution]),
                voxel + np.array([0, 0, -self.resolution]),
            ]
            
            # Count unknown neighbors for this voxel
            unknown_count = 0
            for neighbor in neighbors:
                # Check if neighbor is unknown (search returns None)
                node = self.octree.search(neighbor)
                if node is None:
                    unknown_count += 1
            
            # Add to frontiers if it has enough unknown neighbors
            if unknown_count >= min_unknown_neighbors:
                frontiers.append(voxel)

        # Convert to numpy array and save
        frontiers = np.array(frontiers) if frontiers else np.empty((0, 3))
        self.frontiers = frontiers
        
        return frontiers
    
    def cluster_frontiers(self, frontiers: Optional[np.ndarray] = None,
                            eps: Optional[float] = None, 
                            min_samples: int = 3, 
                            algorithm: str = 'dbscan',
                            n_clusters: Optional[int] = None) -> Dict:
        """
        Cluster frontier voxels into groups using spatial clustering.
        
        Args:
            frontiers: Optional frontier points to cluster (defaults to self.frontiers)
            eps: Maximum distance between two samples for clustering (default: resolution)
                 Only used for 'dbscan' and 'hdbscan'
            min_samples: Minimum number of samples in a neighborhood to form a cluster
                        Only used for 'dbscan' and 'hdbscan'
            algorithm: Clustering algorithm ('dbscan', 'hdbscan', 'kmeans')
            n_clusters: Number of clusters for k-means (required if algorithm='kmeans')
            
        Returns:
            Dict containing:
                - 'labels': Cluster labels for each frontier point (-1 for noise in DBSCAN/HDBSCAN)
                - 'n_clusters': Number of clusters found
                - 'cluster_centers': Centroid of each cluster (n_clusters x 3)
                - 'cluster_sizes': Number of points in each cluster
                - 'clustered_points': Dict mapping cluster_id -> points array
        """
        if frontiers is None:
            frontiers = self.frontiers
        
        if len(frontiers) == 0:
            return {
                'labels': np.array([]),
                'n_clusters': 0,
                'cluster_centers': np.empty((0, 3)),
                'cluster_sizes': np.array([]),
                'clustered_points': {}
            }
        
        # Default eps to resolution (adjacent voxels)
        if eps is None:
            eps = self.resolution
        
        # Perform clustering
        if algorithm == 'dbscan':
            clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = clustering.fit_predict(frontiers)
        elif algorithm == 'hdbscan':
            clustering = HDBSCAN(min_cluster_size=min_samples, 
                                        cluster_selection_epsilon=eps)
            labels = clustering.fit_predict(frontiers)
        elif algorithm == 'kmeans':
            if n_clusters is None:
                # Auto-determine number of clusters based on frontier density
                # Use roughly one cluster per (eps)^3 volume
                n_clusters = max(1, int(len(frontiers) / 40))  # Default heuristic
                print(f"Auto-determined n_clusters={n_clusters} for k-means")
            
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(frontiers)
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'dbscan', 'hdbscan', or 'kmeans'")
        
        # Compute cluster statistics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        
        cluster_centers = []
        cluster_sizes = []
        clustered_points = {}
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            
            mask = labels == label
            cluster_points = frontiers[mask]
            
            # Compute centroid
            centroid = np.mean(cluster_points, axis=0)
            cluster_centers.append(centroid)
            cluster_sizes.append(len(cluster_points))
            clustered_points[int(label)] = cluster_points
        
        cluster_centers = np.array(cluster_centers) if cluster_centers else np.empty((0, 3))
        cluster_sizes = np.array(cluster_sizes)
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'cluster_centers': cluster_centers,
            'cluster_sizes': cluster_sizes,
            'clustered_points': clustered_points
        }
    
    def visualize_frontier_clusters(self, cluster_result: Dict = None, 
                                    point_size: float = 5.0,
                                    show_noise: bool = True,
                                    noise_color: List[float] = [0.5, 0.5, 0.5]) -> List:
        """
        Visualize clustered frontiers with different colors for each cluster.
        
        Args:
            cluster_result: Result dict from compute_frontier_clusters() 
                          (if None, will compute clusters first)
            point_size: Size of debug points
            show_noise: Whether to show noise points (label -1)
            noise_color: RGB color for noise points
            
        Returns:
            List of debug handles
        """
        if cluster_result is None:
            cluster_result = self.cluster_frontiers()
        
        debug_handles = []
        n_clusters = cluster_result['n_clusters']
        
        if n_clusters == 0:
            print("No frontier clusters found")
            return debug_handles
        
        # Generate distinct colors for each cluster
        colors = _generate_distinct_colors(n_clusters)
        
        # Visualize each cluster
        for cluster_id, points in cluster_result['clustered_points'].items():
            color = colors[cluster_id % len(colors)]
            handles = DebugPoints(points, points_rgb=color, size=point_size)
            if isinstance(handles, list):
                debug_handles.extend(handles)
            else:
                debug_handles.append(handles)
        
        # Visualize noise points if requested
        if show_noise:
            labels = cluster_result['labels']
            noise_mask = labels == -1
            if np.any(noise_mask):
                noise_points = self.frontiers[noise_mask]
                handles = DebugPoints(noise_points, points_rgb=noise_color, size=point_size * 0.7)
                if isinstance(handles, list):
                    debug_handles.extend(handles)
                else:
                    debug_handles.append(handles)
        
        print(f"Visualized {n_clusters} frontier clusters with {cluster_result['cluster_sizes'].sum()} total points")
        return debug_handles
    
    def cast_ray(self, origin, direction, end, ignoreUnknownCells=False, maxRange=-1.0):
        """Cast a ray in the octree from origin in the given direction up to end point."""
        return self.octree.castRay(origin, direction, end, ignoreUnknownCells=ignoreUnknownCells, maxRange=maxRange)
    
    def cast_unknown_ray(self, origin, direction, maxRange: float = -1.0):
        """
        March along a ray and return the first *known* voxel (free or occupied).
        Returns (hit_center, is_occupied, distance) in world coords,
        or (None, None, None) if no known cell is found within range/bounds.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(direction)
        if n == 0:
            raise ValueError("direction must be non-zero")
        direction = direction / n

        # Precompute bounds (if provided)
        if getattr(self, "bounds", None) is not None:
            roi_min = np.asarray(self.bounds[:3], dtype=np.float64)
            roi_max = np.asarray(self.bounds[3:], dtype=np.float64)
        else:
            roi_min = roi_max = None

        # Determine max distance
        if maxRange is not None and maxRange > 0:
            max_distance = float(maxRange)
        elif roi_min is not None:
            diagonal = np.linalg.norm(roi_max - roi_min)
            max_distance = float(diagonal * 2.0)
        else:
            max_distance = 100.0  # fallback

        # Step size and starting offset
        res = float(self.octree.getResolution())
        step_size = res * 0.5  # or res/3 if you want to be extra safe
        start_dist = res * 0.25
        steps = int(np.ceil((max_distance - start_dist) / step_size))
        if steps <= 0:
            return None, None, None

        max_depth = int(self.octree.getTreeDepth())

        for s in range(steps + 1):
            distance = start_dist + s * step_size
            point = origin + distance * direction

            # Bounds check
            if roi_min is not None:
                if (point < roi_min).any() or (point > roi_max).any():
                    break  # left ROI, stop searching

            # Look up leaf node at finest depth
            node = self.octree.search(point, depth=max_depth)

            if node is not None:
                # We found a known voxel (either free or occupied)
                is_occupied = bool(self.octree.isNodeOccupied(node))

                # Optional: snap to voxel center and recompute distance
                ok, key = self.octree.coordToKeyChecked(point, depth=max_depth)
                if ok:
                    center = np.asarray(self.octree.keyToCoord(key, depth=max_depth), dtype=np.float64)
                    distance = float(np.linalg.norm(center - origin))
                    return center, is_occupied, distance
                else:
                    # Fallback: return sample point
                    return point, is_occupied, float(distance)

        return None, None, None
    
    def is_occupied(self, point: np.ndarray) -> Optional[bool]:
        """
        Check if a point is occupied, free, or unknown in the octree.
        
        Args:
            point: 3D point to check
            
        Returns:
            True if occupied, False if free, None if unknown
        """
        node = self.octree.search(point)
        if node is None:
            return None
        return self.octree.isNodeOccupied(node)
    
    def save(self, filename: str):
        """Save octree to file."""
        self.octree.writeBinary(filename)
        print(f"OctoMap saved to {filename}")
    
    def load(self, filename: str):
        """Load octree from file."""
        self.octree.readBinary(filename)
        print(f"OctoMap loaded from {filename}")


class SemanticOctoMap(OctoMap):
    """
    Extension of OctoMap that includes semantic information (class labels and confidence scores).
    
    Each voxel stores:
    - Occupancy probability (from base OctoMap)
    - Semantic class label c_s(x)
    - Confidence score p_s(x)
    
    Semantic information is updated using max-fusion method when new observations are available.
    """
    
    def __init__(self, bounds: Optional[Union[np.ndarray, RectangleROI]] = None, resolution: float = 0.05):
        super().__init__(bounds, resolution)
        
        # Store semantic information per voxel
        # Key: voxel coordinate tuple (x, y, z), Value: {'label': int, 'confidence': float}
        self.semantic_map: Dict[Tuple[float, float, float], Dict] = {}
        
        # Optional: track class names for visualization
        self.class_names: Dict[int, str] = {}
    
    def _voxel_key(self, point: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a point to a voxel key by snapping to the voxel center.
        
        Args:
            point: 3D point
            
        Returns:
            Tuple of (x, y, z) representing the voxel center
        """
        # Use OctoMap's coordinate to key conversion
        ok, key = self.octree.coordToKeyChecked(point, depth=self.octree.getTreeDepth())
        if ok:
            center = self.octree.keyToCoord(key, depth=self.octree.getTreeDepth())
            # Round to avoid floating point precision issues
            return tuple(np.round(center, decimals=6))
        else:
            # Fallback: manual snapping
            snapped = np.round(point / self.resolution) * self.resolution
            return tuple(np.round(snapped, decimals=6))
    
    def add_semantic_point_cloud(self, point_cloud: np.ndarray, 
                                 labels: np.ndarray, 
                                 confidences: np.ndarray,
                                 sensor_origin: np.ndarray,
                                 max_range: float = -1.0,
                                 lazy_eval: bool = False,
                                 discretize: bool = True,
                                 mismatch_penalty: float = 0.1):
        """
        Add a semantic point cloud to the octree with max-fusion update.
        
        Args:
            point_cloud: Nx3 array of 3D points
            labels: N array of semantic class labels (integers)
            confidences: N array of confidence scores (0.0 to 1.0)
            sensor_origin: 3D point representing sensor position
            max_range: Maximum range for points (-1.0 = no limit)
            lazy_eval: If True, delay updates to inner nodes for efficiency
            discretize: If True, discretize points to voxel centers before insertion
            mismatch_penalty: Penalty factor for label mismatch (default 0.1 = 10%)
            
        Returns:
            update_stats: Dict with statistics about semantic updates
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        confidences = np.asarray(confidences, dtype=np.float64)
        
        if point_cloud.shape[0] != labels.shape[0] or point_cloud.shape[0] != confidences.shape[0]:
            raise ValueError(f"Point cloud, labels, and confidences must have same length")
        
        # First, update the volumetric occupancy (from base OctoMap)
        success_count = self.add_point_cloud(point_cloud, sensor_origin, max_range, lazy_eval, discretize)
        
        # Track statistics
        stats = {
            'total_points': len(point_cloud),
            'new_voxels': 0,
            'updated_same_label': 0,
            'updated_different_label': 0,
            'volumetric_insertions': success_count
        }
        
        # Update semantic information using max-fusion
        for i, point in enumerate(point_cloud):
            voxel_key = self._voxel_key(point)
            new_label = int(labels[i])
            new_confidence = float(confidences[i])
            
            # Clamp confidence to [0, 1]
            new_confidence = np.clip(new_confidence, 0.0, 1.0)
            
            if voxel_key not in self.semantic_map:
                # No prior semantic information - directly assign
                self.semantic_map[voxel_key] = {
                    'label': new_label,
                    'confidence': new_confidence
                }
                stats['new_voxels'] += 1
            else:
                # Max-fusion update
                prev_label = self.semantic_map[voxel_key]['label']
                prev_confidence = self.semantic_map[voxel_key]['confidence']
                
                if new_label == prev_label:
                    # Same label - average the confidence scores
                    updated_confidence = (prev_confidence + new_confidence) / 2.0
                    self.semantic_map[voxel_key] = {
                        'label': new_label,
                        'confidence': updated_confidence
                    }
                    stats['updated_same_label'] += 1
                else:
                    # Different labels - choose the one with higher confidence
                    # Apply penalty to the chosen confidence
                    if new_confidence > prev_confidence:
                        chosen_label = new_label
                        chosen_confidence = new_confidence * (1.0 - mismatch_penalty)
                    else:
                        chosen_label = prev_label
                        chosen_confidence = prev_confidence * (1.0 - mismatch_penalty)
                    
                    self.semantic_map[voxel_key] = {
                        'label': chosen_label,
                        'confidence': chosen_confidence
                    }
                    stats['updated_different_label'] += 1
        
        return stats
    
    def get_semantic_info(self, point: np.ndarray) -> Optional[Dict]:
        """
        Get semantic information for a point.
        
        Args:
            point: 3D point
            
        Returns:
            Dict with 'label' and 'confidence', or None if no semantic info exists
        """
        voxel_key = self._voxel_key(point)
        return self.semantic_map.get(voxel_key, None)
    
    def get_voxels_by_label(self, label: int, min_confidence: float = 0.0) -> np.ndarray:
        """
        Get all voxel positions with a specific semantic label.
        
        Args:
            label: Semantic class label to filter by
            min_confidence: Minimum confidence threshold
            
        Returns:
            Nx3 array of voxel positions
        """
        voxels = []
        for voxel_key, semantic_info in self.semantic_map.items():
            if semantic_info['label'] == label and semantic_info['confidence'] >= min_confidence:
                voxels.append(voxel_key)
        
        return np.array(voxels) if voxels else np.empty((0, 3))
    
    def get_uncertain_voxels(self, max_confidence: float = 0.5) -> np.ndarray:
        """
        Get voxels with low semantic confidence (high uncertainty).
        
        Args:
            max_confidence: Maximum confidence threshold for uncertainty
            
        Returns:
            Nx3 array of uncertain voxel positions
        """
        voxels = []
        for voxel_key, semantic_info in self.semantic_map.items():
            if semantic_info['confidence'] <= max_confidence:
                voxels.append(voxel_key)
        
        return np.array(voxels) if voxels else np.empty((0, 3))
    
    def update_stats(self, verbose: bool = False, max_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute statistics about the octree occupancy (extends base class).
        Also prints semantic statistics if verbose=True.
        
        Args:
            verbose: If True, print statistics to console
            max_points: Maximum number of voxel points to consider for stats
            
        Returns:
            free_points: Nx3 array of free voxel positions
            occupied_points: Mx3 array of occupied voxel positions
        """
        # Call base class method
        free_points, occupied_points = super().update_stats(verbose=False, max_points=max_points)
        
        # Print semantic statistics
        if verbose:
            print(f"Free voxels: {len(free_points)}")
            print(f"Occupied voxels: {len(occupied_points)}")
            print(f"Total visualized: {len(free_points) + len(occupied_points)}")
            print(f"Total tree size: {self.octree.size()}")
            print(f"Semantic voxels: {len(self.semantic_map)}")
            
            # Count voxels per label
            label_counts = {}
            for semantic_info in self.semantic_map.values():
                label = semantic_info['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print("Voxels per semantic label:")
            for label, count in sorted(label_counts.items()):
                label_name = self.class_names.get(label, f"Class {label}")
                print(f"  {label_name}: {count}")
        
        return free_points, occupied_points
    
    def visualize_semantic(self, label: Optional[int] = None,
                          min_confidence: float = 0.0,
                          color: Optional[List[float]] = None,
                          point_size: float = 5.0,
                          max_points: int = 100000) -> List:
        """
        Visualize semantic voxels, optionally filtered by label.
        
        Args:
            label: If specified, only visualize voxels with this label (None = all)
            min_confidence: Minimum confidence threshold
            color: RGB color for visualization (if None, auto-generate per label)
            point_size: Size of debug points
            max_points: Maximum number of points to visualize
            
        Returns:
            List of debug handles
        """
        debug_handles = []
        
        if label is not None:
            # Visualize specific label
            voxels = self.get_voxels_by_label(label, min_confidence)
            if len(voxels) == 0:
                print(f"No voxels found for label {label}")
                return debug_handles
            
            if len(voxels) > max_points:
                voxels = voxels[:max_points]
                print(f"WARN: Limiting visualization to {max_points} points.")
            
            viz_color = color if color is not None else [1, 0, 1]
            handles = DebugPoints(voxels, points_rgb=viz_color, size=point_size)
            if isinstance(handles, list):
                debug_handles.extend(handles)
            else:
                debug_handles.append(handles)
            
            label_name = self.class_names.get(label, f"Class {label}")
            print(f"Visualized {len(voxels)} voxels for {label_name}")
        else:
            # Visualize all labels with different colors
            label_to_voxels = {}
            for voxel_key, semantic_info in self.semantic_map.items():
                if semantic_info['confidence'] >= min_confidence:
                    lbl = semantic_info['label']
                    if lbl not in label_to_voxels:
                        label_to_voxels[lbl] = []
                    label_to_voxels[lbl].append(voxel_key)
            
            if not label_to_voxels:
                print("No semantic voxels to visualize")
                return debug_handles
            
            # Generate colors for each label
            unique_labels = sorted(label_to_voxels.keys())
            colors = _generate_distinct_colors(len(unique_labels))
            
            total_visualized = 0
            for i, lbl in enumerate(unique_labels):
                voxels = np.array(label_to_voxels[lbl])
                if len(voxels) > max_points // len(unique_labels):
                    voxels = voxels[:max_points // len(unique_labels)]
                
                viz_color = colors[i]
                handles = DebugPoints(voxels, points_rgb=viz_color, size=point_size)
                if isinstance(handles, list):
                    debug_handles.extend(handles)
                else:
                    debug_handles.append(handles)
                
                total_visualized += len(voxels)
            
            print(f"Visualized {total_visualized} semantic voxels across {len(unique_labels)} labels")
        
        return debug_handles
    
    def visualize_uncertainty(self, max_confidence: float = 0.5,
                             color: List[float] = [1, 0.5, 0],
                             point_size: float = 5.0,
                             max_points: int = 100000) -> List:
        """
        Visualize uncertain voxels (low confidence scores).
        
        Args:
            max_confidence: Maximum confidence to be considered uncertain
            color: RGB color for uncertain voxels
            point_size: Size of debug points
            max_points: Maximum number of points to visualize
            
        Returns:
            List of debug handles
        """
        uncertain_voxels = self.get_uncertain_voxels(max_confidence)
        
        if len(uncertain_voxels) == 0:
            print("No uncertain voxels found")
            return []
        
        if len(uncertain_voxels) > max_points:
            uncertain_voxels = uncertain_voxels[:max_points]
            print(f"WARN: Limiting visualization to {max_points} points.")
        
        debug_handles = DebugPoints(uncertain_voxels, points_rgb=color, size=point_size)
        print(f"Visualized {len(uncertain_voxels)} uncertain voxels")
        
        if isinstance(debug_handles, list):
            return debug_handles
        else:
            return [debug_handles]
    
    def set_class_names(self, class_names: Dict[int, str]):
        """
        Set human-readable names for semantic class labels.
        
        Args:
            class_names: Dict mapping label integers to class name strings
        """
        self.class_names = class_names
    
    def save_semantic(self, octree_filename: str, semantic_filename: str):
        """
        Save both octree and semantic information to files.
        
        Args:
            octree_filename: Path to save octree binary
            semantic_filename: Path to save semantic map (numpy)
        """
        # Save base octree
        self.save(octree_filename)
        
        # Save semantic map
        np.savez(semantic_filename, 
                 semantic_map=dict(self.semantic_map),
                 class_names=self.class_names)
        print(f"Semantic information saved to {semantic_filename}")
    
    def load_semantic(self, octree_filename: str, semantic_filename: str):
        """
        Load both octree and semantic information from files.
        
        Args:
            octree_filename: Path to octree binary file
            semantic_filename: Path to semantic map file
        """
        # Load base octree
        self.load(octree_filename)
        
        # Load semantic map
        data = np.load(semantic_filename, allow_pickle=True)
        self.semantic_map = data['semantic_map'].item()
        self.class_names = data['class_names'].item()
        print(f"Semantic information loaded from {semantic_filename}")