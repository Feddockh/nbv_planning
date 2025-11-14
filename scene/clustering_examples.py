"""
Example usage of different frontier clustering algorithms.

Demonstrates DBSCAN, HDBSCAN, and K-Means clustering options.
"""

import numpy as np


def example_dbscan_clustering(octomap):
    """
    DBSCAN clustering - density-based, good for irregular shapes.
    Automatically finds number of clusters, marks low-density points as noise.
    """
    print("\n=== DBSCAN Clustering ===")
    
    # Find frontiers
    frontiers = octomap.find_frontiers()
    print(f"Found {len(frontiers)} frontier voxels")
    
    # Cluster using DBSCAN
    result = octomap.cluster_frontiers(
        algorithm='dbscan',
        eps=2*octomap.resolution,  # Max distance between points in same cluster
        min_samples=5               # Min points to form a cluster
    )
    
    print(f"Found {result['n_clusters']} clusters")
    print(f"Cluster sizes: {result['cluster_sizes']}")
    print(f"Noise points: {np.sum(result['labels'] == -1)}")
    
    # Visualize
    octomap.visualize_frontier_clusters(result)
    
    return result


def example_hdbscan_clustering(octomap):
    """
    HDBSCAN clustering - hierarchical density-based, more robust than DBSCAN.
    Good for varying density clusters.
    """
    print("\n=== HDBSCAN Clustering ===")
    
    frontiers = octomap.find_frontiers()
    
    # Cluster using HDBSCAN
    result = octomap.cluster_frontiers(
        algorithm='hdbscan',
        min_samples=5
    )
    
    print(f"Found {result['n_clusters']} clusters")
    print(f"Cluster sizes: {result['cluster_sizes']}")
    
    # Visualize
    octomap.visualize_frontier_clusters(result)
    
    return result


def example_kmeans_clustering(octomap):
    """
    K-Means clustering - partition-based, requires specifying number of clusters.
    Good when you know how many regions you want to explore.
    """
    print("\n=== K-Means Clustering ===")
    
    frontiers = octomap.find_frontiers()
    
    # Option 1: Specify number of clusters
    result = octomap.cluster_frontiers(
        algorithm='kmeans',
        n_clusters=5  # Explicitly set number of clusters
    )
    
    print(f"Found {result['n_clusters']} clusters (as specified)")
    print(f"Cluster sizes: {result['cluster_sizes']}")
    
    # Visualize
    octomap.visualize_frontier_clusters(result)
    
    return result


def example_kmeans_auto(octomap):
    """
    K-Means with automatic cluster number determination.
    Uses heuristic: roughly 1 cluster per 20 frontier points.
    """
    print("\n=== K-Means with Auto Cluster Count ===")
    
    frontiers = octomap.find_frontiers()
    
    # Option 2: Let it auto-determine number of clusters
    result = octomap.cluster_frontiers(
        algorithm='kmeans'
        # n_clusters not specified - will auto-determine
    )
    
    print(f"Auto-determined {result['n_clusters']} clusters")
    print(f"Cluster sizes: {result['cluster_sizes']}")
    
    # Visualize
    octomap.visualize_frontier_clusters(result)
    
    return result


def compare_all_algorithms(octomap):
    """
    Compare all three clustering algorithms on the same frontier set.
    """
    print("\n" + "="*80)
    print("COMPARING CLUSTERING ALGORITHMS")
    print("="*80)
    
    frontiers = octomap.find_frontiers()
    print(f"Total frontier voxels: {len(frontiers)}\n")
    
    # DBSCAN
    dbscan_result = octomap.cluster_frontiers(
        algorithm='dbscan',
        eps=2*octomap.resolution,
        min_samples=5
    )
    print(f"DBSCAN:  {dbscan_result['n_clusters']} clusters, "
          f"{np.sum(dbscan_result['labels'] == -1)} noise points")
    
    # HDBSCAN
    hdbscan_result = octomap.cluster_frontiers(
        algorithm='hdbscan',
        min_samples=5
    )
    print(f"HDBSCAN: {hdbscan_result['n_clusters']} clusters, "
          f"{np.sum(hdbscan_result['labels'] == -1)} noise points")
    
    # K-Means (auto)
    kmeans_result = octomap.cluster_frontiers(
        algorithm='kmeans'
    )
    print(f"K-Means: {kmeans_result['n_clusters']} clusters (auto-determined), "
          f"0 noise points")
    
    # K-Means (manual)
    kmeans_manual = octomap.cluster_frontiers(
        algorithm='kmeans',
        n_clusters=8
    )
    print(f"K-Means: {kmeans_manual['n_clusters']} clusters (manually set), "
          f"0 noise points")
    
    print("\n" + "="*80)
    print("Algorithm Characteristics:")
    print("="*80)
    print("DBSCAN:   + Handles irregular shapes, + Auto cluster count, - Sensitive to density")
    print("HDBSCAN:  + More robust than DBSCAN, + Varying densities, - Slower")
    print("K-Means:  + Fast, + Even cluster sizes, - Must specify k, - Assumes spherical clusters")
    print("="*80)


def use_cluster_centers_for_nbv(octomap):
    """
    Example of using cluster centers for NBV planning.
    """
    print("\n=== Using Cluster Centers for NBV Planning ===")
    
    # Cluster frontiers
    result = octomap.cluster_frontiers(
        algorithm='kmeans',
        n_clusters=6
    )
    
    # Get cluster centers
    cluster_centers = result['cluster_centers']
    
    print(f"Found {len(cluster_centers)} exploration targets:")
    for i, center in enumerate(cluster_centers):
        cluster_size = result['cluster_sizes'][i]
        print(f"  Cluster {i}: center at {center}, size={cluster_size} voxels")
    
    # Now you can generate viewpoints around each cluster center
    # Example:
    # for center in cluster_centers:
    #     viewpoints = generate_viewpoints_for_target(center, radius=0.5)
    #     scores = [compute_information_gain(vp, octomap) for vp in viewpoints]
    #     best_viewpoint = viewpoints[np.argmax(scores)]
    #     move_to_viewpoint(best_viewpoint)
    
    return cluster_centers


# ============================================================================
# Quick Reference
# ============================================================================

QUICK_REFERENCE = """
FRONTIER CLUSTERING QUICK REFERENCE
====================================

1. DBSCAN (Density-Based):
   result = octomap.cluster_frontiers(
       algorithm='dbscan',
       eps=2*octomap.resolution,  # Max distance
       min_samples=5              # Min cluster size
   )
   
   Pros: Finds arbitrary shapes, auto cluster count
   Cons: Sensitive to density variations
   Use when: Frontiers have clear spatial separation

2. HDBSCAN (Hierarchical Density-Based):
   result = octomap.cluster_frontiers(
       algorithm='hdbscan',
       min_samples=5
   )
   
   Pros: Handles varying densities better than DBSCAN
   Cons: Slower, more complex
   Use when: Frontiers have varying densities

3. K-Means (Partition-Based):
   
   Option A - Manual cluster count:
   result = octomap.cluster_frontiers(
       algorithm='kmeans',
       n_clusters=5  # You specify
   )
   
   Option B - Auto cluster count:
   result = octomap.cluster_frontiers(
       algorithm='kmeans'
       # Auto-determines based on frontier count
   )
   
   Pros: Fast, predictable cluster count
   Cons: Must specify k (or use heuristic), assumes spherical clusters
   Use when: You know how many exploration regions you want

Result Dictionary Contains:
---------------------------
- labels: Array of cluster IDs for each frontier point
- n_clusters: Number of clusters found
- cluster_centers: Centroid of each cluster
- cluster_sizes: Number of points per cluster
- clustered_points: Dict mapping cluster_id -> points array

Visualization:
--------------
octomap.visualize_frontier_clusters(result)
"""


if __name__ == "__main__":
    print(QUICK_REFERENCE)
