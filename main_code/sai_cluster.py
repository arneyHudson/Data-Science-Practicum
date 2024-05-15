import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS


def fuzzy_clustering(dataset, num_clusters, fuzziness=2, max_iterations=100, tolerance=1e-5):
    """
    Apply fuzzy clustering to a dataset.

    **NOTE** For fuzzy clustering these lines may need to be added before calling this method
        -pip install importlib_metadata==4.2.0
        -pip install scikit-fuzzy
       
    Parameters:
        dataset (numpy.ndarray): The dataset to cluster.
        num_clusters (int): The number of clusters to create.
        fuzziness (float): The fuzziness parameter (default=2).
        max_iterations (int): The maximum number of iterations for the clustering algorithm (default=100).
        tolerance (float): The convergence criterion (default=1e-5).

    Returns:
       centroids (numpy.ndarray): The centroids of the clusters.
       membership (numpy.ndarray): The membership values for each data point.
    """
        
    # Transpose the dataset so that each row represents a data point
    dataset = dataset.T

    # Initialize the centroids randomly
    centroids = np.random.rand(num_clusters, dataset.shape[1])
    
    # Apply fuzzy c-means clustering
    centroids, membership, _, _, _, _, _ = fuzz.cluster.cmeans(
        dataset, num_clusters, fuzziness, error=tolerance, maxiter=max_iterations)

    return centroids, membership
    
    
def subspace_clustering(dataset, num_clusters, feature_combinations):
    """
    Apply subspace clustering to a dataset using KMeans.
    
    Parameters:
        dataset (numpy.ndarray): The dataset to cluster.
        num_clusters (int): The number of clusters to create.
        feature_combinations (list of lists): The list of feature combinations to use for clustering.

    Returns:
        clusters (list of tuples): List of tuples containing centroids and labels for each feature combination.
    """
        
    clusters = []
    for features in feature_combinations:
        # Select only the features in the feature_combination
        selected_data = dataset[:, features]
    
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(selected_data)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
    
        clusters.append((centroids, labels))

    return clusters


def birch_clustering(dataset, threshold, branching_factor):
    """
    Apply BIRCH clustering to a dataset.
    Parameters:
        dataset (numpy.ndarray): The dataset to cluster.
        threshold (float): The branching factor threshold.
        branching_factor (int): The maximum number of subclusters in each leaf node.
    Returns:
        labels (numpy.ndarray): Array containing the cluster labels for each data point.
    """
    
    birch = Birch(threshold=threshold, branching_factor=branching_factor)
    labels = birch.fit_predict(dataset)
    return labels
    

def optics_clustering(dataset, min_samples, xi=0.05, min_cluster_size=0.1):
    """
    Apply OPTICS clustering to a dataset.

    Parameters:
        dataset (numpy.ndarray): The dataset to cluster.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary (default=0.05).
        min_cluster_size (float or int): The minimum number of samples in a cluster. If float, it 
            represents the fraction of the number of samples (default=0.1).

    Returns:
        labels (numpy.ndarray): Array containing the cluster labels for each data point.
        reachability (numpy.ndarray): Reachability distances of each point.
        optics (OPTICS object): Fitted OPTICS object.
    """
    
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics.fit(dataset)
    labels = optics.labels_
    reachability = optics.reachability_[optics.ordering_]
    return labels, reachability, optics