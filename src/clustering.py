"""
Clustering algorithms module.
Encapsulates KMeans, DBSCAN, GMM, OPTICS, KMedoids and scoring metrics.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Optional imports
try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except Exception:
    KMEDOIDS_AVAILABLE = False


def run_clustering(X, alg='KMeans', **kwargs):
    """
    Run clustering algorithm on data X.
    
    Args:
        X: numpy array or pd.DataFrame of shape (n_samples, n_features)
        alg: algorithm name ('KMeans', 'DBSCAN', 'GMM', 'OPTICS', 'KMedoids')
        **kwargs: algorithm-specific hyperparameters
                  - KMeans/GMM/KMedoids: n_clusters (default 3)
                  - DBSCAN/OPTICS: eps, min_samples
    
    Returns:
        labels: numpy array of cluster assignments
    """
    if alg == 'KMeans':
        k = kwargs.get('n_clusters', 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    elif alg == 'DBSCAN':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    elif alg == 'GMM':
        k = kwargs.get('n_clusters', 3)
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X)
    elif alg == 'OPTICS':
        min_samples = kwargs.get('min_samples', 5)
        model = OPTICS(min_samples=min_samples)
        labels = model.fit_predict(X)
    elif alg == 'KMedoids' and KMEDOIDS_AVAILABLE:
        k = kwargs.get('n_clusters', 3)
        model = KMedoids(n_clusters=k, random_state=42, method='pam')
        labels = model.fit_predict(X)
    else:
        # fallback to KMeans
        k = kwargs.get('n_clusters', 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    return labels


def cluster_scores(X, labels):
    """
    Compute clustering quality metrics.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        labels: numpy array of cluster assignments
    
    Returns:
        scores: dict with keys 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'n_clusters', 'sizes'
    """
    scores = {}
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])  # ignore noise label for DBSCAN
    
    try:
        if n_clusters > 1:
            scores['silhouette'] = float(silhouette_score(X, labels))
        else:
            scores['silhouette'] = None
    except Exception:
        scores['silhouette'] = None
    
    try:
        scores['davies_bouldin'] = float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else None
    except Exception:
        scores['davies_bouldin'] = None
    
    try:
        scores['calinski_harabasz'] = float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else None
    except Exception:
        scores['calinski_harabasz'] = None
    
    scores['n_clusters'] = int(n_clusters)
    
    # cluster sizes
    vals, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip([int(v) for v in vals], [int(c) for c in counts]))
    scores['sizes'] = sizes
    
    return scores
