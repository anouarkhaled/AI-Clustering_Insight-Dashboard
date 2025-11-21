"""
Dimensionality reduction module.
Encapsulates PCA, t-SNE, UMAP.
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


def reduce_dim(X, method='PCA', n_components=2, random_state=42, perplexity=30):
    """
    Reduce dimensionality using PCA, t-SNE, or UMAP.
    
    Args:
        X: numpy array or pd.DataFrame of shape (n_samples, n_features)
        method: 'PCA', 't-SNE', or 'UMAP'
        n_components: number of output dimensions (default 2)
        random_state: random seed
        perplexity: perplexity for t-SNE (default 30)
    
    Returns:
        proj: numpy array of shape (n_samples, n_components)
    """
    if method == 'PCA':
        pca = PCA(n_components=n_components, random_state=random_state)
        proj = pca.fit_transform(X)
        return proj
    elif method == 't-SNE':
        tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        proj = tsne.fit_transform(X)
        return proj
    elif method == 'UMAP' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        proj = reducer.fit_transform(X)
        return proj
    else:
        # fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        proj = pca.fit_transform(X)
        return proj
