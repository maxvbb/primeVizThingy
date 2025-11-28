"""UMAP dimensionality reduction for semiprime residue vectors."""

import numpy as np
import umap
from typing import Optional, Tuple
import gc

from . import config


def run_umap_reduction(
    feature_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    metric: str | None = None,
    random_state: int | None = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction on feature matrix.

    Args:
        feature_matrix: Input features (n_samples, n_features)
        n_components: Output dimensions (2 or 3)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric ('euclidean' or 'cosine')
        random_state: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Embedded coordinates (n_samples, n_components)
    """
    if n_neighbors is None:
        n_neighbors = config.UMAP_N_NEIGHBORS
    if min_dist is None:
        min_dist = config.UMAP_MIN_DIST
    if metric is None:
        metric = config.UMAP_METRIC
    if random_state is None:
        random_state = config.UMAP_RANDOM_STATE

    n_samples, n_features = feature_matrix.shape
    print(f"Running UMAP: {n_samples:,} samples, {n_features} features -> {n_components}D")
    print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=verbose,
        low_memory=True  # Use less memory for large datasets
    )

    embedding = reducer.fit_transform(feature_matrix)

    print(f"UMAP complete. Output shape: {embedding.shape}")
    return embedding.astype(np.float32)


def run_umap_2d_and_3d(
    feature_matrix: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UMAP for both 2D and 3D embeddings.

    Returns:
        Tuple of (embedding_2d, embedding_3d)
    """
    print("\n" + "="*60)
    print("Computing 2D UMAP embedding...")
    print("="*60)
    embedding_2d = run_umap_reduction(feature_matrix, n_components=2, **kwargs)

    # Force garbage collection between runs
    gc.collect()

    print("\n" + "="*60)
    print("Computing 3D UMAP embedding...")
    print("="*60)
    embedding_3d = run_umap_reduction(feature_matrix, n_components=3, **kwargs)

    return embedding_2d, embedding_3d


def save_embeddings(
    embedding_2d: np.ndarray,
    embedding_3d: np.ndarray,
    path_2d: str | None = None,
    path_3d: str | None = None
):
    """Save UMAP embeddings to numpy files."""
    if path_2d is None:
        path_2d = config.UMAP_2D_FILE
    if path_3d is None:
        path_3d = config.UMAP_3D_FILE

    np.save(path_2d, embedding_2d)
    np.save(path_3d, embedding_3d)

    print(f"Saved 2D embedding to {path_2d}")
    print(f"Saved 3D embedding to {path_3d}")


def load_embeddings(
    path_2d: str | None = None,
    path_3d: str | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load UMAP embeddings from numpy files."""
    if path_2d is None:
        path_2d = config.UMAP_2D_FILE
    if path_3d is None:
        path_3d = config.UMAP_3D_FILE

    embedding_2d = np.load(path_2d)
    embedding_3d = np.load(path_3d)

    return embedding_2d, embedding_3d


if __name__ == "__main__":
    # Test with random data
    print("Testing UMAP with random data...")

    # Simulate residue vectors
    n_samples = 1000
    n_features = 200
    test_data = np.random.rand(n_samples, n_features).astype(np.float32)

    embedding_2d = run_umap_reduction(test_data, n_components=2)
    print(f"\n2D embedding shape: {embedding_2d.shape}")
    print(f"2D embedding range: x=[{embedding_2d[:,0].min():.2f}, {embedding_2d[:,0].max():.2f}], "
          f"y=[{embedding_2d[:,1].min():.2f}, {embedding_2d[:,1].max():.2f}]")
