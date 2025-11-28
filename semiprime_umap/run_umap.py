"""Dimensionality reduction (UMAP/PacMAP) for semiprime residue vectors."""

import numpy as np
import umap
import pacmap
from typing import Tuple
from pathlib import Path
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

    embedding: np.ndarray = np.asarray(reducer.fit_transform(feature_matrix))

    print(f"UMAP complete. Output shape: {embedding.shape}")
    return embedding.astype(np.float32)


def run_pacmap_reduction(
    feature_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int | None = None,
    mn_ratio: float | None = None,
    fp_ratio: float | None = None,
    random_state: int | None = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Run PacMAP dimensionality reduction on feature matrix.

    Args:
        feature_matrix: Input features (n_samples, n_features)
        n_components: Output dimensions (2 or 3)
        n_neighbors: Number of neighbors for local structure
        mn_ratio: Ratio of mid-near pairs to neighbor pairs
        fp_ratio: Ratio of further pairs to neighbor pairs
        random_state: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Embedded coordinates (n_samples, n_components)
    """
    if n_neighbors is None:
        n_neighbors = config.PACMAP_N_NEIGHBORS
    if mn_ratio is None:
        mn_ratio = config.PACMAP_MN_RATIO
    if fp_ratio is None:
        fp_ratio = config.PACMAP_FP_RATIO
    if random_state is None:
        random_state = config.PACMAP_RANDOM_STATE

    n_samples, n_features = feature_matrix.shape
    print(f"Running PacMAP: {n_samples:,} samples, {n_features} features -> {n_components}D")
    print(f"  Parameters: n_neighbors={n_neighbors}, MN_ratio={mn_ratio}, FP_ratio={fp_ratio}")

    reducer = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=mn_ratio,
        FP_ratio=fp_ratio,
        random_state=random_state,
        verbose=verbose
    )

    embedding: np.ndarray = np.asarray(reducer.fit_transform(feature_matrix))

    print(f"PacMAP complete. Output shape: {embedding.shape}")
    return embedding.astype(np.float32)


def run_reduction(
    feature_matrix: np.ndarray,
    n_components: int = 2,
    method: str | None = None,
    **kwargs
) -> np.ndarray:
    """
    Run dimensionality reduction using the specified method.

    Args:
        feature_matrix: Input features (n_samples, n_features)
        n_components: Output dimensions (2 or 3)
        method: Reduction method ('umap' or 'pacmap'), defaults to config
        **kwargs: Method-specific parameters

    Returns:
        Embedded coordinates (n_samples, n_components)
    """
    if method is None:
        method = config.DIMRED_METHOD

    method = method.lower()
    if method == "umap":
        return run_umap_reduction(feature_matrix, n_components=n_components, **kwargs)
    elif method == "pacmap":
        return run_pacmap_reduction(feature_matrix, n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}. Use 'umap' or 'pacmap'.")


def run_2d_and_3d(
    feature_matrix: np.ndarray,
    method: str | None = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run dimensionality reduction for both 2D and 3D embeddings.

    Args:
        feature_matrix: Input features (n_samples, n_features)
        method: Reduction method ('umap' or 'pacmap'), defaults to config
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (embedding_2d, embedding_3d)
    """
    if method is None:
        method = config.DIMRED_METHOD

    method_name = method.upper()

    print("\n" + "="*60)
    print(f"Computing 2D {method_name} embedding...")
    print("="*60)
    embedding_2d = run_reduction(feature_matrix, n_components=2, method=method, **kwargs)

    # Force garbage collection between runs
    gc.collect()

    print("\n" + "="*60)
    print(f"Computing 3D {method_name} embedding...")
    print("="*60)
    embedding_3d = run_reduction(feature_matrix, n_components=3, method=method, **kwargs)

    return embedding_2d, embedding_3d


def run_umap_2d_and_3d(
    feature_matrix: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UMAP for both 2D and 3D embeddings.

    Deprecated: Use run_2d_and_3d(method='umap') instead.

    Returns:
        Tuple of (embedding_2d, embedding_3d)
    """
    return run_2d_and_3d(feature_matrix, method="umap", **kwargs)


def save_embeddings(
    embedding_2d: np.ndarray,
    embedding_3d: np.ndarray,
    path_2d: str | Path | None = None,
    path_3d: str | Path | None = None
):
    """Save embeddings to numpy files."""
    save_path_2d: str | Path = path_2d if path_2d is not None else config.UMAP_2D_FILE
    save_path_3d: str | Path = path_3d if path_3d is not None else config.UMAP_3D_FILE

    np.save(save_path_2d, embedding_2d)
    np.save(save_path_3d, embedding_3d)

    print(f"Saved 2D embedding to {save_path_2d}")
    print(f"Saved 3D embedding to {save_path_3d}")


def load_embeddings(
    path_2d: str | Path | None = None,
    path_3d: str | Path | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings from numpy files."""
    load_path_2d: str | Path = path_2d if path_2d is not None else config.UMAP_2D_FILE
    load_path_3d: str | Path = path_3d if path_3d is not None else config.UMAP_3D_FILE

    embedding_2d = np.load(load_path_2d)
    embedding_3d = np.load(load_path_3d)

    return embedding_2d, embedding_3d


if __name__ == "__main__":
    # Test with random data
    import sys

    # Simulate residue vectors
    n_samples = 1000
    n_features = 200
    test_data = np.random.rand(n_samples, n_features).astype(np.float32)

    # Test method from command line or default to both
    methods = sys.argv[1:] if len(sys.argv) > 1 else ["umap", "pacmap"]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()} with random data...")
        print(f"{'='*60}")

        embedding_2d = run_reduction(test_data, n_components=2, method=method)
        print(f"\n2D {method.upper()} embedding shape: {embedding_2d.shape}")
        print(f"2D embedding range: x=[{embedding_2d[:,0].min():.2f}, {embedding_2d[:,0].max():.2f}], "
              f"y=[{embedding_2d[:,1].min():.2f}, {embedding_2d[:,1].max():.2f}]")
