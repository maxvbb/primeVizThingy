"""Main pipeline for semiprime UMAP visualization."""

import argparse
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path

from . import config
from .generate_semiprimes import generate_semiprimes, compute_basic_metadata
from .compute_residues import compute_residue_vectors
from .compute_difficulty import compute_all_difficulty_metrics
from .run_umap import run_2d_and_3d, save_embeddings
from .visualize import create_interactive_html, create_static_plots, create_3d_visualization


def run_pipeline(
    max_n: int | None = None,
    num_residue_primes: int | None = None,
    dimred_method: str | None = None,
    skip_difficulty: bool = False,
    skip_dimred: bool = False,
    skip_viz: bool = False,
    load_existing: bool = False
):
    """
    Run the full semiprime visualization pipeline.

    Args:
        max_n: Maximum semiprime value (default from config)
        num_residue_primes: Number of primes for residue vectors
        dimred_method: Dimensionality reduction method ('umap' or 'pacmap')
        skip_difficulty: Skip difficulty metric computation
        skip_dimred: Skip dimensionality reduction (use existing embeddings)
        skip_viz: Skip visualization generation
        load_existing: Load existing data instead of regenerating
    """
    if max_n is None:
        max_n = config.MAX_N
    if num_residue_primes is None:
        num_residue_primes = config.NUM_RESIDUE_PRIMES
    if dimred_method is None:
        dimred_method = config.DIMRED_METHOD

    start_time = time.time()

    method_upper = dimred_method.upper()
    print("="*70)
    print(f"SEMIPRIME {method_upper} VISUALIZATION PIPELINE")
    print("="*70)
    print(f"Configuration:")
    print(f"  MAX_N: {max_n:,}")
    print(f"  NUM_RESIDUE_PRIMES: {num_residue_primes}")
    print(f"  DIMRED_METHOD: {dimred_method}")
    if dimred_method.lower() == "umap":
        print(f"  UMAP_N_NEIGHBORS: {config.UMAP_N_NEIGHBORS}")
        print(f"  UMAP_MIN_DIST: {config.UMAP_MIN_DIST}")
        print(f"  UMAP_METRIC: {config.UMAP_METRIC}")
    else:
        print(f"  PACMAP_N_NEIGHBORS: {config.PACMAP_N_NEIGHBORS}")
        print(f"  PACMAP_MN_RATIO: {config.PACMAP_MN_RATIO}")
        print(f"  PACMAP_FP_RATIO: {config.PACMAP_FP_RATIO}")
    print("="*70)

    # Step 1: Generate or load semiprimes
    if load_existing and config.SEMIPRIME_DATA_FILE.exists():
        print("\n[STEP 1] Loading existing semiprime data...")
        df = pd.read_parquet(config.SEMIPRIME_DATA_FILE)
        print(f"Loaded {len(df):,} semiprimes from {config.SEMIPRIME_DATA_FILE}")
    else:
        print("\n[STEP 1] Generating semiprimes...")
        step_start = time.time()

        semiprimes = list(generate_semiprimes(max_n))
        print(f"Generated {len(semiprimes):,} semiprimes in {time.time() - step_start:.1f}s")

        # Compute basic metadata
        print("\n[STEP 2] Computing basic metadata...")
        step_start = time.time()
        metadata = compute_basic_metadata(semiprimes)
        df = pd.DataFrame(metadata)
        print(f"Basic metadata computed in {time.time() - step_start:.1f}s")

        gc.collect()

    # Step 3: Compute difficulty metrics
    if not skip_difficulty:
        if 'fermat_iterations' not in df.columns or not load_existing:
            print("\n[STEP 3] Computing difficulty metrics...")
            step_start = time.time()

            difficulty_metrics = compute_all_difficulty_metrics(
                np.asarray(df['n']),
                np.asarray(df['p']),
                np.asarray(df['q'])
            )

            for col, values in difficulty_metrics.items():
                df[col] = values

            print(f"Difficulty metrics computed in {time.time() - step_start:.1f}s")
            gc.collect()
    else:
        print("\n[STEP 3] Skipping difficulty metrics...")
        # Add placeholder columns if they don't exist
        if 'fermat_iterations' not in df.columns:
            df['fermat_iterations'] = 0
            df['p_minus_1_smooth'] = 0
            df['q_minus_1_smooth'] = 0
            df['min_smoothness'] = 0

    # Save semiprime data
    print(f"\nSaving semiprime data to {config.SEMIPRIME_DATA_FILE}...")
    df.to_parquet(config.SEMIPRIME_DATA_FILE, index=False)
    print(f"Data saved. Shape: {df.shape}")

    # Step 4: Compute residue vectors
    if not skip_dimred:
        print("\n[STEP 4] Computing residue vectors...")
        step_start = time.time()

        residue_matrix = compute_residue_vectors(
            np.asarray(df['n']),
            num_primes=num_residue_primes
        )
        print(f"Residue vectors computed in {time.time() - step_start:.1f}s")
        print(f"Residue matrix shape: {residue_matrix.shape}")

        gc.collect()

        # Step 5: Run dimensionality reduction
        print(f"\n[STEP 5] Running {method_upper} dimensionality reduction...")
        step_start = time.time()

        embedding_2d, embedding_3d = run_2d_and_3d(residue_matrix, method=dimred_method)
        save_embeddings(embedding_2d, embedding_3d)

        print(f"{method_upper} completed in {time.time() - step_start:.1f}s")

        # Clean up large arrays
        del residue_matrix
        gc.collect()
    else:
        print("\n[STEP 4-5] Loading existing embeddings...")
        embedding_2d = np.load(config.UMAP_2D_FILE)
        embedding_3d = np.load(config.UMAP_3D_FILE)
        print(f"Loaded embeddings: 2D={embedding_2d.shape}, 3D={embedding_3d.shape}")

    # Step 6: Generate visualizations
    if not skip_viz:
        print("\n[STEP 6] Generating visualizations...")
        step_start = time.time()

        create_interactive_html(df, embedding_2d)
        create_static_plots(df, embedding_2d)
        create_3d_visualization(df, embedding_3d)

        print(f"Visualizations generated in {time.time() - step_start:.1f}s")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nOutput files:")
    print(f"  {config.SEMIPRIME_DATA_FILE}")
    print(f"  {config.UMAP_2D_FILE}")
    print(f"  {config.UMAP_3D_FILE}")
    print(f"  {config.VISUALIZATION_FILE}")
    print(f"  {config.STATIC_PLOTS_DIR}/")

    return df, embedding_2d, embedding_3d


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Semiprime Dimensionality Reduction Visualization Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--max-n', type=int, default=config.MAX_N,
        help='Maximum semiprime value'
    )
    parser.add_argument(
        '--num-primes', type=int, default=config.NUM_RESIDUE_PRIMES,
        help='Number of primes for residue computation'
    )
    parser.add_argument(
        '--method', type=str, default=config.DIMRED_METHOD,
        choices=['umap', 'pacmap'],
        help='Dimensionality reduction method'
    )
    # UMAP-specific parameters
    parser.add_argument(
        '--umap-neighbors', type=int, default=config.UMAP_N_NEIGHBORS,
        help='UMAP n_neighbors parameter'
    )
    parser.add_argument(
        '--umap-min-dist', type=float, default=config.UMAP_MIN_DIST,
        help='UMAP min_dist parameter'
    )
    parser.add_argument(
        '--umap-metric', type=str, default=config.UMAP_METRIC,
        choices=['euclidean', 'cosine'],
        help='UMAP distance metric'
    )
    # PacMAP-specific parameters
    parser.add_argument(
        '--pacmap-neighbors', type=int, default=config.PACMAP_N_NEIGHBORS,
        help='PacMAP n_neighbors parameter'
    )
    parser.add_argument(
        '--pacmap-mn-ratio', type=float, default=config.PACMAP_MN_RATIO,
        help='PacMAP MN_ratio parameter (ratio of mid-near pairs to neighbor pairs)'
    )
    parser.add_argument(
        '--pacmap-fp-ratio', type=float, default=config.PACMAP_FP_RATIO,
        help='PacMAP FP_ratio parameter (ratio of further pairs to neighbor pairs)'
    )
    # Skip flags
    parser.add_argument(
        '--skip-difficulty', action='store_true',
        help='Skip difficulty metric computation'
    )
    parser.add_argument(
        '--skip-dimred', action='store_true',
        help='Skip dimensionality reduction (load existing embeddings)'
    )
    parser.add_argument(
        '--skip-viz', action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--load-existing', action='store_true',
        help='Load existing semiprime data instead of regenerating'
    )

    args = parser.parse_args()

    # Update config with CLI arguments
    config.DIMRED_METHOD = args.method
    config.UMAP_N_NEIGHBORS = args.umap_neighbors
    config.UMAP_MIN_DIST = args.umap_min_dist
    config.UMAP_METRIC = args.umap_metric
    config.PACMAP_N_NEIGHBORS = args.pacmap_neighbors
    config.PACMAP_MN_RATIO = args.pacmap_mn_ratio
    config.PACMAP_FP_RATIO = args.pacmap_fp_ratio

    run_pipeline(
        max_n=args.max_n,
        num_residue_primes=args.num_primes,
        dimred_method=args.method,
        skip_difficulty=args.skip_difficulty,
        skip_dimred=args.skip_dimred,
        skip_viz=args.skip_viz,
        load_existing=args.load_existing
    )


if __name__ == "__main__":
    main()
