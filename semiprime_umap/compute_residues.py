"""Compute residue feature vectors for semiprimes."""

import numpy as np
from sympy import primerange
from typing import Optional
import gc

from . import config


def get_first_k_primes(k: int) -> np.ndarray:
    """Get the first k prime numbers."""
    # Generate more primes than needed, then take first k
    # The k-th prime is approximately k * ln(k) for large k
    upper_bound = max(int(k * (np.log(k) + np.log(np.log(k + 1)) + 2)), 1000)
    primes = list(primerange(2, upper_bound))
    return np.array(primes[:k], dtype=np.int64)


def compute_residue_vectors(
    n_values: np.ndarray,
    num_primes: int = None,
    batch_size: int = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute normalized residue vectors for an array of integers.

    For each n, computes [n % p_1, n % p_2, ..., n % p_k] / [p_1, p_2, ..., p_k]
    where p_i are the first k primes.

    Args:
        n_values: Array of integers to compute residues for
        num_primes: Number of primes to use (default from config)
        batch_size: Process in batches to manage memory
        normalize: If True, divide residues by modulus to get values in [0, 1)

    Returns:
        numpy array of shape (len(n_values), num_primes) with residue vectors
    """
    if num_primes is None:
        num_primes = config.NUM_RESIDUE_PRIMES
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    primes = get_first_k_primes(num_primes)
    n_samples = len(n_values)

    print(f"Computing residue vectors for {n_samples:,} values using {num_primes} primes")
    print(f"Prime range: {primes[0]} to {primes[-1]}")

    # Pre-allocate output array
    dtype = np.float32 if normalize else np.int32
    residue_matrix = np.zeros((n_samples, num_primes), dtype=dtype)

    # Process in batches to manage memory
    num_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({start:,} to {end:,})")

        batch_n = n_values[start:end]

        # Compute residues for this batch
        # Using broadcasting: batch_n[:, None] % primes[None, :]
        batch_residues = batch_n[:, None] % primes[None, :]

        if normalize:
            # Normalize by dividing by the modulus
            residue_matrix[start:end] = batch_residues / primes[None, :]
        else:
            residue_matrix[start:end] = batch_residues

        # Force garbage collection periodically
        if batch_idx % 50 == 0:
            gc.collect()

    return residue_matrix


def compute_residue_vectors_sparse(
    n_values: np.ndarray,
    num_primes: int = None,
    threshold: float = 0.1
) -> 'scipy.sparse.csr_matrix':
    """
    Compute sparse residue vectors (optional, for very large datasets).

    Only stores values above threshold to save memory.
    Note: Most residue vectors won't be sparse, so this may not help much.
    """
    from scipy import sparse

    dense = compute_residue_vectors(n_values, num_primes, normalize=True)

    # Only keep values above threshold (sparse representation)
    sparse_matrix = sparse.csr_matrix(dense * (dense > threshold))

    return sparse_matrix


if __name__ == "__main__":
    # Test with small example
    test_n = np.array([6, 15, 21, 35, 77], dtype=np.int64)
    primes = get_first_k_primes(10)
    print(f"First 10 primes: {primes}")

    residues = compute_residue_vectors(test_n, num_primes=10)
    print(f"\nResidue vectors (normalized):")
    print(f"Shape: {residues.shape}")
    for i, n in enumerate(test_n):
        print(f"  {n}: {residues[i][:5]}... (first 5 values)")
