"""Compute factorization difficulty metrics for semiprimes."""

import numpy as np
from numba import njit, prange
from sympy import factorint
from typing import Optional
import math

from . import config


@njit
def fermat_iterations_single(n: int, max_iter: int) -> int:
    """
    Count Fermat factorization iterations for a single semiprime.

    Fermat's method: find a such that a^2 - n is a perfect square.
    Start with a = ceil(sqrt(n)) and increment until b^2 = a^2 - n.

    Returns number of iterations, or max_iter + 1 if not found.
    """
    if n % 2 == 0:
        return 1  # Trivially factorable

    # Start with ceil(sqrt(n))
    a = int(math.ceil(math.sqrt(n)))

    for i in range(max_iter):
        b2 = a * a - n
        b = int(math.sqrt(b2))

        if b * b == b2:
            # Found factorization: n = (a+b)(a-b)
            return i + 1

        a += 1

    return max_iter + 1  # Mark as "hard"


@njit(parallel=True)
def compute_fermat_iterations_parallel(
    n_values: np.ndarray,
    max_iter: int
) -> np.ndarray:
    """
    Compute Fermat factorization iterations for all semiprimes in parallel.

    Uses numba's parallel execution for speed.
    """
    n_samples = len(n_values)
    iterations = np.zeros(n_samples, dtype=np.int32)

    for i in prange(n_samples):
        iterations[i] = fermat_iterations_single(n_values[i], max_iter)

    return iterations


def compute_fermat_iterations(
    n_values: np.ndarray,
    max_iter: int | None = None
) -> np.ndarray:
    """
    Compute Fermat factorization iterations for semiprimes.

    Args:
        n_values: Array of semiprimes
        max_iter: Maximum iterations before marking as "hard"

    Returns:
        Array of iteration counts (max_iter + 1 means exceeded limit)
    """
    if max_iter is None:
        max_iter = config.FERMAT_MAX_ITERATIONS

    print(f"Computing Fermat iterations for {len(n_values):,} semiprimes (max={max_iter:,})...")

    iterations = compute_fermat_iterations_parallel(n_values, max_iter)

    hard_count = np.sum(iterations > max_iter)
    print(f"  {hard_count:,} semiprimes exceeded max iterations ({100*hard_count/len(n_values):.2f}%)")

    return iterations


def largest_prime_factor(n: int) -> int:
    """Find the largest prime factor of n."""
    if n <= 1:
        return 1

    factors = factorint(n)
    if not factors:
        return 1
    return max(factors.keys())


def compute_smoothness(
    p_values: np.ndarray,
    q_values: np.ndarray,
    batch_size: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute smoothness of (p-1) and (q-1) for each semiprime.

    Smoothness = largest prime factor of (p-1) or (q-1).
    Lower smoothness means the number is B-smooth for smaller B,
    making it more vulnerable to Pollard's p-1 attack.

    Returns:
        Tuple of (p_minus_1_smooth, q_minus_1_smooth) arrays
    """
    n_samples = len(p_values)
    print(f"Computing smoothness for {n_samples:,} semiprimes...")

    p_smooth = np.zeros(n_samples, dtype=np.int64)
    q_smooth = np.zeros(n_samples, dtype=np.int64)

    # Cache for repeated values
    cache = {}

    for i in range(n_samples):
        if i % batch_size == 0 and i > 0:
            print(f"  Processed {i:,}/{n_samples:,} ({100*i/n_samples:.1f}%)")

        p_minus_1 = int(p_values[i]) - 1
        q_minus_1 = int(q_values[i]) - 1

        # Check cache first
        if p_minus_1 not in cache:
            cache[p_minus_1] = largest_prime_factor(p_minus_1)
        if q_minus_1 not in cache:
            cache[q_minus_1] = largest_prime_factor(q_minus_1)

        p_smooth[i] = cache[p_minus_1]
        q_smooth[i] = cache[q_minus_1]

    return p_smooth, q_smooth


def compute_all_difficulty_metrics(
    n_values: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray
) -> dict:
    """
    Compute all difficulty-related metrics.

    Returns dict with:
        - fermat_iterations: Fermat method iteration count
        - p_minus_1_smooth: Largest prime factor of (p-1)
        - q_minus_1_smooth: Largest prime factor of (q-1)
        - min_smoothness: min(p_minus_1_smooth, q_minus_1_smooth)
    """
    fermat_iter = compute_fermat_iterations(n_values)
    p_smooth, q_smooth = compute_smoothness(p_values, q_values)

    return {
        'fermat_iterations': fermat_iter,
        'p_minus_1_smooth': p_smooth,
        'q_minus_1_smooth': q_smooth,
        'min_smoothness': np.minimum(p_smooth, q_smooth),
    }


if __name__ == "__main__":
    # Test with examples
    test_cases = [
        (15, 3, 5),      # Easy: 3*5
        (21, 3, 7),      # Easy: 3*7
        (143, 11, 13),   # Medium: 11*13, close factors
        (8051, 83, 97),  # Harder: larger close factors
    ]

    print("Testing Fermat iterations:")
    for n, p, q in test_cases:
        iters = fermat_iterations_single(n, 10000)
        print(f"  {n} = {p} × {q}: {iters} iterations")

    print("\nTesting smoothness:")
    for n, p, q in test_cases:
        p_s = largest_prime_factor(p - 1)
        q_s = largest_prime_factor(q - 1)
        print(f"  {n} = {p} × {q}: p-1={p-1} smooth={p_s}, q-1={q-1} smooth={q_s}")
