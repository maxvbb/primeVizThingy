"""Generate semiprimes and their metadata."""

import numpy as np
from sympy import primerange, primepi
from typing import Generator
import math

from . import config


def get_primes_up_to(n: int) -> np.ndarray:
    """Generate all primes up to n."""
    return np.array(list(primerange(2, n + 1)), dtype=np.int64)


def generate_semiprimes(max_n: int | None = None) -> Generator[tuple[int, int, int], None, None]:
    """
    Generate all semiprimes n = p * q where p <= q and n < max_n.

    Yields tuples of (n, p, q).
    """
    if max_n is None:
        max_n = config.MAX_N

    # We need primes up to sqrt(max_n) for the smaller factor
    # and up to max_n / 2 for the larger factor (when p = 2)
    max_small_prime = int(math.isqrt(max_n))
    max_large_prime = max_n // 2

    primes = get_primes_up_to(max_large_prime)
    small_primes = primes[primes <= max_small_prime]

    print(f"Generating semiprimes up to {max_n:,}")
    print(f"Number of candidate small primes (p): {len(small_primes):,}")
    print(f"Number of total primes for q: {len(primes):,}")

    count = 0
    for i, p in enumerate(small_primes):
        if i % 100 == 0:
            print(f"Processing p = {p} ({i}/{len(small_primes)}), semiprimes so far: {count:,}")

        # Find the index where primes >= p
        start_idx = np.searchsorted(primes, p)

        # q can go from p up to max_n // p
        max_q = max_n // p
        end_idx = np.searchsorted(primes, max_q, side='right')

        for q in primes[start_idx:end_idx]:
            n = p * q
            if n < max_n:
                count += 1
                yield (int(n), int(p), int(q))

    print(f"Total semiprimes generated: {count:,}")


def compute_basic_metadata(semiprimes: list[tuple[int, int, int]]) -> dict:
    """
    Compute basic metadata for semiprimes.

    Returns dict with arrays for each metadata field.
    """
    n_arr = np.array([s[0] for s in semiprimes], dtype=np.int64)
    p_arr = np.array([s[1] for s in semiprimes], dtype=np.int64)
    q_arr = np.array([s[2] for s in semiprimes], dtype=np.int64)

    print("Computing basic metadata...")

    # log_ratio: log(q / p) — 0 means balanced, high means unbalanced
    log_ratio = np.log(q_arr / p_arr)

    # bit_length of n
    bit_length = np.array([int(n).bit_length() for n in n_arr], dtype=np.int32)

    # factor_gap: q - p
    factor_gap = q_arr - p_arr

    # Prime indices (using sympy's primepi for accuracy)
    print("Computing prime indices (this may take a while)...")
    unique_primes = np.unique(np.concatenate([p_arr, q_arr]))
    prime_to_idx = {int(p): int(primepi(p)) for p in unique_primes}

    smaller_factor_idx = np.array([prime_to_idx[int(p)] for p in p_arr], dtype=np.int32)
    larger_factor_idx = np.array([prime_to_idx[int(q)] for q in q_arr], dtype=np.int32)

    return {
        'n': n_arr,
        'p': p_arr,
        'q': q_arr,
        'log_ratio': log_ratio.astype(np.float32),
        'smaller_factor_idx': smaller_factor_idx,
        'larger_factor_idx': larger_factor_idx,
        'bit_length': bit_length,
        'factor_gap': factor_gap,
    }


if __name__ == "__main__":
    # Test with smaller range
    test_max = 1000
    semiprimes = list(generate_semiprimes(test_max))
    print(f"\nFirst 10 semiprimes under {test_max}:")
    for s in semiprimes[:10]:
        print(f"  {s[0]} = {s[1]} × {s[2]}")

    metadata = compute_basic_metadata(semiprimes)
    print(f"\nMetadata computed for {len(semiprimes)} semiprimes")
