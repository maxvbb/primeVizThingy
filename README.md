# Semiprime UMAP Visualization

Interactive visualization that embeds semiprimes into 2D/3D space using UMAP based on their residue vectors, then analyzes whether clustering correlates with hidden factor structure.

## Overview

This project:
1. Generates all semiprimes n = p × q where p ≤ q and n < 10,000,000
2. Computes residue feature vectors: `[n % 2, n % 3, n % 5, ..., n % p_k]` normalized to [0, 1)
3. Applies UMAP dimensionality reduction to embed semiprimes in 2D/3D
4. Visualizes with various color schemes to explore factor structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:
```bash
python -m semiprime_umap.main
```

With options:
```bash
python -m semiprime_umap.main --max-n 1000000 --num-primes 100
```

### Command-line Options

- `--max-n`: Maximum semiprime value (default: 10,000,000)
- `--num-primes`: Number of primes for residue computation (default: 200)
- `--umap-neighbors`: UMAP n_neighbors parameter (default: 15)
- `--umap-min-dist`: UMAP min_dist parameter (default: 0.1)
- `--umap-metric`: Distance metric: 'euclidean' or 'cosine'
- `--skip-difficulty`: Skip Fermat/smoothness computation
- `--skip-umap`: Use existing embeddings
- `--skip-viz`: Skip visualization generation
- `--load-existing`: Load existing semiprime data

## Output Files

```
outputs/
├── semiprime_data.parquet    # Full dataset with metadata
├── umap_2d.npy               # 2D UMAP coordinates
├── umap_3d.npy               # 3D UMAP coordinates
├── visualization.html        # Interactive Plotly scatter plot
├── visualization_3d.html     # 3D interactive visualization
└── static_plots/
    ├── colored_by_ratio.png
    ├── colored_by_fermat_difficulty.png
    ├── colored_by_smaller_factor.png
    └── colored_by_magnitude.png
```

## Metadata Fields

For each semiprime, the following metadata is computed:

| Field | Description |
|-------|-------------|
| `log_ratio` | log(q/p) — 0 means balanced factors |
| `smaller_factor_idx` | π(p), prime index of smaller factor |
| `larger_factor_idx` | π(q), prime index of larger factor |
| `bit_length` | Number of bits in n |
| `factor_gap` | q - p |
| `fermat_iterations` | Fermat factorization iterations (capped at 10,000) |
| `p_minus_1_smooth` | Largest prime factor of (p-1) |
| `q_minus_1_smooth` | Largest prime factor of (q-1) |

## Project Structure

```
semiprime_umap/
├── __init__.py
├── config.py              # Configuration parameters
├── generate_semiprimes.py # Generate semiprimes and metadata
├── compute_residues.py    # Build residue feature vectors
├── compute_difficulty.py  # Fermat iterations, smoothness
├── run_umap.py           # UMAP dimensionality reduction
├── visualize.py          # Generate plots and interactive HTML
└── main.py               # Pipeline orchestration
```

## Configuration

Edit `semiprime_umap/config.py` to change defaults:

```python
MAX_N = 10_000_000
NUM_RESIDUE_PRIMES = 200
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
FERMAT_MAX_ITERATIONS = 10_000
```
