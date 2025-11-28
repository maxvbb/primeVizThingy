"""Configuration parameters for semiprime UMAP visualization."""

from pathlib import Path

# Data generation
MAX_N = 10_000_000  # Maximum semiprime value
NUM_RESIDUE_PRIMES = 200  # Number of primes for residue computation

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"  # Can also use "cosine"
UMAP_RANDOM_STATE = 42

# Difficulty computation
FERMAT_MAX_ITERATIONS = 10_000

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STATIC_PLOTS_DIR = OUTPUT_DIR / "static_plots"

# Output files
SEMIPRIME_DATA_FILE = OUTPUT_DIR / "semiprime_data.parquet"
UMAP_2D_FILE = OUTPUT_DIR / "umap_2d.npy"
UMAP_3D_FILE = OUTPUT_DIR / "umap_3d.npy"
VISUALIZATION_FILE = OUTPUT_DIR / "visualization.html"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Batch processing
BATCH_SIZE = 100_000  # For memory-efficient processing
