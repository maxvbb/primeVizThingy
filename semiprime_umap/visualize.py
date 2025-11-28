"""Visualization for semiprime UMAP embeddings."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List
from pathlib import Path

from . import config


def create_interactive_html(
    df: pd.DataFrame,
    embedding_2d: np.ndarray,
    output_path: str = None
):
    """
    Create interactive Plotly HTML visualization with dropdown for color schemes.

    Args:
        df: DataFrame with semiprime data and metadata
        embedding_2d: 2D UMAP coordinates
        output_path: Path to save HTML file
    """
    if output_path is None:
        output_path = config.VISUALIZATION_FILE

    print(f"Creating interactive visualization for {len(df):,} points...")

    # Add UMAP coordinates to dataframe
    plot_df = df.copy()
    plot_df['umap_x'] = embedding_2d[:, 0]
    plot_df['umap_y'] = embedding_2d[:, 1]

    # Create hover text
    plot_df['hover_text'] = plot_df.apply(
        lambda row: (
            f"n = {int(row['n']):,}<br>"
            f"Factors: {int(row['p']):,} × {int(row['q']):,}<br>"
            f"log(q/p) = {row['log_ratio']:.3f}<br>"
            f"Fermat iters: {int(row['fermat_iterations']):,}<br>"
            f"Gap: {int(row['factor_gap']):,}"
        ),
        axis=1
    )

    # Define color schemes
    color_configs = {
        'Factor Balance (log q/p)': {
            'column': 'log_ratio',
            'colorscale': 'RdYlBu_r',
            'title': 'log(q/p) - 0 = balanced'
        },
        'Fermat Difficulty': {
            'column': 'fermat_iterations',
            'colorscale': 'Viridis',
            'title': 'Fermat iterations'
        },
        'Smaller Factor Index π(p)': {
            'column': 'smaller_factor_idx',
            'colorscale': 'Plasma',
            'title': 'Prime index of p'
        },
        'Larger Factor Index π(q)': {
            'column': 'larger_factor_idx',
            'colorscale': 'Plasma',
            'title': 'Prime index of q'
        },
        'Magnitude (bit length)': {
            'column': 'bit_length',
            'colorscale': 'Cividis',
            'title': 'Bit length of n'
        },
        'Factor Gap (q - p)': {
            'column': 'factor_gap',
            'colorscale': 'Hot',
            'title': 'q - p'
        },
        'p-1 Smoothness': {
            'column': 'p_minus_1_smooth',
            'colorscale': 'Turbo',
            'title': 'Largest prime factor of (p-1)'
        },
        'q-1 Smoothness': {
            'column': 'q_minus_1_smooth',
            'colorscale': 'Turbo',
            'title': 'Largest prime factor of (q-1)'
        },
        'Min Smoothness': {
            'column': 'min_smoothness',
            'colorscale': 'Turbo',
            'title': 'min(p-1 smooth, q-1 smooth)'
        },
    }

    # Create figure with first color scheme
    first_scheme = list(color_configs.keys())[0]
    first_config = color_configs[first_scheme]

    # Sample for performance if dataset is large
    max_points = 100_000
    if len(plot_df) > max_points:
        print(f"Sampling {max_points:,} points for interactive visualization...")
        sample_idx = np.random.choice(len(plot_df), max_points, replace=False)
        sample_idx = np.sort(sample_idx)
        plot_df_sampled = plot_df.iloc[sample_idx].reset_index(drop=True)
    else:
        plot_df_sampled = plot_df

    # Create traces for each color scheme
    traces = []
    for i, (scheme_name, cfg) in enumerate(color_configs.items()):
        trace = go.Scattergl(
            x=plot_df_sampled['umap_x'],
            y=plot_df_sampled['umap_y'],
            mode='markers',
            marker=dict(
                size=3,
                color=plot_df_sampled[cfg['column']],
                colorscale=cfg['colorscale'],
                colorbar=dict(title=cfg['title']),
                opacity=0.6
            ),
            text=plot_df_sampled['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            name=scheme_name,
            visible=(i == 0)  # Only first trace visible initially
        )
        traces.append(trace)

    # Create dropdown buttons
    buttons = []
    for i, scheme_name in enumerate(color_configs.keys()):
        visibility = [j == i for j in range(len(color_configs))]
        buttons.append(dict(
            label=scheme_name,
            method='update',
            args=[{'visible': visibility}]
        ))

    # Create figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text=f'Semiprime UMAP Visualization ({len(plot_df_sampled):,} points)',
            x=0.5
        ),
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.15,
                yanchor='top'
            )
        ],
        annotations=[
            dict(
                text='Color by:',
                showarrow=False,
                x=0,
                y=1.12,
                xref='paper',
                yref='paper',
                align='left'
            )
        ],
        hovermode='closest',
        template='plotly_white',
        width=1200,
        height=800
    )

    # Add range slider for magnitude filtering
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Max bit length: "},
            pad={"t": 50},
            steps=[
                dict(
                    label=str(bl),
                    method="update",
                    args=[{"visible": [True] * len(traces)}]
                )
                for bl in range(10, 25)
            ],
            visible=False  # Hide for now - complex to implement properly
        )]
    )

    # Save to HTML
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Saved interactive visualization to {output_path}")


def create_static_plots(
    df: pd.DataFrame,
    embedding_2d: np.ndarray,
    output_dir: str = None
):
    """Create static PNG plots with different colorings."""
    if output_dir is None:
        output_dir = config.STATIC_PLOTS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample for static plots
    max_points = 50_000
    if len(df) > max_points:
        print(f"Sampling {max_points:,} points for static plots...")
        sample_idx = np.random.choice(len(df), max_points, replace=False)
        df_sample = df.iloc[sample_idx]
        coords_sample = embedding_2d[sample_idx]
    else:
        df_sample = df
        coords_sample = embedding_2d

    x, y = coords_sample[:, 0], coords_sample[:, 1]

    plot_configs = [
        ('colored_by_ratio.png', 'log_ratio', 'Factor Balance: log(q/p)', 'RdYlBu_r'),
        ('colored_by_fermat_difficulty.png', 'fermat_iterations', 'Fermat Factorization Iterations', 'viridis'),
        ('colored_by_smaller_factor.png', 'smaller_factor_idx', 'Smaller Factor Index π(p)', 'plasma'),
        ('colored_by_magnitude.png', 'bit_length', 'Semiprime Magnitude (bit length)', 'cividis'),
    ]

    for filename, column, title, cmap in plot_configs:
        print(f"Creating {filename}...")

        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            x, y,
            c=df_sample[column],
            cmap=cmap,
            s=1,
            alpha=0.5
        )

        plt.colorbar(scatter, ax=ax, label=column)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Semiprime UMAP: {title}')

        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(plot_configs)} static plots to {output_dir}")


def create_3d_visualization(
    df: pd.DataFrame,
    embedding_3d: np.ndarray,
    output_path: str = None
):
    """Create 3D interactive visualization."""
    if output_path is None:
        output_path = config.OUTPUT_DIR / "visualization_3d.html"

    print(f"Creating 3D visualization...")

    # Sample for performance
    max_points = 50_000
    if len(df) > max_points:
        sample_idx = np.random.choice(len(df), max_points, replace=False)
        df_sample = df.iloc[sample_idx]
        coords_sample = embedding_3d[sample_idx]
    else:
        df_sample = df
        coords_sample = embedding_3d

    fig = go.Figure(data=[
        go.Scatter3d(
            x=coords_sample[:, 0],
            y=coords_sample[:, 1],
            z=coords_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=df_sample['log_ratio'],
                colorscale='RdYlBu_r',
                colorbar=dict(title='log(q/p)'),
                opacity=0.6
            ),
            text=[
                f"n={int(row['n']):,}<br>{int(row['p']):,} × {int(row['q']):,}"
                for _, row in df_sample.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>'
        )
    ])

    fig.update_layout(
        title='3D Semiprime UMAP',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1000,
        height=800
    )

    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Saved 3D visualization to {output_path}")


if __name__ == "__main__":
    # Test with dummy data
    print("Testing visualization with dummy data...")

    n_points = 1000
    dummy_df = pd.DataFrame({
        'n': np.random.randint(100, 10000, n_points),
        'p': np.random.randint(2, 100, n_points),
        'q': np.random.randint(10, 200, n_points),
        'log_ratio': np.random.rand(n_points) * 3,
        'smaller_factor_idx': np.random.randint(1, 100, n_points),
        'larger_factor_idx': np.random.randint(10, 500, n_points),
        'bit_length': np.random.randint(7, 14, n_points),
        'factor_gap': np.random.randint(0, 1000, n_points),
        'fermat_iterations': np.random.randint(1, 1000, n_points),
        'p_minus_1_smooth': np.random.randint(2, 100, n_points),
        'q_minus_1_smooth': np.random.randint(2, 200, n_points),
        'min_smoothness': np.random.randint(2, 100, n_points),
    })

    dummy_embedding = np.random.randn(n_points, 2).astype(np.float32)
    dummy_embedding_3d = np.random.randn(n_points, 3).astype(np.float32)

    print("Creating test visualizations...")
    create_interactive_html(dummy_df, dummy_embedding, config.OUTPUT_DIR / "test_viz.html")
    print("Done!")
