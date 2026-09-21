"""Plot per-edge selection frequencies for the fixed-support scaling experiment.

Each cell is the fraction of seed runs in which an edge was selected. True
edges are shown in red and false edges in blue; the saturation is the fraction.
"""

import argparse
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "experiments" / "output" / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import numpy as np


SAMPLE_DIR_RE = re.compile(r"num_samples_(\d+)$")
SEED_DIR_RE = re.compile(r"seed_(\d+)$")


def _sample_directories(input_dir):
    directories = []
    for path in Path(input_dir).iterdir():
        match = SAMPLE_DIR_RE.fullmatch(path.name)
        if path.is_dir() and match:
            directories.append((int(match.group(1)), path))
    if not directories:
        raise FileNotFoundError(f"No num_samples_* directories found under {input_dir}")
    return sorted(directories)


def _seed_json_files(sample_dir):
    files = []
    for path in sample_dir.iterdir():
        match = SEED_DIR_RE.fullmatch(path.name)
        json_path = path / "selection_plateau_bootstrap.json"
        if path.is_dir() and match and json_path.is_file():
            files.append((int(match.group(1)), json_path))
    return sorted(files)


def _load_truth(input_dir, sample_dir):
    files = _seed_json_files(sample_dir)
    if not files:
        raise FileNotFoundError(f"No selection JSON files found under {sample_dir}")
    first = json.loads(files[0][1].read_text())
    npz_path = Path(first["input"])
    if not npz_path.is_absolute() and not npz_path.is_file():
        # Resolve paths written relative to the repository root.
        npz_path = Path(input_dir).parents[2] / npz_path
    with np.load(npz_path, allow_pickle=False) as data:
        n = int(np.asarray(data["n"]))
        true_edges = {tuple(map(int, edge)) for edge in data["lambda_star_support_edges"]}
    return n, true_edges


def _edge_label(edge):
    return f"({edge[0] + 1},{edge[1] + 1})"


def collect_frequencies(input_dir):
    input_dir = Path(input_dir)
    sample_dirs = _sample_directories(input_dir)
    n, true_edges = _load_truth(input_dir, sample_dirs[0][1])
    candidate_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    true_order = [edge for edge in candidate_edges if edge in true_edges]
    false_order = [edge for edge in candidate_edges if edge not in true_edges]
    edges = true_order + false_order

    frequencies = np.zeros((len(edges), len(sample_dirs)), dtype=float)
    counts = np.zeros_like(frequencies, dtype=int)
    denominators = []
    for column, (sample_count, sample_dir) in enumerate(sample_dirs):
        files = _seed_json_files(sample_dir)
        if not files:
            raise FileNotFoundError(f"No selection JSON files found under {sample_dir}")
        denominators.append(len(files))
        for row, edge in enumerate(edges):
            counts[row, column] = sum(
                edge in {tuple(map(int, selected)) for selected in json.loads(path.read_text())["selected_edges"]}
                for _, path in files
            )
        frequencies[:, column] = counts[:, column] / len(files)
    return edges, true_edges, [sample_count for sample_count, _ in sample_dirs], frequencies, counts, denominators


def plot_heatmap(input_dir, output_path):
    edges, true_edges, sample_counts, frequencies, counts, denominators = collect_frequencies(input_dir)
    row_count, column_count = frequencies.shape
    colors = np.ones((row_count, column_count, 4), dtype=float)
    for row, edge in enumerate(edges):
        cmap = plt.get_cmap("Reds" if edge in true_edges else "Blues")
        colors[row] = cmap(frequencies[row])

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.imshow(colors, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(column_count), [f"{value:,}" for value in sample_counts])
    ax.set_yticks(range(row_count), [_edge_label(edge) for edge in edges])
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Edge (true edges first)")
    ax.set_title("Edge selection frequency across 10-seed runs (n=4)")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(-0.5, column_count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
    ax.grid(which="minor", color="0.75", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    if true_edges and len(true_edges) < row_count:
        ax.axhline(len(true_edges) - 0.5, color="0.25", linewidth=1.8)

    for row in range(row_count):
        for column in range(column_count):
            ax.text(column, row, f"{counts[row, column]}/{denominators[column]}",
                    ha="center", va="center", fontsize=9, color="black")

    norm = Normalize(vmin=0, vmax=1)
    true_bar = ScalarMappable(norm=norm, cmap="Reds")
    false_bar = ScalarMappable(norm=norm, cmap="Blues")
    true_bar.set_array([])
    false_bar.set_array([])
    # Keep the bars clearly separated and label each one directly above its
    # own scale, so the text cannot be mistaken for the neighboring colorbar.
    cbar_true = fig.colorbar(true_bar, ax=ax, fraction=0.035, pad=0.04)
    cbar_false = fig.colorbar(false_bar, ax=ax, fraction=0.035, pad=0.18)
    cbar_true.ax.set_title("TRUE", fontsize=9, pad=8)
    cbar_false.ax.set_title("FALSE", fontsize=9, pad=8)
    ax.legend(handles=[Patch(facecolor="tab:red", label="True edge"), Patch(facecolor="tab:blue", label="False edge")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.text(0.01, 0.01, "Cell text: selected seeds / available seeds; saturation is fixed to 0–1.", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 0.92, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        "--input-dir",
        dest="input_dir",
        default=PROJECT_ROOT / "experiments" / "output" / "fixed_support_scaling_n4",
        type=Path,
        help="Experiment output directory (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output image path (default: <input_dir>/edge_recovery_heatmap.png)")
    args = parser.parse_args()
    output = args.output or args.input_dir / "edge_recovery_heatmap.png"
    plot_heatmap(args.input_dir, output)
    print(f"Saved heatmap to {output}")


if __name__ == "__main__":
    main()
