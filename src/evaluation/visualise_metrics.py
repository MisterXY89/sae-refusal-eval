from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class DatasetMetrics:
    context_length: float
    self_repetition: float
    ngram_diversity: float
    compression_ratio: float
    perplexity: Optional[float]
    perplexity_gap: Optional[float]


@dataclass
class PerformanceMetrics:
    mmlu: float
    sparsity: float
    reconstruction_error: float
    refusal_rate: float
    over_refusal: float


def plot_heatmap_and_multiples(
    metrics_list: List[DatasetMetrics],
    performance_list: Optional[List[PerformanceMetrics]] = None,
):
    # Define metric names
    dataset_metric_names = [
        "context_length",
        "self_repetition",
        "ngram_diversity",
        "compression_ratio",
        "perplexity",
        "perplexity_gap",
    ]

    performance_metric_names = [
        "mmlu",
        "sparsity",
        "reconstruction_error",
        "refusal_rate",
        "over_refusal",
    ]

    # Prepare data for DataFrame
    data = []
    for i, metrics in enumerate(metrics_list):
        for name in dataset_metric_names:
            value = getattr(metrics, name, None)
            data.append(
                {
                    "Metric": name,
                    "Value": value if value is not None else 0,  # Default to 0 for None values
                    "Dataset": f"Dataset {i+1}",
                }
            )

    if performance_list:
        for i, perf in enumerate(performance_list):
            for name in performance_metric_names:
                value = getattr(perf, name, None)
                data.append(
                    {
                        "Metric": name,
                        "Value": (
                            value if value is not None else 0
                        ),  # Default to 0 for None values
                        "Dataset": f"Performance {i+1}",
                    }
                )

    df = pd.DataFrame(data)

    # Create a heatmap
    heatmap_data = df.pivot("Metric", "Dataset", "Value").fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Metrics Heatmap", fontsize=16)
    plt.ylabel("Metrics", fontsize=12)
    plt.xlabel("Datasets", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Create small multiples
    unique_metrics = df["Metric"].unique()
    n_metrics = len(unique_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), squeeze=False)
    for i, metric in enumerate(unique_metrics):
        row, col = divmod(i, n_cols)
        metric_data = df[df["Metric"] == metric]
        sns.barplot(x="Dataset", y="Value", data=metric_data, ax=axes[row, col])
        axes[row, col].set_title(metric, fontsize=14)
        axes[row, col].set_ylabel("Value", fontsize=12)
        axes[row, col].set_xlabel("Dataset", fontsize=12)

    # Remove empty subplots
    for j in range(i + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


# # Example usage
# dataset1 = DatasetMetrics(5.0, 0.2, 0.8, 1.5, 30.0, 0.1)
# dataset2 = DatasetMetrics(4.8, 0.25, 0.75, 1.6, 28.5, 0.12)
# performance1 = PerformanceMetrics(70.0, 0.1, 0.05, 0.02, 0.01)
# performance2 = PerformanceMetrics(72.0, 0.08, 0.04, 0.015, 0.02)
# plot_heatmap_and_multiples([dataset1, dataset2], [performance1, performance2])
