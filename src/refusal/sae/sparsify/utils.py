import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_effect_sizes(results):
    """
    Compute Cohen's d for each latent feature based on harmful vs harmless representations.

    Parameters:
        results: dict
            Dictionary with keys "harmful_reps" and "harmless_reps" (numpy arrays of shape [num_prompts, latent_dim]).

    Returns:
        dict with keys:
          - "cohens_d": numpy array of shape [latent_dim] with raw Cohen's d values.
          - "df": Pandas DataFrame summarizing the effect sizes.
    """
    harmful_reps = results["harmful_reps"]  # shape: [num_harmful, latent_dim]
    harmless_reps = results["harmless_reps"]  # shape: [num_harmless, latent_dim]

    latent_dim = harmful_reps.shape[1]
    cohens_d = []

    for dim in range(latent_dim):
        group1 = harmful_reps[:, dim]
        group2 = harmless_reps[:, dim]
        diff = group1.mean() - group2.mean()
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = diff / pooled_std if pooled_std != 0 else 0
        cohens_d.append(d)

    cohens_d = np.array(cohens_d)
    df = pd.DataFrame(
        {
            "Latent Dimension": np.arange(latent_dim),
            "Cohen's d": cohens_d,
            "Absolute d": np.abs(cohens_d),
        }
    ).sort_values("Absolute d", ascending=False)

    return {"cohens_d": cohens_d, "df": df}


def build_comparison_df(mean_harmful, mean_harmless, diff):
    latent_dim = len(diff)
    df = pd.DataFrame(
        {
            "Latent Dimension": np.arange(latent_dim),
            "Mean Harmful": mean_harmful,
            "Mean Harmless": mean_harmless,
            "Difference": diff,
            "Absolute Difference": np.abs(diff),
        }
    )
    df = df.sort_values("Absolute Difference", ascending=False)
    return df


def visualize_latent_differences(harmful_reps_list, harmless_reps_list, diff_list):
    """
    Visualize heatmaps for harmful and harmless representations, and their differences, across multiple layers.

    Args:
        harmful_reps_list: List of numpy arrays, each containing harmful prompt representations for a layer.
        harmless_reps_list: List of numpy arrays, each containing harmless prompt representations for a layer.
        diff_list: List of numpy arrays, each containing the per-dimension difference (harmful - harmless) for a layer.
    """
    num_layers = len(harmful_reps_list)
    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 4 * num_layers))

    for i in range(num_layers):
        harmful_reps = harmful_reps_list[i]
        harmless_reps = harmless_reps_list[i]
        diff = diff_list[i]

        ax_row = axes[i] if num_layers > 1 else axes  # Handle the case of a single layer

        im0 = ax_row[0].imshow(harmful_reps, cmap="viridis", aspect="auto")
        ax_row[0].set_title(f"Harmful Prompts (Layer {i+1})")
        ax_row[0].set_xlabel("Latent Dimension")
        ax_row[0].set_ylabel("Prompt Index")
        fig.colorbar(im0, ax=ax_row[0])

        im1 = ax_row[1].imshow(harmless_reps, cmap="viridis", aspect="auto")
        ax_row[1].set_title(f"Harmless Prompts (Layer {i+1})")
        ax_row[1].set_xlabel("Latent Dimension")
        fig.colorbar(im1, ax=ax_row[1])

        im2 = ax_row[2].imshow(diff.reshape(1, -1), cmap="coolwarm", aspect="auto")
        ax_row[2].set_title(f"Mean Difference (Layer {i+1})\n(Harmful - Harmless)")
        ax_row[2].set_xlabel("Latent Dimension")
        ax_row[2].set_yticks([])
        fig.colorbar(im2, ax=ax_row[2])

    plt.tight_layout()
    plt.show()
