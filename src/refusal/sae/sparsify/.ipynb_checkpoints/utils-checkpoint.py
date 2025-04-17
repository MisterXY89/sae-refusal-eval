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
    ).sort_values("Cohen's d", ascending=False)

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


def visualize_latent_differences(harmful, harmless, diff):
    """
    Visualize heatmaps for harmful and harmless representations, and their differences.

    Can be called either as
        visualize_latent_differences(
            harmful_reps: 2D array (N_prompts × latent_dim),
            harmless_reps: 2D array,
            diff: 1D array (latent_dim,)
        )
    or as
        visualize_latent_differences(
            harmful_list: [2D array, 2D array, ...],
            harmless_list: [...],
            diff_list: [...]
        )
    """
    if isinstance(harmful, np.ndarray) and harmful.ndim == 2:
        harmful_list, harmless_list, diff_list = [harmful], [harmless], [diff]
    else:
        harmful_list, harmless_list, diff_list = harmful, harmless, diff

    num_layers = len(harmful_list)    
    fig, axes = plt.subplots(num_layers, 3, figsize=(35, 5 * num_layers)) 
    if num_layers == 1:
        axes = axes[None, :]

    for i in range(num_layers):
        harm = harmful_list[i]
        safe = harmless_list[i]
        d = diff_list[i]
        ax_row = axes[i]

        # 1) Harmful
        im0 = ax_row[0].imshow(
            harm,
            cmap="PRGn",
            aspect="auto",
            interpolation="nearest"      # no smoothing between pixels
        )
        ax_row[0].set_title(f"Harmful Prompts (Layer {i})")
        ax_row[0].set_xlabel("Latent Dimension")
        ax_row[0].set_ylabel("Prompt Index")
        # draw a tick at every integer row
        ax_row[0].set_yticks(np.arange(harm.shape[0]))
        ax_row[0].set_yticklabels(np.arange(harm.shape[0]))
        fig.colorbar(im0, ax=ax_row[0])

        # 2) Harmless
        im1 = ax_row[1].imshow(
            safe,
            cmap="PRGn",
            aspect="auto",
            interpolation="nearest"
        )
        ax_row[1].set_title(f"Harmless Prompts (Layer {i})")
        ax_row[1].set_xlabel("Latent Dimension")
        ax_row[1].set_yticks(np.arange(safe.shape[0]))
        ax_row[1].set_yticklabels(np.arange(safe.shape[0]))
        fig.colorbar(im1, ax=ax_row[1])

        # 3) Difference (1×latent_dim)
        im2 = ax_row[2].imshow(
            d.reshape(1, -1),
            cmap="PRGn", #coolwarm
            aspect="auto",
            interpolation="nearest"
        )
        ax_row[2].set_title(f"Mean Difference (Layer {i})\n(Harmful − Harmless)")
        ax_row[2].set_xlabel("Latent Dimension")
        ax_row[2].set_yticks([0])
        ax_row[2].set_yticklabels(["Δ"])
        ax_row[2].set_box_aspect(0.1)
        fig.colorbar(im2, ax=ax_row[2])

    plt.tight_layout()
    plt.show()