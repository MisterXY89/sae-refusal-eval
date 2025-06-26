
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace, arange
from matplotlib.ticker import MaxNLocator
import pandas as pd

def compute_effect_sizes(results, layer, epsilon: float = 1e-8):
    """
    Given the dict output from analyze_latent_reps, compute Cohen's d for each latent dimension.
    Returns a numpy array cohens_d of shape [latent_dim].
    """
    harmful = results[layer]["stats"]["harmful_reps"]   # shape [num_harmful, latent_dim]
    harmless = results[layer]["stats"]["harmless_reps"] # shape [num_harmless, latent_dim]
    n_h, n_dim = harmful.shape
    n_c, _ = harmless.shape

    # means and variances
    mu_h = harmful.mean(axis=0)
    mu_c = harmless.mean(axis=0)
    var_h = harmful.var(axis=0, ddof=1)
    var_c = harmless.var(axis=0, ddof=1)

    # pooled std
    pooled_std = np.sqrt(((n_h-1)*var_h + (n_c-1)*var_c) / (n_h + n_c - 2))
    # avoid zero‐division
    pooled_std[pooled_std == 0] = np.nan

    cohens_d = (mu_h - mu_c) / pooled_std
    cohens_d = np.nan_to_num(cohens_d)  # replace nan→0

    fisher = (mu_h - mu_c)**2 / (var_h + var_c + epsilon)
    
    return cohens_d, fisher


def identify_top_features(
    results,
    layers,
    N: int = 5,
    rank_metric: str = "cohens_d"
):
    """
    Compute effect sizes (Cohen's d & Fisher) per layer, then
    return the top N dimensions most associated with harmful
    (positive) and harmless (negative) along the chosen metric.

    Args:
      results: output dict from analyze_latent_reps
      layers: list of layer identifiers
      N: number of top dims to return
      metric: one of "cohens_d" or "fisher"
    """
    # validate
    rank_metric = rank_metric.lower()
    if rank_metric not in ("cohens_d", "fisher"):
        raise ValueError("metric must be 'cohens_d' or 'fisher'")

    recs = []
    for idx, layer in enumerate(layers):
        d, fisher = compute_effect_sizes(results, idx)
        for dim, (dv, fv) in enumerate(zip(d, fisher)):
            recs.append({
                "layer": layer,
                "latent_dim": dim,
                "cohens_d": dv,
                "fisher": fv
            })

    df = pd.DataFrame(recs)
    col = rank_metric
    return {
        "top_harmful": df.nlargest(N, col).reset_index(drop=True),
        "top_harmless": df.nsmallest(N, col).reset_index(drop=True)
    }


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


def visualize_latent_differences(harmful, harmless, diff, sae_name):
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
        locator = MaxNLocator(nbins=5, integer=True)
        
        harm = harmful_list[i]
        safe = harmless_list[i]
        d = diff_list[i]
        ax_row = axes[i]

        # 1) Harmful
        im0 = ax_row[0].imshow(
            harm,
            # cmap="PRGn",
            cmap="Blues", 
            aspect="auto",
            interpolation="nearest"      # no smoothing between pixels
        )
        ax_row[0].set_title(f"Harmful Prompts (Layer {i})")
        ax_row[0].set_xlabel("Latent Dimension")
        ax_row[0].set_ylabel("Prompt Index")
        # draw a tick at every integer row
        # ax_row[0].set_yticks(np.arange(harm.shape[0]))
        # ax_row[0].set_yticklabels(np.arange(harm.shape[0]))
        ax_row[0].yaxis.set_major_locator(locator)
        fig.colorbar(im0, ax=ax_row[0])

        # 2) Harmless
        im1 = ax_row[1].imshow(
            safe,
            # cmap="PRGn",
            cmap="Blues", 
            aspect="auto",
            interpolation="nearest"
        )
        ax_row[1].set_title(f"Harmless Prompts (Layer {i})")
        ax_row[1].set_xlabel("Latent Dimension")
        # ax_row[1].set_yticks(np.arange(safe.shape[0]))
        # ax_row[1].set_yticklabels(np.arange(safe.shape[0]))
        ax_row[1].yaxis.set_major_locator(locator)
        fig.colorbar(im1, ax=ax_row[1])

        # 3) Difference (1×latent_dim)
        im2 = ax_row[2].imshow(
            d.reshape(1, -1),
            cmap="Blues", #coolwarm
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
    plt.savefig(f"/home/tilman.kerl/mech-interp/src/results/visualizations/feature_id_{sae_name}", dpi=150, bbox_inches='tight')
    plt.show()