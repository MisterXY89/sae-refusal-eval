import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def compute_effect_sizes(results, layer):
    """
    Given the dict output from analyze_latent_reps, compute Cohen's d for each latent dimension.
    Returns a numpy array cohens_d of shape [latent_dim].
    """
    harmful = results[layer]["harmful_reps"]   # shape [num_harmful, latent_dim]
    harmless = results[layer]["harmless_reps"] # shape [num_harmless, latent_dim]
    n_h, n_dim = harmful.shape
    n_c, _     = harmless.shape

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
    return cohens_d

def identify_top_features(results, layers, N=5):
    """
    Compute effect sizes and return the top N dimensions most associated
    with harmful (positive d) and harmless (negative d) prompts.
    """
    records = []
    for layer_idx, layer in enumerate(layers):
        stats = results[layer_idx]
        # print(stats)
        harmful = stats["stats"]["harmful_reps"]
        harmless = stats["stats"]["harmless_reps"]
        # compute Cohen's d
        n_h, latent_dim = harmful.shape
        n_c, _ = harmless.shape
        mu_h = harmful.mean(axis=0)
        mu_c = harmless.mean(axis=0)
        var_h = harmful.var(axis=0, ddof=1)
        var_c = harmless.var(axis=0, ddof=1)
        pooled = np.sqrt(((n_h-1)*var_h + (n_c-1)*var_c) / (n_h + n_c - 2))
        pooled[pooled == 0] = np.nan
        d = (mu_h - mu_c) / pooled
        d = np.nan_to_num(d)
        # collect per-dimension
        for dim, val in enumerate(d):
            records.append({
                "layer": layer,
                "latent_dim": dim,
                "Cohen's d": val
            })

    df_all = pd.DataFrame(records)
    # top harmful = largest positive d
    top_harmful = df_all.nlargest(N, "Cohen's d").reset_index(drop=True)
    # top harmless = most negative d
    top_harmless = df_all.nsmallest(N, "Cohen's d").reset_index(drop=True)

    return {
        "top_harmful": top_harmful,
        "top_harmless": top_harmless
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
    plt.savefig(f"/home/tilman.kerl/mech-interp/src/results/visualizations/feature_id_{sae_name}")
    plt.show()