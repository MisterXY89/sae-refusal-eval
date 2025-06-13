import wandb
import pandas as pd
import matplotlib.pyplot as plt

def fetch_and_report_wandb(run_path: str):
    """
    Fetch a W&B run and produce:
      1) df_long: DataFrame of full selected summary metrics (no NaNs).
      2) df_short: DataFrame of concise summary metrics.
      3) latex_long: LaTeX table string of full metrics.
      4) latex_short: LaTeX table string of concise metrics.
      5) fig: single multi-panel figure of selected history curves.
    Usage:
      %matplotlib inline
      df_long, df_short, latex_long, latex_short, fig = fetch_and_report_wandb(run_path)
      display(df_long)      # validate full table
      display(df_short)     # validate shortlist
      print(latex_long)     # paste into thesis (full)
      print(latex_short)    # paste into thesis (short)
    """
    
    api = wandb.Api()
    run = api.run(run_path)
    summary = run.summary._json_dict
    config = dict(run.config)

    # Extract model, hook, dataset for caption
    model_full = config.get("model_name", "")
    model = model_full.split("/")[-1]
    hook = config.get("hook_name", "")
    dataset_full = config.get("dataset_path", "")
    dataset = dataset_full.split("/")[-1]
    caption = (
        f"SAE Training results for \\texttt{{{model}}} "
        f"on hook \\texttt{{{hook}}}, using the \\texttt{{{dataset}}} dataset."
    )

    def get_val(key_path):
        # nested lookup
        if key_path in summary:
            return summary[key_path]
        v = summary
        for p in key_path.split('/'):
            if isinstance(v, dict) and p in v:
                v = v[p]
            else:
                return None
        return v

    # ---- Full (Long) Metrics ----
    metrics_map_long = {
        "Explained Variance":                "reconstruction_quality/explained_variance",
        "Reconstruction MSE":                "reconstruction_quality/mse",
        "Cosine Similarity":                 "reconstruction_quality/cossim",
        "Overall Loss":                      "losses/overall_loss",
        "MSE Loss":                          "losses/mse_loss",
        "L1 Loss":                           "losses/l1_loss",
        "Activation Sparsity (L0)":          "metrics/l0",
        "L1 Sparsity (sum abs activ.)":      "sparsity/l1",
        "Dead Features":                     "sparsity/dead_features",
        "Shrinkage (L2 Ratio)":              "shrinkage/l2_ratio",
        "Relative Recon. Bias":              "shrinkage/relative_reconstruction_bias",
        "CE Loss (no SAE)":                  "model_performance_preservation/ce_loss_without_sae",
        "CE Loss (with SAE)":                "model_performance_preservation/ce_loss_with_sae",
        "CE Loss (ablation)":                "model_performance_preservation/ce_loss_with_ablation",
        "CE Loss Score":                     "model_performance_preservation/ce_loss_score"
    }
    rows_long = []
    for name, key in metrics_map_long.items():
        val = get_val(key)
        if val is not None:
            rows_long.append({"Metric": name, "Value": float(val)})
    df_long = pd.DataFrame(rows_long)

    latex_long = df_long.to_latex(
        index=False,
        caption=caption,
        label="tab:sae_metrics_full",
        float_format="%.4f"
    )

    # ---- Concise (Short) Metrics ----
    metrics_map_short = {
        "Explained Variance":         "reconstruction_quality/explained_variance",
        "Reconstruction MSE":         "reconstruction_quality/mse",
        "Activation Sparsity (L0)":   "metrics/l0",
        "L1 Sparsity":                "sparsity/l1",
        "Overall Loss":               "losses/overall_loss",
        "CE Loss (no SAE)":           "model_performance_preservation/ce_loss_without_sae"
    }
    rows_short = []
    for name, key in metrics_map_short.items():
        val = get_val(key)
        if val is not None:
            rows_short.append({"Metric": name, "Value": float(val)})
    df_short = pd.DataFrame(rows_short)

    latex_short = df_short.to_latex(
        index=False,
        caption=caption,
        label="tab:sae_metrics_short",
        float_format="%.4f"
    )

    # ---- History & Plot ----
    history_keys = [
        "losses/overall_loss",
        "losses/mse_loss",
        "losses/l1_loss",
        "metrics/explained_variance",
        "metrics/l0",
        "shrinkage/l2_ratio"
    ]
    history_names = [
        "Overall Loss", "MSE Loss", "L1 Loss",
        "Explained Variance", "Activation Sparsity (L0)", "Shrinkage (L2 Ratio)"
    ]
    df_hist = run.history(keys=history_keys, pandas=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, key, title in zip(axes.flatten(), history_keys, history_names):
        if key in df_hist:
            ax.plot(df_hist.index, df_hist[key], linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Training Step")
            ax.set_ylabel(title)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    return df_long, df_short, latex_long, latex_short, fig