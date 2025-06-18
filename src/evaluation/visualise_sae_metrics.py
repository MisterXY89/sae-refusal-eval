import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re

def parse_token_size(token_str: str) -> float:
    """
    Converts a token size string (e.g., '419M', '1B') into a numeric value in millions.
    Returns NaN for unparsable strings.
    """
    if not isinstance(token_str, str):
        return float('nan')
    
    token_str = token_str.upper()
    try:
        if 'B' in token_str:
            num = float(re.findall(r'[\d\.]+', token_str)[0])
            return num * 1000
        elif 'M' in token_str:
            num = float(re.findall(r'[\d\.]+', token_str)[0])
            return num
    except (IndexError, ValueError):
        return float('nan')
    return float('nan')

def sanitize_filename(name: str) -> str:
    """Removes special characters from a string to make it a valid filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def load_sae_results(results_dir: str = "results/saes/") -> pd.DataFrame:
    """
    Loads all SAE evaluation .json files from a directory into a pandas DataFrame.

    Args:
        results_dir: The directory containing the .json result files.

    Returns:
        A pandas DataFrame with the aggregated results, or an empty DataFrame if no files are found.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Directory not found at '{results_dir}'")
        return pd.DataFrame()

    json_files = list(results_path.glob("*.json"))
    if not json_files:
        print(f"No .json files found in '{results_dir}'")
        return pd.DataFrame()

    print(f"Found {len(json_files)} result files. Loading...")
    
    all_results = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
            
    df = pd.DataFrame(all_results)

    # --- Data Cleaning and Feature Engineering ---
    if 'sae_token_size' in df.columns:
        df['sae_token_size_mil'] = df['sae_token_size'].apply(parse_token_size)

    df = df.sort_values('sae_token_size_mil')
    print("Data loaded and processed successfully.")
    return df

def visualize_sae_results_grouped(df: pd.DataFrame, results_path: str = "results/visualizations/"):
    """
    Generates and saves a separate, more interpretable dashboard for each 
    evaluation dataset found in the results.

    Args:
        df: DataFrame containing the aggregated SAE results.
        results_path: Directory to save the output plot images.
    """
    if df.empty:
        print("DataFrame is empty. Skipping visualization.")
        return

    Path(results_path).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    
    eval_datasets = df['eval_dataset'].unique()
    print(f"\nFound {len(eval_datasets)} evaluation datasets. Generating a dashboard for each...")

    for i, dataset in enumerate(eval_datasets):
        print(f"  [{i+1}/{len(eval_datasets)}] Generating dashboard for eval_dataset: '{dataset}'")
        
        df_group = df[df['eval_dataset'] == dataset].copy()
        
        # --- Plot 1: Reconstruction Quality (Bar Plot Grid) ---
        g1 = sns.catplot(
            data=df_group,
            x='sae_token_size_mil',
            y='explained_variance',
            hue='sae_train_dataset',
            col='sae_expansion_factor',
            kind='bar',
            palette='Blues',
            height=6,
            aspect=1.2,
            legend_out=True
        )
        g1.fig.suptitle(f"Reconstruction Quality (EV) on '{dataset}'", y=1.03, fontsize=20, weight='bold')
        g1.set_axis_labels("Training Tokens (Millions)", "Explained Variance (EV)")
        g1.set_titles("Expansion Factor: {col_name}")
        g1.set(ylim=(0.8, 1.0))
        g1.despine(left=True)

        save_filename_1 = f"dashboard_1_EV_eval_on_{sanitize_filename(dataset)}.png"
        save_path_1 = Path(results_path) / save_filename_1
        plt.savefig(save_path_1, dpi=150, bbox_inches='tight')
        print(f"    -> Plot 1 saved to '{save_path_1}'")
        plt.close(g1.fig)

        # --- Plot 2: Feature Health (Bar Plot Grid) ---
        g2 = sns.catplot(
            data=df_group,
            x='sae_token_size_mil',
            y='dead_features_pct',
            hue='sae_train_dataset',
            col='sae_expansion_factor',
            kind='bar',
            palette='Oranges',
            height=6,
            aspect=1.2,
            legend_out=True
        )
        g2.fig.suptitle(f"Feature Health on '{dataset}'", y=1.03, fontsize=20, weight='bold')
        g2.set_axis_labels("Training Tokens (Millions)", "Dead Features (%)")
        g2.set_titles("Expansion Factor: {col_name}")
        # Format y-axis as percentage
        for ax in g2.axes.flat:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        g2.despine(left=True)

        save_filename_2 = f"dashboard_2_DeadFeatures_eval_on_{sanitize_filename(dataset)}.png"
        save_path_2 = Path(results_path) / save_filename_2
        plt.savefig(save_path_2, dpi=150, bbox_inches='tight')
        print(f"    -> Plot 2 saved to '{save_path_2}'")
        plt.close(g2.fig)

        # --- Plot 3: Reconstruction vs. Sparsity Trade-off (Scatter Plot) ---
        g3 = sns.relplot(
            data=df_group,
            x='explained_variance',
            y='dead_features_pct',
            hue='sae_expansion_factor',
            size='sae_token_size_mil',
            style='sae_train_dataset',
            palette='viridis',
            height=8,
            aspect=1.4,
            sizes=(100, 400),
            legend="full"
        )
        g3.fig.suptitle(f"Reconstruction vs. Dead Features Trade-off on '{dataset}'", y=1.03, fontsize=20, weight='bold')
        g3.set_axis_labels("Explained Variance (Higher is Better)", "Dead Features (Lower is Better)")
        g3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        g3.despine(left=True, bottom=True)
        
        save_filename_3 = f"dashboard_3_Tradeoff_eval_on_{sanitize_filename(dataset)}.png"
        save_path_3 = Path(results_path) / save_filename_3
        plt.savefig(save_path_3, dpi=150, bbox_inches='tight')
        print(f"    -> Plot 3 saved to '{save_path_3}'")
        plt.close(g3.fig)


if __name__ == '__main__':
    # Define the directory where your results are stored
    RESULTS_DIRECTORY = "results/saes/"
    VISUALIZATION_DIRECTORY = "visualizations/"
    
    # 1. Load the data
    results_df = load_sae_results(RESULTS_DIRECTORY)
    
    # 2. Generate and save the visualizations, grouped by evaluation dataset
    if not results_df.empty:
        visualize_sae_results_grouped(results_df, VISUALIZATION_DIRECTORY)

