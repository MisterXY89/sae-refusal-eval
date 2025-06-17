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
    # Convert token size string to a numeric value for plotting
    if 'sae_token_size' in df.columns:
        df['sae_token_size_mil'] = df['sae_token_size'].apply(parse_token_size)

    print("Data loaded and processed successfully.")
    return df

def visualize_sae_results(df: pd.DataFrame, save_path: str = "sae_evaluation_dashboard.png"):
    """
    Generates and saves a dashboard of plots to visualize SAE performance.

    Args:
        df: DataFrame containing the aggregated SAE results.
        save_path: Path to save the output plot image.
    """
    if df.empty:
        print("DataFrame is empty. Skipping visualization.")
        return

    # Set a nice theme for the plots
    sns.set_theme(style="whitegrid", palette="viridis", context="talk")

    # Determine the number of unique datasets to adjust plot layout
    num_datasets = df['sae_train_dataset'].nunique()
    
    # Create a figure with subplots. We'll create 3 main plots.
    fig, axes = plt.subplots(3, 1, figsize=(15, 25))
    fig.suptitle('SAE Performance Analysis Dashboard', fontsize=24, weight='bold')

    # --- Plot 1: Explained Variance vs. Expansion Factor, faceted by Dataset ---
    ax1 = axes[0]
    sns.barplot(
        data=df,
        x='sae_expansion_factor',
        y='explained_variance',
        hue='sae_train_dataset',
        ax=ax1,
        errorbar='sd' # Show standard deviation as error bars
    )
    ax1.set_title('Explained Variance vs. SAE Expansion Factor', fontsize=18, weight='bold')
    ax1.set_xlabel('SAE Expansion Factor', fontsize=14)
    ax1.set_ylabel('Explained Variance (EV)', fontsize=14)
    ax1.set_ylim(bottom=max(0, df['explained_variance'].min() - 0.05), top=1.01)
    ax1.legend(title='Train Dataset', loc='best')

    # --- Plot 2: Cosine Similarity vs. Token Size, faceted by Expansion Factor ---
    ax2 = axes[1]
    # Use a pointplot to show evolution over token sizes
    sns.pointplot(
        data=df.sort_values('sae_token_size_mil'),
        x='sae_token_size_mil',
        y='cosine_similarity',
        hue='sae_expansion_factor',
        ax=ax2,
        palette='magma',
        errorbar='sd',
        dodge=True
    )
    ax2.set_title('Reconstruction Cosine Similarity vs. Training Tokens', fontsize=18, weight='bold')
    ax2.set_xlabel('SAE Training Tokens (Millions)', fontsize=14)
    ax2.set_ylabel('Cosine Similarity', fontsize=14)
    ax2.legend(title='Expansion Factor', loc='best')

    # --- Plot 3: Dead Feature Percentage vs. Training Dataset ---
    ax3 = axes[2]
    sns.boxplot(
        data=df,
        x='sae_train_dataset',
        y='dead_features_pct',
        hue='sae_expansion_factor',
        ax=ax3,
        palette='crest'
    )
    ax3.set_title('Dead Feature Percentage by Training Dataset', fontsize=18, weight='bold')
    ax3.set_xlabel('SAE Training Dataset', fontsize=14)
    ax3.set_ylabel('Dead Features (%)', fontsize=14)
    # Format y-axis as percentage
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax3.legend(title='Expansion Factor', loc='best')

    # --- Finalize and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nDashboard saved successfully to '{save_path}'")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    plt.show()


if __name__ == '__main__':
    # Define the directory where your results are stored
    RESULTS_DIRECTORY = "results/saes/"
    
    # 1. Load the data
    results_df = load_sae_results(RESULTS_DIRECTORY)
    
    # 2. Generate and save the visualizations
    if not results_df.empty:
        visualize_sae_results(results_df)

