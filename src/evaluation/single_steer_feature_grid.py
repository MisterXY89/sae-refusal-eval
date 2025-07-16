import os
import sys
import json
from types import SimpleNamespace
import argparse

from evaluation.run_eval import refusal_eval  

def main():
    """
    This script runs a single evaluation based on a configuration from a JSON
    file and saves the configuration and its results to the specified 'out_path'.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a single evaluation and save results.")
    parser.add_argument(
        '--config_path',
        type=str,
        # Updated default config file name
        default='evaluation/feature_grid_configs_135M.json',
        help='Path to the grid search configuration JSON file.'
    )
    parser.add_argument(
        '--config_index',
        type=int,
        required=True,
        help='The index of the configuration to run from the JSON file.'
    )
    cli_args = parser.parse_args()

    # --- Load Configuration ---
    try:
        with open(cli_args.config_path, 'r') as f:
            configs = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {cli_args.config_path}", file=sys.stderr)
        sys.exit(1)

    if not 0 <= cli_args.config_index < len(configs):
        print(f"❌ Error: Index {cli_args.config_index} is out of bounds for {len(configs)} configs.", file=sys.stderr)
        sys.exit(1)

    # Select the specific configuration for this job
    conf = configs[cli_args.config_index]

    # --- Prepare and Run Evaluation ---
    args = SimpleNamespace(**conf)
    
    # Extract details for logging
    model_name = os.path.basename(conf.get('sparse_model', 'N/A').rstrip('/'))
    hook = conf.get('hookpoint', 'N/A')
    num_indices = len(conf.get('feature_indices', []))
    coefficient = conf.get('steering_coefficients', [None])[0]
    
    print(f"🚀 [RUNNING] Config index: {cli_args.config_index}")
    print(f"Model: {model_name}@{hook} | N={num_indices} | C={coefficient}")

    # hotfix wrong config
    args.action = "add"

    # Run the evaluation
    rr, orr = refusal_eval(args)

    print(f"✅ [COMPLETED] Config index: {cli_args.config_index}")
    print(f"Results: Refusal Rate = {rr}, Open Refusal Rate = {orr}")

    # --- Save Results ---
    out_path = conf.get("out_path")
    if not out_path:
        print(f"⚠️ Warning: No 'out_path' specified in config index {cli_args.config_index}. Skipping save.", file=sys.stderr)
        return

    # Create the directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except OSError as e:
        print(f"❌ Error creating directory {os.path.dirname(out_path)}: {e}", file=sys.stderr)
        sys.exit(1)

    # Combine the original config and the results
    result_data = {
        "config": conf,
        "results": {
            "refusal_rate": rr,
            "open_refusal_rate": orr
        }
    }
    
    # Write the combined data to the specified file
    with open(out_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"💾 Results and config saved to: {out_path}")


if __name__ == "__main__":
    main()