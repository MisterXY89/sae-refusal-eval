import pandas as pd
from typing import Tuple, Optional, Dict, Any
from colorama import init, Fore, Style

import matplotlib.pyplot as plt
import seaborn as sns

# initialize colorama
init(autoreset=True)

def extract_mmlu_results(
    results: Dict[str, dict],
    overall_key: str = 'mmlu',
    json_filename: Optional[str] = None
) -> Tuple[str, pd.DataFrame]:
    """
    Returns:
      - summary: a styled, multi-line summary including overall and the four official aggregates.
      - df: full DataFrame of every MMLU category, with `is_aggregate` flagged.
    """
    # Official aggregates per _mmlu.yaml
    AGG_KEYS = {
        'STEM':               'mmlu_stem',
        'Social Sciences':    'mmlu_social_sciences',
        'Humanities':         'mmlu_humanities',
        'Other':              'mmlu_other'
    }

    # Build master records list
    records = []
    for key, metrics in results.items():
        name = key if key == overall_key else metrics.get('alias', key).strip(' -')
        records.append({
            'task':          name,
            'accuracy':      metrics.get('acc,none'),
            'stderr':        metrics.get('acc_stderr,none'),
            'is_aggregate':  key in AGG_KEYS.values() or key == overall_key
        })

    # Create and sort DataFrame
    df = (pd.DataFrame(records)
            .sort_values('accuracy', ascending=False)
            .reset_index(drop=True))

    # Helper to format a single entry
    def fmt(key: str) -> str:
        m = results.get(key, {})
        return f"{m.get('acc,none', 0):.3f} ± {m.get('acc_stderr,none', 0):.3f}"

    # Build styled summary
    lines = [
        f"{Style.BRIGHT}MMLU Accuracy:{Style.RESET_ALL} {Fore.GREEN}{fmt(overall_key)}"
    ]
    for label, key in AGG_KEYS.items():
        if key in results:
            lines.append(
                f"- {Style.BRIGHT}{label:<15}{Style.RESET_ALL}\t{Fore.CYAN}{fmt(key)}"
            )

    summary = "\n".join(lines)

    # Optionally save full JSON
    if json_filename:
        df.to_json(json_filename, orient='records')

    return summary, df


def visualize_mmlu(
    model_results: List[Dict[str, Any]],
    overall_key: str = 'mmlu'
) -> None:
    """
    Compare multiple MMLU runs: subgroup bars plus overall scores in a top-left text box.
    """
    AGG = {
        'STEM': 'mmlu_stem',
        'Social Sci': 'mmlu_social_sciences',
        'Humanities': 'mmlu_humanities',
        'Other': 'mmlu_other'
    }
    labels = list(AGG.keys())
    n = len(model_results)
    palette = sns.color_palette("colorblind", n)

    # x positions and bar widths
    x = range(len(labels))
    width = 0.8 / n
    offsets = [(i - (n-1)/2) * width for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, entry in enumerate(model_results):
        name, res = entry['name'], entry['results']
        accs = [res[AGG[l]]['acc,none'] for l in labels]
        errs = [res[AGG[l]]['acc_stderr,none'] for l in labels]
        pos = [xi + offsets[i] for xi in x]
        ax.bar(pos, accs, width, yerr=errs, capsize=5,
               label=name, color=palette[i], edgecolor='black')

    # Move overall scores box to upper-left
    text = "\n".join([
        f"{e['name']} overall: {e['results'][overall_key]['acc,none']:.2f}"
        for e in model_results
    ])
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            ha='left', va='top', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Aesthetics
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_title('MMLU Aggregate Subgroups', fontsize=16)
    ax.legend(title='Models', loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    
## Dummy data example
# dummy_res1 = {
#     'mmlu': {'acc,none': 0.30, 'acc_stderr,none': 0.015},
#     'mmlu_stem': {'acc,none': 0.32, 'acc_stderr,none': 0.025},
#     'mmlu_social_sciences': {'acc,none': 0.29, 'acc_stderr,none': 0.035},
#     'mmlu_humanities': {'acc,none': 0.27, 'acc_stderr,none': 0.030},
#     'mmlu_other': {'acc,none': 0.24, 'acc_stderr,none': 0.025}
# }

# dummy_res2 = {
#     'mmlu': {'acc,none': 0.26, 'acc_stderr,none': 0.020},
#     'mmlu_stem': {'acc,none': 0.28, 'acc_stderr,none': 0.030},
#     'mmlu_social_sciences': {'acc,none': 0.275, 'acc_stderr,none': 0.041},
#     'mmlu_humanities': {'acc,none': 0.246, 'acc_stderr,none': 0.039},
#     'mmlu_other': {'acc,none': 0.223, 'acc_stderr,none': 0.036}
# }

## Call with dummy data
# visualize_mmlu([
#     {'name': 'Dummy Model A', 'results': dummy_res1},
#     {'name': 'Dummy Model B', 'results': dummy_res2}
# ])
