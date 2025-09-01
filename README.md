# Evaluation of Sparse Autoencoder-based Refusal Features in LLMs

This repository contains the source code and experimental setup for the master's thesis,

**"Evaluation of Sparse Autoencoder-based Refusal Features in LLMs: A Dataset-dependence Study"**.

The core objective of this research is to investigate how training data of SAEs influences its ability to identify, represent, and control refusal behaviours in (base) LLMs. Our findings demonstrate that SAE feature discovery is highly dependent on its training data + the intervention layer.
A mix of pre-training and instruction-style data yields the most effective and steerable refusal-related features, aligning with previous and concurrent work. 

## Methodology Overview

We follow a three-stage pipeline, here a simplified overview:

### Data
Assemble two corpora: 
SmolLM2 pre-training data (PRE) and LMSys-Chat instruction data (INS), plus mixtures at 30:70, 50:50, 70:30 ratios. 
Compute dataset statistics (context length, type–token ratio, $n$-gram diversity, compression ratio, refusal prevalence).

### Training 
Select promising transformer layers via linear refusal probes. 
Train Top-k SAEs on those layers under varied expansion factors ($E \in \{8,16,32\}$) and token budgets ($T \in \{50\text{M},125\text{M},250\text{M},419\text{M}\}$). 
Evaluate reconstruction quality (explained variance, cosine similarity, MSE, sparsity).
    
### Evaluation 
Identify refusal-related latents via effect-size ranking (Cohen’s $d$) and annotate them with Delphi. 
Perform feature interventions (injection, clamping) through SAE-based hooks.
Benchmark refusal (AdvBench, Alpaca), general knowledge (MMLU), and commonsense reasoning (HellaSwag).

## Limitations

Experiments are limited to small open models (SmolLM2-135M); results may not scale to frontier LLMs. Evaluation benchmarks (AdvBench, Alpaca, MMLU, HellaSwag) cover only part of the safety landscape. SAE-based steering remains fragile, with feature saturation, dataset sensitivity, and strong dependence on hook-layer choice. See related discussions [here](https://www.alignmentforum.org/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse), [here](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team) and [here](https://www.alignmentforum.org/s/AtTZjoDm8q3DbDT8Z/p/4uXCAJNuPKtKBsi28).

----

*Most experiments were run on TU Wien’s HPC cluster (A100 80GB GPUs). Smaller runs are possible on consumer GPUs using reduced token budgets.*

## Citation
If you use this code or datasets, please cite:
```bibtex
@mastersthesis{kerl2025saerefusal,
  title={Evaluation of Sparse Autoencoder-based Refusal Features in LLMs: A Dataset-dependence Study},
  author={Tilman Kerl},
  school={Technische Universit{\"a}t Wien}
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

(C) 2024-2025 [Tilman Kerl](https://linkedin.com/in/tilman-kerl)
