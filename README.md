# Sparse Autoencoder for Feature Learning

Contains code, data & results.


## Data & Instruct
Pythia quantized models: https://huggingface.co/Crataco/Pythia-Deduped-Series-GGML 
--> or use bits & bytes

- The [pile](https://pile.eleuther.ai/).
  - https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated
- https://github.com/EleutherAI/pythia/issues/155 
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)

160M
- https://huggingface.co/EleutherAI/sae-pythia-160m-32k

410M
- https://huggingface.co/SummerSigh/Pythia410m-V0-Instruct
  - https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25
- https://huggingface.co/EleutherAI/sae-pythia-410m-65k


1.4B
- https://huggingface.co/EleutherAI/pythia-1.4b
- https://huggingface.co/lambdalabs/pythia-1.4b-deduped-synthetic-instruct
  - Dataset: Dahoas/synthetic-instruct-gptj-pairwise
- 

2.8B:
- jacobdunefsky/pythia-2.8B-saes 
- https://huggingface.co/lambdalabs/pythia-2.8b-deduped-synthetic-instruct
- Dahoas/synthetic-instruct-gptj-pairwise
  
6.9B
- 6.9: https://huggingface.co/allenai/open-instruct-pythia-6.9b-tulu

Olmo 7b -> instruct + base model
-> SAE missing


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

(C) 2024 [Tilman Kerl](https://linkedin.com/in/tilman-kerl)
