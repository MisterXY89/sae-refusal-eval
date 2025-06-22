import os
import zlib
import json
import random
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List

from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


os.environ["HF_HOME"] = "/share/tilman.kerl/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/share/tilman.kerl/huggingface"


@dataclass
class DatasetMetrics:
    context_length: float
    ngram_diversity: float
    compression_ratio: float
    refusal_rate: float
    type_token_ratio: float


class DatasetAnalyzer:
    REFUSAL_PROTOTYPES = [        
        "I'm sorry",
        "I am sorry",
        "I apologize",
        "As an AI",
        "As a language model",
        "As an assistant",
        "I cannot",
        "I can't",        
        "I am unable to",
        "I'm unable to",
        "I am not able to",
        "I'm not able to",
        "I will not",
        "No, I will not",
        "I am not",
        # extra for LLaMA 1
        "I'm sorry",
        "\"I'm sorry",
        "“I’m sorry",
        "I’m sorry",
        "I'm not going to do that",
        # added 
        "I don't",
        "I do not",
    ]

    def __init__(self, texts: List[str], refusal_threshold: float = 0.75):
        self.texts = texts
        self.refusal_threshold = refusal_threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.prototype_embeddings = self.model.encode(
            self.REFUSAL_PROTOTYPES, normalize_embeddings=True
        )

    def context_length(self) -> float:
        return float(np.mean([len(t.split()) for t in self.texts]))

    def ngram_diversity(self, n: int = 3) -> float:
        all_ngrams = []
        for text in self.texts:
            words = text.split()
            all_ngrams.extend([" ".join(words[i : i + n]) for i in range(len(words) - n + 1)])
        return len(set(all_ngrams)) / (len(all_ngrams) or 1)

    def compression_ratio(self) -> float:
        joined = " ".join(self.texts).encode()
        compressed = zlib.compress(joined)
        return len(joined) / len(compressed)

    def compute_type_token_ratio(self) -> float:
        tokens = [tok for t in self.texts for tok in t.split()]
        return len(set(tokens)) / (len(tokens) or 1)


    def refusal_rate(self) -> float:
        all_sentences = [sent for text in self.texts for sent in sent_tokenize(text)]
        sentence_embeddings = self.model.encode(all_sentences, normalize_embeddings=True)
        sims = util.cos_sim(sentence_embeddings, self.prototype_embeddings).max(dim=1).values
        refusal_sent_flags = (sims >= self.refusal_threshold).float().numpy()

        # map sentence-level refusals back to document level
        refusal_flags = []
        idx = 0
        for text in self.texts:
            n_sents = len(sent_tokenize(text))
            refusal_flags.append(int(np.any(refusal_sent_flags[idx : idx + n_sents])))
            idx += n_sents

        return float(np.mean(refusal_flags))

    def analyze(self) -> DatasetMetrics:
        return DatasetMetrics(
            context_length=self.context_length(),
            ngram_diversity=self.ngram_diversity(),
            compression_ratio=self.compression_ratio(),
            refusal_rate=self.refusal_rate(),
            type_token_ratio=self.compute_type_token_ratio()
        )        


def analyze_hf_dataset(
    hf_dataset_path: str,
    ds_name: str,
    split: str = "train",
    field: str = "text",
    save: bool = True,
    save_path: str = None,
    max_samples: int = 10000,
    seed: int = 42,
    streaming: bool = False,    
) -> DatasetMetrics:
    ds = load_dataset(hf_dataset_path, split=split, streaming=streaming)

    rng = random.Random(seed)
    sampled_texts = []

    save_path = Path(save_path)

    if streaming:
        for example in ds:
            if field in example:
                if rng.random() < max_samples / 1e6:  # subsample from stream (approximate)
                    sampled_texts.append(example[field])
                    if len(sampled_texts) >= max_samples:
                        break
    else:
        full_ds = ds.shuffle(seed=seed)
        sampled_texts = full_ds.select(range(min(len(full_ds), max_samples)))[field]

    analyzer = DatasetAnalyzer(sampled_texts)
    metrics = analyzer.analyze()

    if save:
        fname = save_path / f"{ds_name}_metrics.json"
        print(fname)
        with open(fname, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    return metrics
