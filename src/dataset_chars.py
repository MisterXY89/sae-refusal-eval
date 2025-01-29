import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.sparse import csr_matrix
import zlib

from apply_evaluation import PerformanceMetrics

@dataclass
class DatasetMetrics:
    context_length: float
    self_repetition: float
    ngram_diversity: float
    compression_ratio: float
    perplexity: Optional[float]
    perplexity_gap: Optional[float]
    kmeans_clusters: Optional[Dict]

class EnhancedDatasetAnalyzer:
    def __init__(
        self, 
        texts: List[str],
        use_gpu: bool = False,
        model_name: str = 'gpt2'
    ):
        self.texts = texts
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self._init_models(model_name)
    
    def _init_models(self, model_name: str) -> None:
        """initialise language models for perplexity calculation"""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        except Exception:
            self.model = self.tokenizer = None
    
    def context_length(self) -> float:
        """average context length across dataset"""
        return np.mean([len(text.split()) for text in self.texts])
    
    def self_repetition(self, k: int = 3) -> float:
        """measure repetition using frequency of n-grams"""
        ngram_freqs = []
        for text in self.texts:
            words = text.split()
            ngrams = [' '.join(words[i:i+k]) for i in range(len(words)-k+1)]
            if ngrams:
                ngram_freqs.append(len(set(ngrams)))
        return np.log(k * sum(ni + 1 for ni in ngram_freqs))
    
    def ngram_diversity(self, n: int = 3) -> float:
        """ratio of unique to total n-grams"""
        all_ngrams = []
        for text in self.texts:
            words = text.split()
            all_ngrams.extend([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
        return len(set(all_ngrams)) / (len(all_ngrams) or 1)
    
    def compression_ratio(self) -> float:
        """ratio of original to compressed size using zlib"""
        original = ' '.join(self.texts).encode()
        compressed = zlib.compress(original)
        return len(original) / len(compressed)
    
    def calculate_perplexity(self, max_length: int = 512) -> Dict[str, float]:
        """
        calculates cross-entropy based perplexity and gap between xl/base models
        uses gpt2-xl for high-capacity and gpt2 for base model comparison
        """
        if not self.model or not self.tokenizer:
            return {'perplexity': None, 'perplexity_gap': None}
            
        def get_model_perplexity(model_name: str) -> float:
            model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
            total_nll = 0
            total_tokens = 0
            
            with torch.no_grad():
                for text in self.texts:
                    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                                    max_length=max_length).to(self.device)
                    outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
                    total_nll += outputs.loss.item() * inputs['input_ids'].size(1)
                    total_tokens += inputs['input_ids'].size(1)
            
            return torch.exp(torch.tensor(total_nll / total_tokens)).item()

        # calculate perplexity for both models
        ppl_xl = get_model_perplexity('gpt2-xl')
        ppl_base = get_model_perplexity('gpt2')
        
        return {
            'perplexity': ppl_xl,
            'perplexity_gap': abs(ppl_xl - ppl_base)
        }
        
    def kmeans_clustering(self, n_clusters: int = 8) -> Dict[str, Union[np.ndarray, List]]:
        """cluster texts using tf-idf and k-means"""
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.texts)
        
        # training
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        
        # inference
        distances = [min(np.linalg.norm(x - mu) for mu in kmeans.cluster_centers_)
                    for x in X.toarray()]
        
        return {
            'cluster_assignments': kmeans.labels_,
            'distances': distances,
            # 'cluster_centers': kmeans.cluster_centers_
        }
    
    def analyze(self) -> DatasetMetrics:
        """compute all diversity metrics"""
        perplexity_metrics = self.calculate_perplexity()
        
        return DatasetMetrics(
            context_length=self.context_length(),
            self_repetition=self.self_repetition(),
            ngram_diversity=self.ngram_diversity(),
            compression_ratio=self.compression_ratio(),
            perplexity=perplexity_metrics['perplexity'],
            perplexity_gap=perplexity_metrics['perplexity_gap'],
            kmeans_clusters=self.kmeans_clustering()
        )
    

if __name__ == '__main__':
    from base_models_refuse.utils import get_harmless_instructions, get_harmful_instructions
    from visualise_metrics import plot_metrics


    # harmless_inst_train, harmless_inst_test = get_harmless_instructions()
    # harmless_analyzer = EnhancedDatasetAnalyzer(harmless_inst_test[:15])
    # harmless_metrics = harmless_analyzer.analyze()

    # print(harmless_metrics)

    # harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    # harmful_analyzer = EnhancedDatasetAnalyzer(harmful_inst_test[:15])
    # harmful_metrics = harmful_analyzer.analyze()

    # print(harmful_metrics)

    performance1 = PerformanceMetrics(70.0, 0.1, 0.05, 0.02, 0.01)
    performance2 = PerformanceMetrics(72.0, 0.08, 0.04, 0.015, 0.02)    

    harmless_metrics = DatasetMetrics(context_length=9.733333333333333, self_repetition=5.973809611869261, ngram_diversity=0.9913793103448276, compression_ratio=1.7151394422310757, perplexity=31.38776206970215, perplexity_gap=18.539030075073242, kmeans_clusters={'cluster_assignments': array([1, 3, 2, 4, 5, 0, 4, 6, 7, 1, 0, 3, 0, 2, 0], dtype=int32), 'distances': [0.5796517516395013, 0.645582224740537, 0.6758676488988081, 0.6347451362830715, 0.0, 0.7723972050210062, 0.6347451362830715, 0.0, 0.0, 0.5796517516395013, 0.7638800844556473, 0.645582224740537, 0.8807016588096739, 0.6758676488988081, 0.7820985122352259]})
    harmful_metrics = DatasetMetrics(context_length=11.533333333333333, self_repetition=6.161207321695077, ngram_diversity=0.8741258741258742, compression_ratio=2.117760617760618, perplexity=37.02283477783203, perplexity_gap=16.631553649902344, kmeans_clusters={'cluster_assignments': array([2, 6, 3, 4, 5, 0, 2, 1, 7, 2, 3, 1, 1, 2, 1], dtype=int32), 'distances': [0.6165458850261806, 0.0, 0.6128411560004505, 0.0, 0.0, 0.0, 0.85054592239412, 0.757463767942645, 0.0, 0.7195291209114426, 0.6128411560004505, 0.7164716985701415, 0.654489317896674, 0.5108163918651756, 0.8754084515658427]})
    plot_metrics([harmless_metrics, harmful_metrics], [performance1, performance2])
