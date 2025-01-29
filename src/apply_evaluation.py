
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    mmlu: float
    sparsity: float
    reconstruction_error: float
    refusal_rate: float
    over_refusal: float
