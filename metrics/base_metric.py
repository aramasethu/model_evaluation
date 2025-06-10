# metrics/base_metric.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from datasets import load_dataset

class BaseMetric(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Dict:
        """Load the dataset specific to this metric"""
        return load_dataset(self.dataset_name)
    
    @abstractmethod
    def evaluate(self, model_handler, split_name: str) -> Dict[str, Any]:
        """Implement the evaluation logic for this metric"""
        pass
    
    @abstractmethod
    def aggregate_results(self, results: Dict) -> Dict[str, Any]:
        """Implement how to aggregate results for this metric"""
        pass