# metrics/accuracy_per_flop_metric.py
from .base_metric import BaseMetric
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from typing import Dict, Any

class AccuracyPerFLOPMetric(BaseMetric):
    def __init__(self, dataset_name: str = "hellaswag"):
        super().__init__(dataset_name)
        self.sample_size = 50
        self.max_new_tokens = 128
        
    def evaluate(self, model_handler, split_name: str) -> Dict[str, Any]:
        model = model_handler.model
        tokenizer = model_handler.tokenizer
        
        dataset = self.dataset[split_name].select(range(self.sample_size))
        correct = 0
        param_count = sum(p.numel() for p in model.parameters())

        for row in tqdm(dataset, desc="Evaluating Accuracy"):
            prompt = row["ctx"] if "ctx" in row else row["prompt"]
            label = str(row["label"])

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_new_tokens=self.max_new_tokens)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).lower()

            if label.lower() in output_text:
                correct += 1

        accuracy = correct / self.sample_size
        flops_per_token = 6 * param_count
        total_flops = flops_per_token * self.max_new_tokens * self.sample_size
        performance_per_flop = accuracy / total_flops

        return {
            "total_examples": self.sample_size,
            "accuracy": accuracy,
            "total_flops": total_flops,
            "performance_per_flop": performance_per_flop
        }

    def aggregate_results(self, results: Dict) -> Dict[str, Any]:
        """Aggregate results across splits if needed"""
        total_examples = sum(split_data["total_examples"] for split_data in results.values())
        avg_accuracy = sum(split_data["accuracy"] * split_data["total_examples"] 
                         for split_data in results.values()) / total_examples
        total_flops = sum(split_data["total_flops"] for split_data in results.values())
        avg_perf_per_flop = sum(split_data["performance_per_flop"] * split_data["total_examples"] 
                               for split_data in results.values()) / total_examples
        
        return {
            "total_examples": total_examples,
            "average_accuracy": avg_accuracy,
            "total_flops": total_flops,
            "average_performance_per_flop": avg_perf_per_flop
        }