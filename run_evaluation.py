# run_evaluation.py
from models.base_model import ModelHandler
from metrics.sst import SafetyMetric
from metrics.base_metric import BaseMetric
# Import other metrics as needed
import json
from datetime import datetime
import os
import yaml
from pathlib import Path
from typing import List, Dict
from pprint import pformat

class ModelConfigError(Exception):
    """Custom exception for model configuration errors"""
    pass

def validate_model_config(model: Dict, index: int) -> None:
    """Validate a single model configuration"""
    required_fields = ['name']
    optional_fields = ['prompt_template', 'generation_config']
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in model]
    if missing_fields:
        raise ModelConfigError(
            f"Model at index {index} is missing required fields: {missing_fields}"
        )
    
    # Validate generation_config if present
    if 'generation_config' in model and not isinstance(model['generation_config'], dict):
        raise ModelConfigError(
            f"Model '{model['name']}': generation_config must be a dictionary"
        )

def load_model_configs(config_path: str) -> List[Dict]:
    """Load and validate model configurations from YAML file"""
    try:
        # Check if file exists
        config_path = Path(config_path)
        if not config_path.exists():
            raise ModelConfigError(f"Config file not found: {config_path}")
        
        # Load YAML
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ModelConfigError(f"Error parsing YAML file: {str(e)}")
        
        # Basic structure validation
        if not isinstance(config, dict):
            raise ModelConfigError("Config file must contain a dictionary")
        
        if 'models' not in config:
            raise ModelConfigError("Config file must have a 'models' key")
            
        if not isinstance(config['models'], list):
            raise ModelConfigError("'models' must be a list")
            
        if not config['models']:
            raise ModelConfigError("No models found in config file")
        
        # Validate each model
        for idx, model in enumerate(config['models']):
            validate_model_config(model, idx)
        
        return config['models']
        
    except ModelConfigError as e:
        print(f"\nModel Configuration Error:")
        print(f"{'='*50}")
        print(f"{str(e)}")
        print(f"{'='*50}")
        raise
    
    except Exception as e:
        print(f"\nUnexpected Error:")
        print(f"{'='*50}")
        print(f"An unexpected error occurred while loading model configs:")
        print(f"{str(e)}")
        print(f"{'='*50}")
        raise

def run_evaluation(models_config: List[Dict], metrics: List[BaseMetric], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for model_config in models_config:
        print(f"\nEvaluating model: {model_config['name']}")
        model_handler = ModelHandler(model_config)
        model_results = {}
        
        for metric in metrics:
            metric_name = metric.__class__.__name__
            print(f"\nRunning {metric_name}...")
            
            split_results = {}
            for split in metric.dataset.keys():
                results = metric.evaluate(model_handler, split)
                split_results[split] = results
            
            model_results[metric_name] = {
                "raw_results": split_results,
                "aggregated": metric.aggregate_results(split_results)
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{model_config['name'].replace('/', '_')}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(model_results, f, indent=2)

if __name__ == "__main__":
    try:
        # Load models from config file
        models = load_model_configs("config/models.yaml")
        print(f"Successfully loaded {len(models)} model configurations:")
        for model in models:
            print(f"- {model['name']}")
            
        metrics = [
            SafetyMetric(),
            # Add more metrics...
        ]
        
        run_evaluation(models, metrics, "evaluation_results")
        
    except ModelConfigError:
        print("Exiting due to configuration error.")
        exit(1)
    except Exception as e:
        print(f"Exiting due to unexpected error: {str(e)}")
        exit(1)