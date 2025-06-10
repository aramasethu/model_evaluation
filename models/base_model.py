# models/base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

class ModelHandler:
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.model_name = model_config["name"]
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        print(f"Loading model: {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **self.config.get("load_args", {})
        )
        return model, tokenizer
        
    def generate(self, prompt: str) -> str:
        template = self.config.get("prompt_template", 
            "<|system|>You are a helpful AI assistant.</s><|user|>{prompt}</s><|assistant|>")
        formatted_prompt = template.format(prompt=prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            **self.config.get("generation_config", {})
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_marker = self.config.get("response_marker", "<|assistant|>")
        if response_marker in response:
            response = response.split(response_marker)[-1].strip()
        return response