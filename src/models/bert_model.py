import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any

class BERTModel:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Initialize base BERT model"""
        self.device = "cpu"  # Force CPU usage to avoid CUDA issues
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer with error handling"""
        try:
            print(f"Loading {self.model_name}...")
            # Load tokenizer with explicit settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                local_files_only=False
            )
            
            # Load model with explicit settings
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=False,
                torch_dtype=torch.float32,  # Force float32
                trust_remote_code=False
            )
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction for a single text"""
        try:
            # Ensure text is a string
            text = str(text)
            
            # Tokenize with explicit settings
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.cpu().numpy()[0]
                
            return {
                'raw_scores': scores,
                'logits': outputs.logits.cpu().numpy()[0]
            }
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'model_type': self.model.config.model_type if self.model else None
        }