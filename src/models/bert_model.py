import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any

class BERTModel:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize base BERT model
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make prediction for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Raw model outputs
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
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

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Tokenized inputs
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'vocab_size': self.tokenizer.vocab_size,
            'model_type': self.model.config.model_type
        }