import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the BERT sentiment analyzer"""
        # Load pre-trained model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load FinBERT, which is pre-trained on financial texts
        self.model_name = "ProsusAI/finbert"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def analyze_text(self, text):
        """
        Analyze sentiment of a single text
        
        Returns:
            dict: Contains sentiment scores and label
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.cpu().numpy()[0]
            
            # Map predictions to labels
            labels = ['negative', 'neutral', 'positive']
            label_scores = {label: float(score) for label, score in zip(labels, scores)}
            predicted_label = labels[scores.argmax()]
            
            return {
                'text': text,
                'sentiment': predicted_label,
                'scores': label_scores,
                'confidence': float(scores.max())
            }
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return None
        
    def analyze_dataframe(self, df, text_columns=['title', 'text']):
        """
        Analyze sentiment for text columns in a DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_columns (list): Columns containing text to analyze
            
        Returns:
            pd.DataFrame: Original DataFrame with sentiment columns added
        """
        results_df = df.copy()
        
        for column in text_columns:
            if column in df.columns:
                print(f"\nAnalyzing sentiment for {column}...")
                sentiments = []
                
                for text in tqdm(df[column], desc=f"Analyzing {column}"):
                    if pd.isna(text) or text == "":
                        sentiments.append({
                            'sentiment': 'neutral',
                            'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                            'confidence': 1.0
                        })
                    else:
                        result = self.analyze_text(text)
                        sentiments.append(result if result else {
                            'sentiment': 'neutral',
                            'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                            'confidence': 1.0
                        })
                
                # Add sentiment columns
                results_df[f'{column}_sentiment'] = [s['sentiment'] for s in sentiments]
                results_df[f'{column}_confidence'] = [s['confidence'] for s in sentiments]
                
                # Add individual scores
                for score_type in ['negative', 'neutral', 'positive']:
                    results_df[f'{column}_{score_type}_score'] = [
                        s['scores'][score_type] for s in sentiments
                    ]
        
        return results_df

if __name__ == "__main__":
    # Test the sentiment analyzer
    try:
        print("Initializing sentiment analyzer...")
        analyzer = SentimentAnalyzer()
        
        # Test single text analysis
        test_texts = [
            "AAPL is looking very bullish, great earnings and strong growth potential!",
            "The market is crashing, selling everything immediately!",
            "Stock price remained stable throughout the trading session."
        ]
        
        print("\nTesting single text analysis:")
        for text in test_texts:
            result = analyzer.analyze_text(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print("Scores:", {k: f"{v:.2f}" for k, v in result['scores'].items()})
        
        # Test DataFrame analysis
        print("\nTesting DataFrame analysis:")
        test_df = pd.DataFrame({
            'title': ["Bullish on AAPL!", "Market looking bearish", "Neutral market conditions"],
            'text': ["Great growth potential", "Preparing for a crash", "Steady trading day"]
        })
        
        results = analyzer.analyze_dataframe(test_df)
        print("\nResults DataFrame:")
        print(results)
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")