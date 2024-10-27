import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the BERT sentiment analyzer"""
        # Force CPU usage
        self.device = "cpu"
        print(f"Using device: {self.device}")
        
        # Load FinBERT
        self.model_name = "ProsusAI/finbert"
        try:
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
                torch_dtype=torch.float32,
                trust_remote_code=False
            )
            self.model.to(self.device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def analyze_text(self, text):
        """Analyze sentiment of a single text"""
        try:
            if not isinstance(text, str) or not text.strip():
                return self._get_default_result()
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            # Get prediction
            self.model.eval()  # Ensure model is in evaluation mode
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.cpu().detach().numpy()[0]  # Explicitly detach
            
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
            return self._get_default_result()
    
    def _get_default_result(self):
        """Get default sentiment result"""
        return {
            'sentiment': 'neutral',
            'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
            'confidence': 1.0
        }

    def analyze_dataframe(self, df, text_columns=['title', 'text']):
        """Analyze sentiment for text columns in a DataFrame"""
        results_df = df.copy()
        
        for column in text_columns:
            if column in df.columns:
                print(f"\nAnalyzing sentiment for {column}...")
                sentiments = []
                
                try:
                    # Process texts in batches to avoid memory issues
                    batch_size = 32
                    texts = df[column].tolist()
                    for i in tqdm(range(0, len(texts), batch_size), desc=f"Analyzing {column}"):
                        batch_texts = texts[i:i + batch_size]
                        batch_sentiments = [
                            self.analyze_text(text) if pd.notna(text) and text 
                            else self._get_default_result()
                            for text in batch_texts
                        ]
                        sentiments.extend(batch_sentiments)
                    
                    # Add results to DataFrame
                    results_df[f'{column}_sentiment'] = [s['sentiment'] for s in sentiments]
                    results_df[f'{column}_confidence'] = [s['confidence'] for s in sentiments]
                    
                    for score_type in ['negative', 'neutral', 'positive']:
                        results_df[f'{column}_{score_type}_score'] = [
                            s['scores'][score_type] for s in sentiments
                        ]
                        
                except Exception as e:
                    print(f"Error processing column {column}: {str(e)}")
                    # Add default values in case of error
                    results_df[f'{column}_sentiment'] = 'neutral'
                    results_df[f'{column}_confidence'] = 1.0
                    for score_type in ['negative', 'neutral', 'positive']:
                        results_df[f'{column}_{score_type}_score'] = 0.0
        
        return results_df

if __name__ == "__main__":
    try:
        print("Initializing sentiment analyzer...")
        analyzer = SentimentAnalyzer()
        
        # Test with a simple example first
        print("\nTesting single text analysis:")
        test_text = "AAPL is looking very bullish!"
        result = analyzer.analyze_text(test_text)
        if result:
            print(f"Text: {test_text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2f}")
        
        # If single text works, test DataFrame
        print("\nTesting DataFrame analysis:")
        test_df = pd.DataFrame({
            'title': ["Bullish on AAPL!"],
            'text': ["Great growth potential"]
        })
        
        results = analyzer.analyze_dataframe(test_df)
        print("\nResults DataFrame:")
        print(results)
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")