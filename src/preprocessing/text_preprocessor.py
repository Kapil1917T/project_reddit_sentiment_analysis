import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Explicitly download required NLTK data at the start
print("Checking NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("NLTK data check complete")

class TextPreprocessor:
    def __init__(self):
        """Initialize text preprocessing tools"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stock-related terms to keep
        self.custom_terms = {
            'bullish', 'bearish', 'calls', 'puts', 'buy', 'sell',
            'long', 'short', 'hold', 'moon', 'dip', 'rip', 'dump'
        }
        # Remove these terms from stopwords to keep them in our analysis
        self.stop_words -= self.custom_terms

    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers but keep stock tickers (e.g., AMD)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def process_text(self, text):
        """Complete text processing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(processed_tokens)
    
    def process_dataframe(self, df, text_columns=['title', 'text']):
        """Process text columns in a DataFrame"""
        df_processed = df.copy()
        
        for column in text_columns:
            if column in df.columns:
                df_processed[f'{column}_processed'] = df[column].apply(self.process_text)
        
        return df_processed
    
    def extract_sentiment_features(self, text):
        """Extract basic sentiment features from text"""
        # Convert to lowercase for consistent matching
        text = text.lower()
        
        # Define sentiment indicators
        bullish_words = {'buy', 'bull', 'bullish', 'long', 'moon', 'rocket', 'up'}
        bearish_words = {'sell', 'bear', 'bearish', 'short', 'down', 'put', 'dump'}
        
        # Count occurrences
        words = set(text.split())
        features = {
            'bullish_count': len(words.intersection(bullish_words)),
            'bearish_count': len(words.intersection(bearish_words)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'word_count': len(text.split())
        }
        
        return features

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()

    # Example text
    sample_text = """
    🚀🚀🚀 YOLO'd $10k into AAPL calls! The stock is looking super BULLISH 
    and I think it's going to the moon! 🌙 Should I buy more if it dips below $170? 
    https://example.com Check out these gains! 📈
    """
    
    # Test basic processing
    processed_text = preprocessor.process_text(sample_text)
    print("\nOriginal text:")
    print(sample_text)
    print("\nProcessed text:")
    print(processed_text)

    # Test feature extraction
    features = preprocessor.extract_sentiment_features(sample_text)
    print("\nExtracted features:")
    for feature, value in features.items():
        print(f"{feature}: {value}")
    
    # Test with a small DataFrame
    test_df = pd.DataFrame({
        'title': ['AAPL to the moon! 🚀', 'Bearish on TSLA'],
        'text': ['Just bought more calls!', 'Time to short this stock']
    })
    
    processed_df = preprocessor.process_dataframe(test_df)
    print("\nProcessed DataFrame:")
    print(processed_df)