# tests/test_sentiment_analyzer.py

import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_analyzer_initialization(analyzer):
    """Test if analyzer initializes correctly"""
    assert analyzer is not None
    assert hasattr(analyzer, 'model')
    assert hasattr(analyzer, 'tokenizer')

def test_single_text_analysis(analyzer):
    """Test sentiment analysis on single text"""
    text = "AAPL is showing strong growth potential"
    result = analyzer.analyze_text(text)
    
    assert result is not None
    assert 'sentiment' in result
    assert 'confidence' in result
    assert 'scores' in result
    assert all(k in result['scores'] for k in ['positive', 'negative', 'neutral'])

def test_dataframe_analysis(analyzer):
    """Test sentiment analysis on DataFrame"""
    test_df = pd.DataFrame({
        'title': ["Bullish market", "Bearish trends"],
        'text': ["Strong growth", "Market crash"]
    })
    
    results = analyzer.analyze_dataframe(test_df)
    
    assert len(results) == len(test_df)
    assert 'title_sentiment' in results.columns
    assert 'text_sentiment' in results.columns
    assert 'title_confidence' in results.columns
    assert 'text_confidence' in results.columns