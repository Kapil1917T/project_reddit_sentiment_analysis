#test_reddit_scraper.py

import pytest
from datetime import datetime
import pandas as pd
from src.data.reddit_scraper import RedditScraper

@pytest.fixture
def scraper():
    return RedditScraper()

def test_scraper_initialization(scraper):
    """Test if scraper initializes correctly"""
    assert scraper is not None
    assert hasattr(scraper, 'reddit')

def test_fetch_subreddit_posts(scraper):
    """Test fetching posts from a subreddit"""
    df = scraper.fetch_subreddit_posts('wallstreetbets', limit=5)
    
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        required_columns = ['id', 'title', 'text', 'score', 'created_utc']
        assert all(col in df.columns for col in required_columns)
        assert len(df) <= 5
        assert isinstance(df['created_utc'].iloc[0], datetime)

def test_fetch_stock_related_posts(scraper):
    """Test fetching stock-related posts"""
    df = scraper.fetch_stock_related_posts('AAPL', subreddits=['stocks'], days=1)
    
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert all(df['created_utc'] >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
        assert 'AAPL' in ' '.join(df['title'].str.upper().tolist() + df['text'].str.upper().tolist())

def test_get_post_comments(scraper):
    """Test fetching comments from a post"""
    # First get a post
    posts_df = scraper.fetch_subreddit_posts('wallstreetbets', limit=1)
    if not posts_df.empty:
        post_id = posts_df.iloc[0]['id']
        comments_df = scraper.get_post_comments(post_id, limit=3)
        
        assert isinstance(comments_df, pd.DataFrame)
        if not comments_df.empty:
            assert 'text' in comments_df.columns
            assert 'score' in comments_df.columns
            assert len(comments_df) <= 3
