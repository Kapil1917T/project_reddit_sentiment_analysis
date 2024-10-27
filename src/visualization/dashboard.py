# src/visualization/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.reddit_scraper import RedditScraper
from src.preprocessing.text_processor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer

# Initialize our components
@st.cache_resource
def load_models():
    return {
        'scraper': RedditScraper(),
        'preprocessor': TextPreprocessor(),
        'analyzer': SentimentAnalyzer()
    }

def create_sentiment_plot(df, title):
    """Create sentiment distribution plot"""
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=title,
        color_discrete_map={
            'positive': 'green',
            'neutral': 'gray',
            'negative': 'red'
        }
    )
    return fig

def create_confidence_plot(df):
    """Create confidence score distribution"""
    fig = px.histogram(
        df, 
        x='confidence',
        title='Sentiment Confidence Distribution',
        nbins=20
    )
    return fig

def main():
    st.set_page_config(page_title="Reddit Stock Sentiment Analysis", layout="wide")
    
    # Header
    st.title("Reddit Stock Sentiment Analysis Dashboard")
    st.write("Analyze sentiment of Reddit posts about stocks")
    
    try:
        # Load models
        models = load_models()
        
        # Sidebar
        st.sidebar.header("Configuration")
        stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)").upper()
        subreddits = st.sidebar.multiselect(
            "Select Subreddits",
            ["wallstreetbets", "stocks", "investing"],
            default=["wallstreetbets"]
        )
        days_back = st.sidebar.slider("Days to Analyze", 1, 7, 3)
        
        if stock_symbol and subreddits:
            with st.spinner("Fetching and analyzing Reddit posts..."):
                # Fetch posts
                df = models['scraper'].fetch_stock_related_posts(
                    stock_symbol,
                    subreddits=subreddits
                )
                
                if df.empty:
                    st.warning(f"No posts found for {stock_symbol}")
                    return
                
                # Preprocess text
                df = models['preprocessor'].process_dataframe(df)
                
                # Analyze sentiment
                df = models['analyzer'].analyze_dataframe(df)
                
                # Display results
                st.header(f"Analysis Results for {stock_symbol}")
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Overall Sentiment Distribution")
                    sentiment_fig = create_sentiment_plot(
                        df, 
                        f"Sentiment Distribution for {stock_symbol}"
                    )
                    st.plotly_chart(sentiment_fig)
                
                with col2:
                    st.subheader("Confidence Scores")
                    confidence_fig = create_confidence_plot(df)
                    st.plotly_chart(confidence_fig)
                
                # Display most positive and negative posts
                st.header("Notable Posts")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Positive Posts")
                    positive_posts = df.nlargest(3, 'text_positive_score')
                    for _, post in positive_posts.iterrows():
                        st.write("📈 " + post['title'])
                        st.write("Sentiment Score:", f"{post['text_positive_score']:.2f}")
                        st.write("---")
                
                with col2:
                    st.subheader("Most Negative Posts")
                    negative_posts = df.nlargest(3, 'text_negative_score')
                    for _, post in negative_posts.iterrows():
                        st.write("📉 " + post['title'])
                        st.write("Sentiment Score:", f"{post['text_negative_score']:.2f}")
                        st.write("---")
                
                # Raw data table
                st.header("Detailed Data")
                st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again or contact support if the error persists.")

if __name__ == "__main__":
    main()
