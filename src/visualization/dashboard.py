# src/visualization/dashboard.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Fix the import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import our modules
from src.data.reddit_scraper import RedditScraper
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer

# Initialize our components
@st.cache_resource
def load_models():
    """Load and cache our models"""
    return {
        'scraper': RedditScraper(),
        'preprocessor': TextPreprocessor(),
        'analyzer': SentimentAnalyzer()
    }

def create_sentiment_plot(df, column_prefix='text'):
    """Create sentiment distribution pie chart"""
    sentiment_counts = df[f'{column_prefix}_sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=f'Sentiment Distribution',
        color_discrete_map={
            'positive': '#2ECC71',
            'neutral': '#95A5A6',
            'negative': '#E74C3C'
        }
    )
    return fig

def create_confidence_histogram(df, column_prefix='text'):
    """Create confidence score histogram"""
    fig = px.histogram(
        df,
        x=f'{column_prefix}_confidence',
        title='Sentiment Confidence Distribution',
        color=f'{column_prefix}_sentiment',
        color_discrete_map={
            'positive': '#2ECC71',
            'neutral': '#95A5A6',
            'negative': '#E74C3C'
        }
    )
    return fig

def create_competitor_comparison(analyzers, stock_symbol, competitors):
    """Create comparison chart with competitors"""
    comparison_data = []
    
    # Analyze main stock and competitors
    for symbol in [stock_symbol] + competitors:
        df = analyzers['scraper'].fetch_stock_related_posts(symbol, days=7)
        if not df.empty:
            df = analyzers['analyzer'].analyze_dataframe(df)
            sentiment_scores = {
                'symbol': symbol,
                'positive_ratio': (df['text_sentiment'] == 'positive').mean() * 100,
                'post_count': len(df),
                'avg_score': df['score'].mean(),
                'avg_confidence': df['text_confidence'].mean()
            }
            comparison_data.append(sentiment_scores)
    
    comp_df = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    
    # Add bars for positive sentiment
    fig.add_trace(go.Bar(
        name='Positive Sentiment %',
        x=comp_df['symbol'],
        y=comp_df['positive_ratio'],
        marker_color='#2ECC71'
    ))
    
    # Add markers for average confidence
    fig.add_trace(go.Scatter(
        name='Avg Confidence',
        x=comp_df['symbol'],
        y=comp_df['avg_confidence'] * 100,
        mode='markers',
        marker=dict(size=12, symbol='diamond', color='#E74C3C')
    ))
    
    fig.update_layout(
        title=f'Sentiment Comparison with Competitors',
        barmode='group',
        yaxis_title='Percentage (%)'
    )
    
    return fig

def create_volume_sentiment_correlation(df):
    """Create scatter plot of post volume vs sentiment"""
    try:
        df['date'] = pd.to_datetime(df['created_utc']).dt.date
        daily_data = df.groupby('date').agg({
            'id': 'count',
            'text_sentiment': lambda x: (x == 'positive').mean() * 100,
            'score': 'mean'
        }).reset_index()
        
        # Create scatter plot without trendline if statsmodels is not available
        fig = px.scatter(
            daily_data,
            x='id',
            y='text_sentiment',
            size='score',
            title='Post Volume vs Sentiment Correlation',
            labels={
                'id': 'Number of Posts',
                'text_sentiment': 'Positive Sentiment %',
                'score': 'Average Score'
            }
        )
        
        try:
            import statsmodels.api as sm
            # Add trendline if statsmodels is available
            X = daily_data['id']
            y = daily_data['text_sentiment']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            daily_data['trendline'] = model.predict(X)
            
            fig.add_trace(
                go.Scatter(
                    x=daily_data['id'],
                    y=daily_data['trendline'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
        except ImportError:
            pass  # Skip trendline if statsmodels is not available
        
        return fig
    except Exception as e:
        print(f"Error creating volume-sentiment correlation: {str(e)}")
        # Return empty figure in case of error
        return go.Figure().update_layout(
            title='Volume vs Sentiment Correlation (Error occurred)'
        )

def create_sentiment_heatmap(df):
    """Create hourly sentiment heatmap"""
    df['hour'] = pd.to_datetime(df['created_utc']).dt.hour
    df['day'] = pd.to_datetime(df['created_utc']).dt.day_name()
    
    sentiment_pivot = pd.pivot_table(
        df,
        values='text_confidence',
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=sentiment_pivot.values,
        x=sentiment_pivot.columns,
        y=sentiment_pivot.index,
        colorscale='RdYlGn'
    ))
    
    fig.update_layout(
        title='Sentiment Heatmap by Hour and Day',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week'
    )
    return fig

def main():
    st.set_page_config(
        page_title="Reddit Stock Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.title("🚀 Reddit Stock Sentiment Analysis Dashboard")
    st.write("Analyze sentiment of Reddit posts about stocks")
    
    try:
        # Load models
        with st.spinner("Loading models..."):
            models = load_models()
        
        # Sidebar configuration
        st.sidebar.header("📊 Analysis Configuration")
        
        stock_symbol = st.sidebar.text_input(
            "Enter Stock Symbol (e.g., AAPL)",
            value="AAPL"
        ).upper()
        
        competitors = st.sidebar.multiselect(
            "Select Competitors for Comparison",
            ["MSFT", "GOOGL", "AMZN", "META", "NVDA"],
            default=["MSFT", "GOOGL"]
        )
        
        subreddits = st.sidebar.multiselect(
            "Select Subreddits",
            ["wallstreetbets", "stocks", "investing"],
            default=["wallstreetbets"]
        )
        
        days_back = st.sidebar.slider(
            "Days to Analyze",
            min_value=1,
            max_value=7,
            value=3
        )
        
        min_score = st.sidebar.slider(
            "Minimum Post Score",
            min_value=1,
            max_value=1000,
            value=10
        )
        
        if st.sidebar.button("🔄 Run Analysis"):
            if stock_symbol and subreddits:
                # Fetch and analyze data
                with st.spinner(f"Fetching and analyzing posts about {stock_symbol}..."):
                    # Get posts
                    df = models['scraper'].fetch_stock_related_posts(
                        stock_symbol,
                        subreddits=subreddits,
                        days=days_back
                    )
                    
                    if df.empty:
                        st.warning(f"No posts found for {stock_symbol}")
                        return
                    
                    # Filter by score
                    df = df[df['score'] >= min_score]
                    
                    # Process text
                    df = models['preprocessor'].process_dataframe(df)
                    
                    # Analyze sentiment
                    df = models['analyzer'].analyze_dataframe(df)
                    
                    # Display results
                    st.header(f"📈 Analysis Results for ${stock_symbol}")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Posts", len(df))
                    with col2:
                        positive_pct = (df['text_sentiment'] == 'positive').mean() * 100
                        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
                    with col3:
                        avg_score = df['score'].mean()
                        st.metric("Average Score", f"{avg_score:.1f}")
                    with col4:
                        avg_comments = df['num_comments'].mean()
                        st.metric("Average Comments", f"{avg_comments:.1f}")
                    
                    # Original visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_sentiment_plot(df),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_confidence_histogram(df),
                            use_container_width=True
                        )
                    
                    # New comparative visualizations
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.plotly_chart(
                            create_volume_sentiment_correlation(df),
                            use_container_width=True
                        )
                    
                    with col4:
                        st.plotly_chart(
                            create_competitor_comparison(models, stock_symbol, competitors),
                            use_container_width=True
                        )
                    
                    # Sentiment heatmap (full width)
                    st.plotly_chart(
                        create_sentiment_heatmap(df),
                        use_container_width=True
                    )
                    
                    # Most discussed posts
                    st.header("📝 Most Discussed Posts")
                    
                    # Positive posts
                    st.subheader("Most Positive Posts")
                    positive_posts = df[df['text_sentiment'] == 'positive'].nlargest(3, 'score')
                    for _, post in positive_posts.iterrows():
                        with st.expander(f"💚 {post['title']}", expanded=False):
                            st.write(f"**Subreddit:** r/{post['subreddit']}")
                            st.write(f"**Score:** {post['score']}")
                            st.write(f"**Confidence:** {post['text_confidence']:.2f}")
                            st.write(f"**Text:** {post['text']}")
                    
                    # Negative posts
                    st.subheader("Most Negative Posts")
                    negative_posts = df[df['text_sentiment'] == 'negative'].nlargest(3, 'score')
                    for _, post in negative_posts.iterrows():
                        with st.expander(f"❌ {post['title']}", expanded=False):
                            st.write(f"**Subreddit:** r/{post['subreddit']}")
                            st.write(f"**Score:** {post['score']}")
                            st.write(f"**Confidence:** {post['text_confidence']:.2f}")
                            st.write(f"**Text:** {post['text']}")
                    
                    # Raw data
                    if st.checkbox("Show Raw Data"):
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download Data as CSV",
                            csv,
                            f"reddit_sentiment_{stock_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv",
                            key='download-csv'
                        )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again or contact support if the error persists.")

if __name__ == "__main__":
    main()