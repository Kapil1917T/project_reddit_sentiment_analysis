# src/data/reddit_scraper.py

import praw
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

class RedditScraper:
    def __init__(self):
        """Initialize Reddit API client"""
        print("\n=== Initializing Reddit Scraper ===")
        # Load environment variables
        load_dotenv()
        
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            print("✓ Reddit API connection established")
        except Exception as e:
            print(f"✗ Error connecting to Reddit API: {str(e)}")
            raise

    def fetch_subreddit_posts(self, subreddit_name, time_filter='week', limit=100):
        """
        Fetch posts from specified subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            time_filter (str): One of [hour, day, week, month, year, all]
            limit (int): Number of posts to fetch
        
        Returns:
            pd.DataFrame: DataFrame containing post data
        """
        print(f"\nFetching posts from r/{subreddit_name}")
        posts_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                posts_data.append({
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'upvote_ratio': post.upvote_ratio,
                    'url': post.url,
                    'subreddit': subreddit_name
                })
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
            
            df = pd.DataFrame(posts_data)
            print(f"✓ Successfully fetched {len(df)} posts from r/{subreddit_name}")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching posts from r/{subreddit_name}: {str(e)}")
            return pd.DataFrame()

    def fetch_stock_related_posts(self, stock_symbol, subreddits=['wallstreetbets', 'stocks', 'investing'], days=7):
        """Fetch posts related to a specific stock"""
        all_posts = pd.DataFrame()
        found_posts = False
        
        print(f"\nSearching for posts about {stock_symbol}")
        
        for subreddit_name in subreddits:
            try:
                # Fetch posts from subreddit
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_data = []
                
                # Try different time filters to find posts
                for time_filter in ['day', 'week', 'month']:
                    for post in subreddit.top(time_filter=time_filter, limit=100):
                        # Case-insensitive search in title and text
                        if (stock_symbol.lower() in post.title.lower() or 
                            (post.selftext and stock_symbol.lower() in post.selftext.lower())):
                            posts_data.append({
                                'id': post.id,
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': datetime.fromtimestamp(post.created_utc),
                                'upvote_ratio': post.upvote_ratio,
                                'subreddit': subreddit_name,
                                'url': f"https://reddit.com{post.permalink}"
                            })
                    
                    if posts_data:  # If found posts, no need to check other time filters
                        break
                
                if posts_data:
                    subreddit_df = pd.DataFrame(posts_data)
                    print(f"Found {len(subreddit_df)} posts about {stock_symbol} in r/{subreddit_name}")
                    all_posts = pd.concat([all_posts, subreddit_df], ignore_index=True)
                    found_posts = True
                else:
                    print(f"No posts about {stock_symbol} found in r/{subreddit_name}")
                
            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue
        
        # Process final results
        if not all_posts.empty:
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            all_posts = all_posts[all_posts['created_utc'] >= cutoff_date]
            
            # Remove duplicates
            all_posts = all_posts.drop_duplicates(subset=['id'])
            
            # Sort by score
            all_posts = all_posts.sort_values('score', ascending=False)
            
            final_count = len(all_posts)
            if final_count > 0:
                print(f"\nFinal results for {stock_symbol}:")
                print(f"Total posts found: {final_count}")
                return all_posts.reset_index(drop=True)
        
        print(f"\n No posts found about {stock_symbol}")
        return pd.DataFrame()

    def get_post_comments(self, post_id, limit=100):
        """
        Fetch comments for a specific post
        
        Args:
            post_id (str): Reddit post ID
            limit (int): Maximum number of comments to fetch
            
        Returns:
            pd.DataFrame: DataFrame containing comments
        """
        try:
            submission = self.reddit.submission(id=post_id)
            comments_data = []
            
            submission.comments.replace_more(limit=0)  # Remove MoreComments objects
            
            for comment in submission.comments[:limit]:
                comments_data.append({
                    'comment_id': comment.id,
                    'post_id': post_id,
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc)
                })
            
            return pd.DataFrame(comments_data)
            
        except Exception as e:
            print(f"Error fetching comments for post {post_id}: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test the scraper
    try:
        print("Testing Reddit Scraper...")
        scraper = RedditScraper()
        
        # Test 1: Fetch posts from a single subreddit
        print("\nTest 1: Fetching recent posts from r/wallstreetbets")
        wsb_posts = scraper.fetch_subreddit_posts('wallstreetbets', limit=5)
        if not wsb_posts.empty:
            print("\nSample post:")
            print(f"Title: {wsb_posts.iloc[0]['title']}")
            print(f"Score: {wsb_posts.iloc[0]['score']}")
            print(f"Comments: {wsb_posts.iloc[0]['num_comments']}")
        
        # Test 2: Search for stock-related posts
        print("\nTest 2: Searching for posts about AAPL")
        apple_posts = scraper.fetch_stock_related_posts('AAPL', days=3)
        if not apple_posts.empty:
            print("\nMost popular post about AAPL:")
            print(f"Title: {apple_posts.iloc[0]['title']}")
            print(f"Subreddit: r/{apple_posts.iloc[0]['subreddit']}")
            print(f"Score: {apple_posts.iloc[0]['score']}")
            
            # Test 3: Fetch comments for the top post
            print("\nTest 3: Fetching comments from top post")
            comments = scraper.get_post_comments(apple_posts.iloc[0]['id'], limit=3)
            if not comments.empty:
                print(f"\nFound {len(comments)} comments")
                print("Sample comment:")
                print(comments.iloc[0]['text'][:200])
        
    except Exception as e:
        print(f"\nError in testing: {str(e)}")
