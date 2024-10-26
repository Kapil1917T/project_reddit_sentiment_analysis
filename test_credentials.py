import praw
from dotenv import load_dotenv
import os
import time

def test_reddit_connection():
    # Load environment variables
    load_dotenv()
    
    # Print debugging information first
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    print("\nChecking credentials:")
    print(f"Client ID present: {'Yes' if client_id else 'No'}")
    print(f"Client Secret present: {'Yes' if client_secret else 'No'}")
    print(f"User Agent present: {'Yes' if user_agent else 'No'}")
    
    try:
        # Initialize Reddit instance in read-only mode
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True  # Explicitly set read-only mode
        )
        
        print("\nInitial connection successful!")
        
        try:
            # Test simple subreddit access first
            subreddit = reddit.subreddit('wallstreetbets')
            print(f"\nAccessing basic subreddit info...")
            
            # Access basic information with error handling
            try:
                display_name = subreddit.display_name
                print(f"Successfully accessed r/{display_name}")
            except Exception as e:
                print(f"Error accessing display name: {str(e)}")
                
            try:
                subscribers = subreddit.subscribers
                print(f"Number of subscribers: {subscribers:,}")
            except Exception as e:
                print(f"Error accessing subscriber count: {str(e)}")
            
            # Add a small delay before fetching posts
            time.sleep(5)
            
            print("\nTrying to fetch posts...")
            posts_list = list(subreddit.hot(limit=2))  # Reduce limit to 2 posts
            for post in posts_list:
                print(f"- Post title: {post.title[:50]}...")
                
            print("\nAll tests completed successfully!")
            return True
            
        except Exception as sub_error:
            print(f"\nError accessing subreddit data: {str(sub_error)}")
            print("This might be due to rate limiting or permission issues.")
            return False
            
    except Exception as e:
        print(f"\nError with initial Reddit connection: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify your Reddit application is of type 'script'")
        print("2. Check client ID and secret are correct")
        print("3. Ensure user agent format is: 'script:your_app_name:v1.0 (by /u/username)'")
        print("4. Try recreating your application at https://www.reddit.com/prefs/apps")
        return False

if __name__ == "__main__":
    test_reddit_connection()