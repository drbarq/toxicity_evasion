import json
import time
import random
from gradio_client import Client
from config import FETCH_TOXIC_TWEETS_API, TOXIC_TWEETS_PATH


def fetch_toxic_tweets(num_samples, max_retries=5):
    client = Client(FETCH_TOXIC_TWEETS_API)
    toxic_tweets = []
    
    print(f"Attempting to collect {num_samples} toxic tweets...")
    
    while len(toxic_tweets) < num_samples:
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = client.predict(api_name="/fetch_toxic_tweets")
                elapsed_time = time.time() - start_time
                
                new_tweets = result[0].split('\n')
                toxic_tweets.extend(new_tweets)
                
                print(f"Collected {len(new_tweets)} tweets in {elapsed_time:.2f} seconds. Total: {len(toxic_tweets)}")
                
                time.sleep(random.uniform(0.5, 1.5))  # Random delay between requests
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Moving to next batch.")
        
        if len(toxic_tweets) >= num_samples:
            break
    
    # Trim to exact number of samples requested
    toxic_tweets = toxic_tweets[:num_samples]
    
    with open(TOXIC_TWEETS_PATH, 'w') as f:
        json.dump(toxic_tweets, f)
    
    print(f"Collection completed. Saved {len(toxic_tweets)} toxic tweets.")


if __name__ == "__main__":
    fetch_toxic_tweets(1000)  # Collect 1000 tweets for testing