import json
from collections import Counter
from config import TOXIC_TWEETS_PATH


def analyze_toxic_tweets():
    with open(TOXIC_TWEETS_PATH, 'r') as f:
        toxic_tweets = json.load(f)
    
    # Simple word frequency analysis
    words = ' '.join(toxic_tweets).lower().split()
    word_freq = Counter(words)
    
    print("Top 10 most common words:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")
    
    # Basic statistics
    print(f"\nTotal number of tweets: {len(toxic_tweets)}")
    avg_length = sum(len(tweet.split()) for tweet in toxic_tweets) / len(toxic_tweets)
    print(f"Average tweet length (in words): {avg_length:.2f}")
    
    # Sample tweets
    print("\nSample tweets:")
    for i in range(min(5, len(toxic_tweets))):
        print(f"{i+1}. {toxic_tweets[i]}")


if __name__ == "__main__":
    analyze_toxic_tweets()
