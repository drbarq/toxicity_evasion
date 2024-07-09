import json
from src import text_generation, toxicity_testing, model_training
from config import TOXIC_TWEETS_PATH, GENERATED_SAMPLES_PATH


def analyze_samples():
    with open(GENERATED_SAMPLES_PATH, 'r') as f:
        samples = json.load(f)
    
    # Analyze word frequency, sentence structure, etc.
    # This is a simplified analysis
    word_freq = {}
    for sample in samples:
        words = sample.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    print("Most common words in generated samples:")
    for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{word}: {count}")
    
    return word_freq


def update_generation_strategy(word_freq):
    # Use the word frequency to adjust the text generation process
    # This is a simplified example - you might want to implement more sophisticated strategies
    common_words = [word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    print(f"Updating generation strategy to avoid these common words: {', '.join(common_words)}")
    
    # Update the text_generation module to avoid these words
    text_generation.update_avoid_words(common_words)


def iterative_learning(num_iterations=5, samples_per_iteration=20):
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Generate new samples
        text_generation.generate_texts(samples_per_iteration)
        
        # Test toxicity
        toxicity_testing.test_toxicity()
        
        # Analyze results
        word_freq = analyze_samples()
        
        # Update generation strategy
        update_generation_strategy(word_freq)
        
        # Check for successful samples
        with open('non_toxic_samples.json', 'r') as f:
            non_toxic_samples = json.load(f)
        
        if non_toxic_samples:
            print(f"Found {len(non_toxic_samples)} successful samples!")
            print("Samples:")
            for sample in non_toxic_samples:
                print(f"- {sample}")
            
            # Add successful samples to training data
            with open(TOXIC_TWEETS_PATH, 'r') as f:
                toxic_tweets = json.load(f)
            
            toxic_tweets.extend(non_toxic_samples)
            
            with open(TOXIC_TWEETS_PATH, 'w') as f:
                json.dump(toxic_tweets, f)
            
            # Retrain the model
            model_training.train_model()
        else:
            print("No successful samples found in this iteration.")
    
    print("\nIterative learning process completed.")


if __name__ == "__main__":
    iterative_learning()