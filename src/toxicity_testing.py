import json
import time
from gradio_client import Client
from config import CENSOR_ME_API, GENERATED_SAMPLES_PATH


def test_toxicity(max_retries=3):
    client = Client(CENSOR_ME_API)
    
    with open(GENERATED_SAMPLES_PATH, 'r') as f:
        generated_samples = json.load(f)
    
    non_toxic_samples = []
    
    for i, sample in enumerate(generated_samples):
        print(f"Testing sample {i+1}/{len(generated_samples)}")
        for attempt in range(max_retries):
            try:
                result = client.predict(
                    sample,
                    0.005,
                    api_name="/censor_me"
                )
                # Assuming the API returns a dictionary with a 'plot' key for toxic content
                if isinstance(result, dict) and 'plot' not in result:
                    non_toxic_samples.append(sample)
                print(f"Sample {i+1} processed successfully.")
                break  # Successful API call, break the retry loop
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Max retries reached for sample {i+1}. Moving to next sample.")
        
        # Add a small delay between API calls to avoid rate limiting
        time.sleep(2)
    
    with open('non_toxic_samples.json', 'w') as f:
        json.dump(non_toxic_samples, f)
    
    print(f"Found {len(non_toxic_samples)} non-toxic samples out of {len(generated_samples)} generated samples.")


if __name__ == "__main__":
    test_toxicity()
