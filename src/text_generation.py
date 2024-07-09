import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import MAX_LENGTH, GENERATED_SAMPLES_PATH

avoid_words = set()

def update_avoid_words(words):
    global avoid_words
    avoid_words.update(words)

def generate_texts(num_samples, max_attempts=50):
    global avoid_words
    model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated_samples = []
    
    for i in range(num_samples):
        prompt = "I think that"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
        max_length = min(MAX_LENGTH, 100)
        
        for attempt in range(max_attempts):
            try:
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                )

                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                
                if attempt < max_attempts // 2:
                    if not any(word in generated_text.lower() for word in avoid_words):
                        generated_samples.append(generated_text)
                        print(f"Generated sample {i+1}: {generated_text}")
                        break
                    else:
                        print(f"Regenerating sample {i+1} due to avoided words (attempt {attempt + 1})")
                else:
                    # Relax constraints for the second half of attempts
                    generated_samples.append(generated_text)
                    print(f"Generated sample {i+1} (with relaxed constraints): {generated_text}")
                    break
            
            except Exception as e:
                print(f"Error generating sample {i+1}: {str(e)}")
                break
        
        if attempt == max_attempts - 1:
            print(f"Failed to generate sample {i+1} after {max_attempts} attempts")
    
    with open(GENERATED_SAMPLES_PATH, 'w') as f:
        json.dump(generated_samples, f)
    
    print(f"Generated {len(generated_samples)} text samples.")

if __name__ == "__main__":
    generate_texts(20)