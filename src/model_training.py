import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import MODEL_NAME, TOXIC_TWEETS_PATH


class ToxicTweetsDataset(Dataset):
    def __init__(self, tweets, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for tweet in tweets:
            encodings_dict = tokenizer(tweet, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def train_model():
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    with open(TOXIC_TWEETS_PATH, 'r') as f:
        toxic_tweets = json.load(f)
    
    # Clean the tweets
    clean_tweets = [tweet.replace('<tr>', '').replace('</tr>', '').strip() for tweet in toxic_tweets]
    clean_tweets = [tweet for tweet in clean_tweets if tweet]  # Remove empty tweets

    # Create dataset and dataloader
    dataset = ToxicTweetsDataset(clean_tweets, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss: {loss.item():.4f}", end="\r")
        print()

    # Save the model
    model.save_pretrained("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")
    
    print("Model training completed and saved.")


if __name__ == "__main__":
    train_model()