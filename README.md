# Toxicity Evasion Project

## Overview

This project aims to explore the challenge of generating toxic content that might be classified as non-toxic by AI moderation systems. It uses machine learning techniques to analyze toxic tweets, generate new text samples, and iteratively improve the generation process. The goal is to understand and potentially improve AI content moderation systems.

**Note:** This project is for research and educational purposes only. The creation and use of toxic content can be harmful. Always use this knowledge responsibly and ethically.

## Project Structure

```
toxicity_evasion/
├── data/
│   ├── toxic_tweets.json
│   └── generated_samples.json
├── src/
│   ├── data_collection.py
│   ├── data_analysis.py
│   ├── model_training.py
│   ├── text_generation.py
│   ├── toxicity_testing.py
│   └── iterative_learning.py
├── config.py
├── main.py
└── requirements.txt
```

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/toxicity_evasion.git
   cd toxicity_evasion
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the toxicity evasion process:

```
python main.py
```

This script will:

1. Collect toxic tweets
2. Analyze the collected data
3. Train a language model
4. Generate new text samples
5. Test the toxicity of generated samples
6. Iteratively improve the generation process

## Components

- `data_collection.py`: Fetches toxic tweets from an API
- `data_analysis.py`: Analyzes the collected toxic tweets, focusing on offensive language
- `model_training.py`: Trains a GPT-2 model on the collected toxic tweets
- `text_generation.py`: Generates new text samples using the trained model
- `toxicity_testing.py`: Tests the toxicity of generated samples using an API
- `iterative_learning.py`: Manages the iterative learning process to improve text generation

## Configuration

Edit `config.py` to modify:

- API endpoints
- Model parameters
- File paths
- Other configuration options

## Results

The project generates two main output files:

- `data/generated_samples.json`: Contains the generated text samples
- `non_toxic_samples.json`: Contains generated samples that were classified as non-toxic

## Ethical Considerations

This project deals with potentially offensive and harmful content. It's crucial to use this tool responsibly and not for creating or spreading harmful content. The primary purpose is to understand and improve content moderation systems.

## Contributors

Joe Tustin

## Acknowledgments

- Thanks to the creators of the GPT-2 model and the Hugging Face Transformers library
- Acknowledgment to the providers of the toxic tweet dataset and toxicity testing API
