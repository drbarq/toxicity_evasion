import sys
from src import data_collection, data_analysis, model_training, iterative_learning


def main():
    steps = [
        ("Collecting toxic tweets", data_collection.fetch_toxic_tweets, 1000),
        ("Analyzing toxic tweets", data_analysis.analyze_toxic_tweets),
        ("Training the initial model", model_training.train_model),
        ("Starting iterative learning process", iterative_learning.iterative_learning, 5, 20),
    ]

    for i, (step_name, step_func, *args) in enumerate(steps, 1):
        print(f"\nStep {i}: {step_name}...")
        try:
            if step_name == "Starting iterative learning process":
                iterative_learning.iterative_learning(*args)
            else:
                step_func(*args)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred during {step_name.lower()}: {str(e)}")
            print("Skipping to the next step.")

    print("\nProcess completed. Check generated_samples.json and non_toxic_samples.json for results.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)