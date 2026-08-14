"""
SBA 928 - Market Research LLM Fine-Tuning
Dataset preparation script.

Creates training, validation, and held-out test datasets
from the complete market research prompt-response dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/market_research_dataset.csv"


def prepare_dataset():
    """Load and split the market research dataset."""

    print("Loading market research dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Total examples: {len(df)}")

    # 70% training, 30% temporary evaluation data
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42
    )

    # Split remaining data approximately equally
    # between validation and held-out testing.
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )

    train_df.to_csv("data/train.csv", index=False)
    validation_df.to_csv("data/validation.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Dataset preparation complete.")
    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(validation_df)}")
    print(f"Held-out test examples: {len(test_df)}")


if __name__ == "__main__":
    prepare_dataset()
