"""
SBA 928 - Market Research LLM Fine-Tuning

Evaluation utility for comparing generated responses
against expected market research responses.
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


COMPARISON_FILE = "outputs/model_comparison.csv"


def similarity_score(reference, candidate):
    """Calculate TF-IDF cosine similarity."""

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [reference, candidate]
    )

    return cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]


def evaluate_models():

    comparison_df = pd.read_csv(
        COMPARISON_FILE
    )

    base_scores = []
    fine_tuned_scores = []

    for _, row in comparison_df.iterrows():

        base_score = similarity_score(
            row["expected_response"],
            row["base_model_response"]
        )

        fine_score = similarity_score(
            row["expected_response"],
            row["fine_tuned_response"]
        )

        base_scores.append(base_score)
        fine_tuned_scores.append(fine_score)

    comparison_df["base_similarity"] = base_scores

    comparison_df[
        "fine_tuned_similarity"
    ] = fine_tuned_scores

    comparison_df.to_csv(
        "outputs/model_comparison_with_scores.csv",
        index=False
    )

    print(
        "Average Base Model Similarity:",
        comparison_df["base_similarity"].mean()
    )

    print(
        "Average Fine-Tuned Model Similarity:",
        comparison_df[
            "fine_tuned_similarity"
        ].mean()
    )


if __name__ == "__main__":
    evaluate_models()
