"""Train a TF-IDF text classifier for the chat topic competition.

The script reads train/test CSV files, detects the message text column,
validates a classical text classification pipeline and creates a submission file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


ID_COLUMN = "message_id"
DEFAULT_TARGET_COLUMN = "topic"
TEXT_COLUMN_CANDIDATES = [
    "message",
    "text",
    "message_text",
    "chat_text",
    "content",
    "body",
]


def detect_text_column(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    explicit_text_column: str | None = None,
) -> str:
    """Detect the text column shared by train and test data."""
    if explicit_text_column is not None:
        if explicit_text_column not in train.columns or explicit_text_column not in test.columns:
            raise ValueError(f"Text column '{explicit_text_column}' was not found in both train and test files.")
        return explicit_text_column

    for column in TEXT_COLUMN_CANDIDATES:
        if column in train.columns and column in test.columns:
            return column

    excluded_columns = {ID_COLUMN, target_column}
    shared_columns = [
        column
        for column in train.columns
        if column in test.columns and column not in excluded_columns
    ]

    object_columns = [
        column
        for column in shared_columns
        if train[column].dtype == "object" or test[column].dtype == "object"
    ]

    if len(object_columns) == 1:
        return object_columns[0]

    if object_columns:
        return max(
            object_columns,
            key=lambda column: train[column].fillna("").astype(str).str.len().mean(),
        )

    raise ValueError(
        "Could not detect a text column automatically. "
        "Pass it explicitly with --text-column."
    )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test and sample submission files."""
    train = pd.read_csv(data_dir / "train_chat.csv")
    test = pd.read_csv(data_dir / "test_chat.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission_chat.csv")
    return train, test, sample_submission


def build_model(seed: int) -> Pipeline:
    """Build a TF-IDF + Logistic Regression text classification pipeline."""
    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )

    features = FeatureUnion(
        [
            ("word_tfidf", word_features),
            ("char_tfidf", char_features),
        ]
    )

    classifier = LogisticRegression(
        C=4.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def prepare_text(series: pd.Series) -> pd.Series:
    """Fill missing values and convert text data to strings."""
    return series.fillna("").astype(str)


def validate_model(
    model: Pipeline,
    train: pd.DataFrame,
    text_column: str,
    target_column: str,
    seed: int,
) -> None:
    """Run a stratified validation split and print model quality."""
    X = prepare_text(train[text_column])
    y = train[target_column]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model.fit(X_train, y_train)
    valid_predictions = model.predict(X_valid)
    macro_f1 = f1_score(y_valid, valid_predictions, average="macro")

    print(f"Validation macro F1: {macro_f1:.5f}")
    print(classification_report(y_valid, valid_predictions, digits=4))


def train_final_model(
    train: pd.DataFrame,
    text_column: str,
    target_column: str,
    seed: int,
) -> Pipeline:
    """Train the final model on the full training data."""
    model = build_model(seed)
    X = prepare_text(train[text_column])
    y = train[target_column]
    model.fit(X, y)
    return model


def create_submission(
    model: Pipeline,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    text_column: str,
    target_column: str,
    output_path: Path,
) -> None:
    """Predict test topics and save a competition submission file."""
    X_test = prepare_text(test[text_column])
    predictions = model.predict(X_test)

    submission = sample_submission[[ID_COLUMN]].copy()
    submission[target_column] = predictions
    submission.to_csv(output_path, index=False)


def run_pipeline(
    data_dir: Path,
    output_path: Path,
    seed: int,
    target_column: str,
    text_column: str | None,
    skip_validation: bool,
) -> None:
    """Run validation, full training and submission generation."""
    train, test, sample_submission = load_data(data_dir)
    detected_text_column = detect_text_column(
        train=train,
        test=test,
        target_column=target_column,
        explicit_text_column=text_column,
    )

    print(f"Detected text column: {detected_text_column}")
    print(f"Target column: {target_column}")
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")

    if not skip_validation:
        print("Running validation...")
        validation_model = build_model(seed)
        validate_model(
            model=validation_model,
            train=train,
            text_column=detected_text_column,
            target_column=target_column,
            seed=seed,
        )

    print("Training final model...")
    final_model = train_final_model(
        train=train,
        text_column=detected_text_column,
        target_column=target_column,
        seed=seed,
    )

    print("Creating submission...")
    create_submission(
        model=final_model,
        test=test,
        sample_submission=sample_submission,
        text_column=detected_text_column,
        target_column=target_column,
        output_path=output_path,
    )
    print(f"Saved submission to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF classifier and create a chat topic submission.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--target-column", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation and only train the final model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        seed=args.seed,
        target_column=args.target_column,
        text_column=args.text_column,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
