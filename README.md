# ML Competition 4: Chat Topic Classification

This repository contains a compact machine learning solution for a text classification competition. The task is to predict the topic of a chat message.

The repository currently includes the competition CSV files and a reproducible baseline pipeline based on TF-IDF features and a linear classifier.

## Task

For each message from the test set, the model predicts one topic label:

```text
message_id, topic
```

The sample submission uses the following format:

```text
message_id,topic
msg_120001,other_chat
msg_120002,other_chat
...
```

## Repository structure

```text
ML-competition-4/
├── README.md
├── requirements.txt
├── .gitignore
├── train_chat.csv
├── test_chat.csv
├── sample_submission_chat.csv
└── src/
    └── train_text_classifier.py
```

## Approach

The solution is designed as a clear and reproducible baseline for text classification:

- load train, test and sample submission CSV files;
- automatically detect the text column;
- clean missing text values;
- convert messages into TF-IDF features;
- train a linear classifier;
- evaluate the model on a validation split;
- predict topics for the test messages;
- save the final `submission.csv` file in the required format.

This approach is intentionally simple and reliable. For many classical text classification tasks, TF-IDF with a linear model is a strong baseline: it is fast, interpretable and easy to reproduce.

## Model

The pipeline uses:

- `TfidfVectorizer` for text features;
- word n-grams from 1 to 2;
- character n-grams from 3 to 5;
- `LogisticRegression` as the classifier;
- a weighted combination of word-level and character-level features through `FeatureUnion`.

Character n-grams help the model handle typos, short messages and informal chat text. Word n-grams capture more meaningful phrases and topic-specific expressions.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script from the repository root:

```bash
python src/train_text_classifier.py --data-dir . --output submission.csv
```

The script expects the following files inside `--data-dir`:

```text
train_chat.csv
test_chat.csv
sample_submission_chat.csv
```

## Main parameters

The script supports the following arguments:

```text
--data-dir       folder with CSV files
--output         path to generated submission file
--seed           random seed
--text-column    optional explicit text column name
--target-column  target column name, default: topic
```

Example with an explicit text column:

```bash
python src/train_text_classifier.py \
  --data-dir . \
  --output submission.csv \
  --text-column message_text
```

## Output

The final submission has two columns:

```text
message_id, topic
```

The script aligns predictions with `sample_submission_chat.csv`, keeps the original `message_id` order and writes the selected output file.

## Main technologies

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Logistic Regression

## Notes

The project is intentionally focused on a clean competition baseline. The code is structured so that the full pipeline can be read from top to bottom: data loading, text column detection, validation, model training, prediction and submission generation.