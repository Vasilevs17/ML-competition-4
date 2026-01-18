# Решение задачи классификации сообщений домового чата.
# Скопируйте этот файл целиком в одну ячейку Google Colab.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# --- Настройки путей ---
TRAIN_PATHS = ["train.csv", "train_chat.csv"]
TEST_PATHS = ["test.csv", "test_chat.csv"]
SUBMISSION_PATH = "submission.csv"


def _find_first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Не найден файл среди: {paths}")


# --- Загрузка данных ---
train_path = _find_first_existing(TRAIN_PATHS)
test_path = _find_first_existing(TEST_PATHS)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- Признаки ---
# Текстовые: TF-IDF по словам + TF-IDF по символам
# Числовые: day и hour добавим как отдельные признаки через соединение матриц.

text_word = TfidfVectorizer(
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),
    analyzer="word",
)

text_char = TfidfVectorizer(
    min_df=2,
    max_df=0.9,
    ngram_range=(3, 5),
    analyzer="char",
)

# Числовые признаки (day, hour)
num_features = FunctionTransformer(lambda x: x[["day", "hour"]].values, validate=False)

features = FeatureUnion([
    ("word_tfidf", text_word),
    ("char_tfidf", text_char),
])

# Подготовим отдельные матрицы, затем объединим их вручную
from scipy.sparse import hstack

X_text = features.fit_transform(train_df["text"].fillna(""))
X_num = num_features.transform(train_df)
X_train = hstack([X_text, X_num])

y_train = train_df["topic"]

# --- Модель ---
# Логистическая регрессия обычно дает отличное качество на таком типе задач.
model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    C=6.0,
)

# Для контроля качества можно сделать валидацию
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)
val_f1 = f1_score(y_val, val_pred, average="macro")
print(f"Валидация Macro-F1: {val_f1:.4f}")

# Обучение на всем train
model.fit(X_train, y_train)

# --- Предсказание ---
X_test_text = features.transform(test_df["text"].fillna(""))
X_test_num = num_features.transform(test_df)
X_test = hstack([X_test_text, X_test_num])

test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "message_id": test_df["message_id"],
    "topic": test_pred,
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Файл сабмита сохранен: {SUBMISSION_PATH}")
