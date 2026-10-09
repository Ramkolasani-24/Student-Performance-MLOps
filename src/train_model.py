import json
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

DATA_PATH = "data/study_performance.csv"

FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course"
]

SCORES = ["math_score", "reading_score", "writing_score"]

df = pd.read_csv(DATA_PATH)

df["average_score"] = df[SCORES].mean(axis=1)
df["result"] = (
    df["average_score"] >= 40
).map({True: "Pass", False: "Needs Support"})

X = df[FEATURES]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        FEATURES
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "macro_f1": float(
        f1_score(y_test, predictions, average="macro", zero_division=0)
    ),
    "needs_support_recall": float(
        recall_score(
            y_test,
            predictions,
            labels=["Needs Support"],
            average="macro",
            zero_division=0
        )
    )
}

os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    model,
    "artifacts/student_performance_model.joblib"
)

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model training completed!")
print(json.dumps(metrics, indent=4))
