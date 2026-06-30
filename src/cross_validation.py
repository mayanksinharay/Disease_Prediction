import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from models import get_models


df = pd.read_csv("dataset/heart.csv")
df = df.drop_duplicates()
X = df.drop("target", axis=1)
y = df["target"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]


models = get_models()

results = []

for model_name, model in models.items():

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring
    )

    accuracy = scores["test_accuracy"].mean()
    precision = scores["test_precision"].mean()
    recall = scores["test_recall"].mean()
    f1 = scores["test_f1"].mean()

    std = scores["test_accuracy"].std()

    print(f"Accuracy : {accuracy:.4f} ± {std:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Std": std,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    "reports/metrics/cross_validation_results.csv",
    index=False
)

print("\nCross Validation Complete!")

print(results_df)

