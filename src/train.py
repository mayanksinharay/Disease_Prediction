from preprocessing import prepare_data

from models import get_models

from evaluate import evaluate_model

from utils import (
    create_directories,
    save_model,
    reset_results_csv
)

# --------------------------------------------------
# Create Required Directories
# --------------------------------------------------

create_directories()
reset_results_csv()

# --------------------------------------------------
# Load Data
# No scaling here because Pipelines handle it.
# --------------------------------------------------

X_train, X_test, y_train, y_test, scaler = prepare_data(scale=False)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

models = get_models()

best_accuracy = 0
best_model = None
best_name = None

# --------------------------------------------------
# Train & Evaluate Every Model
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    accuracy = evaluate_model(
        model,
        X_test,
        y_test,
        model_name
    )

    save_model(
        model,
        f"{model_name}.pkl"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_name = model_name

# --------------------------------------------------
# Save Best Model
# --------------------------------------------------

save_model(
    best_model,
    "best_model.pkl"
)

print("\n" + "=" * 60)
print(f"Best Model : {best_name}")
print(f"Accuracy   : {best_accuracy:.4f}")
print("=" * 60)