import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import prepare_data
from models import get_models
from utils import create_directories, save_plot


# --------------------------------------------------
# Create Required Directories
# --------------------------------------------------

create_directories()

# --------------------------------------------------
# Load Data
# --------------------------------------------------

X_train, X_test, y_train, y_test, _ = prepare_data(scale=False)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

models = get_models()

# ==================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

print("=" * 60)
print("Random Forest Feature Importance")
print("=" * 60)

rf = models["Random Forest"]

rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print(importance_df)

# --------------------------------------------------
# Plot Feature Importance
# --------------------------------------------------

plt.figure(figsize=(10, 6))

bars = plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center"
    )

save_plot(
    "feature_importance_rf.png",
    show=True
)

# ==================================================
# TOP 10 MOST IMPORTANT FEATURES
# ==================================================

top10 = importance_df.head(10)

plt.figure(figsize=(9, 6))

bars = plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Top 10 Most Important Features")
plt.xlabel("Importance")

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center"
    )

save_plot(
    "top10_features.png",
    show=True
)

# ==================================================
# LOGISTIC REGRESSION COEFFICIENTS
# ==================================================

print("\n")
print("=" * 60)
print("Logistic Regression Coefficients")
print("=" * 60)

lr = models["Logistic Regression"]

lr.fit(X_train, y_train)

coefficients = lr.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coefficients
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n")
print(coef_df)

# --------------------------------------------------
# Plot Logistic Regression Coefficients
# --------------------------------------------------

plt.figure(figsize=(10, 6))

colors = [
    "green" if value > 0 else "red"
    for value in coef_df["Coefficient"]
]

bars = plt.barh(
    coef_df["Feature"],
    coef_df["Coefficient"],
    color=colors
)

plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Coefficient")

for bar in bars:

    width = bar.get_width()

    if width >= 0:

        plt.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center"
        )

    else:

        plt.text(
            width - 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center"
        )

save_plot(
    "logistic_coefficients.png",
    show=True
)

print("\n")
print("=" * 60)
print("Feature Importance Analysis Complete!")
print("=" * 60)