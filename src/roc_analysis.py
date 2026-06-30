import matplotlib.pyplot as plt
from preprocessing import prepare_data
from models import get_models
from visualization import plot_roc_curve
from utils import save_plot
X_train, X_test, y_train, y_test, scaler = prepare_data()
models = get_models()
plt.figure(figsize=(9,7))
for name, model in models.items():
    model.fit(X_train, y_train)
    plot_roc_curve(
        model,
        X_test,
        y_test,
        name
    )
plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="gray",
    linewidth=2,
    label="Random Guess"
)

plt.grid(alpha=0.3)

plt.xlabel("False Positive Rate", fontsize=13)

plt.ylabel("True Positive Rate", fontsize=13)

plt.title(
    "ROC Curve Comparison",
    fontsize=18,
    weight="bold"
)

plt.legend(
    loc="lower right",
    fontsize=11
)
plt.show()
save_plot(
    "roc_curve_comparison.png",
    show=False
)
print("ROC Curve Generated Successfully!")