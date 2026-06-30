import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------
# Create Required Directories
# -----------------------------------------

def create_directories():
    folders = [
        "models",
        "reports",
        "reports/plots",
        "reports/metrics"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# -----------------------------------------
# Save Plot
# -----------------------------------------

def save_plot(filename, show=True):
    """
    Save the current matplotlib figure.
    """

    plt.tight_layout()

    plt.savefig(
        f"reports/plots/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[INFO] Saved plot -> reports/plots/{filename}")

    if show:
        plt.show()

    plt.close()


# -----------------------------------------
# Save Model
# -----------------------------------------

def save_model(model, filename):

    joblib.dump(model, f"models/{filename}")

    print(f"[INFO] Saved model -> models/{filename}")


# -----------------------------------------
# Save Metrics
# -----------------------------------------

def save_metrics(model_name, metrics_text):

    filename = f"reports/metrics/{model_name}_metrics.txt"

    with open(filename, "w") as file:
        file.write(metrics_text)

    print(f"[INFO] Saved metrics -> {filename}")


# -----------------------------------------
# Save Results CSV
# -----------------------------------------

def save_results_csv(result_dict):

    csv_path = "reports/metrics/model_results.csv"

    df = pd.DataFrame([result_dict])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"[INFO] Updated -> {csv_path}")


# -----------------------------------------
# Load Results
# -----------------------------------------

def load_results():

    return pd.read_csv("reports/metrics/model_results.csv")


# -----------------------------------------
# Reset Results CSV
# -----------------------------------------

def reset_results_csv():

    csv_path = "reports/metrics/model_results.csv"

    if os.path.exists(csv_path):
        os.remove(csv_path)
        print("[INFO] Previous model_results.csv removed.")