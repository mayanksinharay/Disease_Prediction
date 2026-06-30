import matplotlib.pyplot as plt

from utils import (
    create_directories,
    load_results,
    save_plot
)

create_directories()

df = load_results()

df = df.sort_values(by="Accuracy", ascending=False)

print(df)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for metric in metrics:

    plt.figure(figsize=(9,5))

    bars = plt.bar(
        df["Model"],
        df[metric]
    )

    plt.title(f"{metric} Comparison", fontsize=16)

    plt.xlabel("Models", fontsize=12)

    plt.ylabel(metric, fontsize=12)

    plt.ylim(0,1)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=10)

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.2f}",
            ha="center"
        )

    save_plot(
        f"{metric.lower().replace(' ','_')}_comparison.png",
        show=False
    )

print("\nAll comparison plots generated successfully!")