import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import create_directories, save_plot

create_directories()

# Load Dataset
df = pd.read_csv("dataset/heart.csv")

print("=" * 50)
print("Dataset Shape:", df.shape)
print("=" * 50)

print("\nHeart Disease Count\n")
print(df["target"].value_counts())

# ---------------------------------------
# Class Distribution
# ---------------------------------------

plt.figure(figsize=(6,4))

df["target"].value_counts().plot(kind="bar")

plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")

save_plot("class_distribution.png")

# ---------------------------------------
# Age Distribution
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.hist(df["age"], bins=15)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

save_plot("age_distribution.png")

# ---------------------------------------
# Gender Distribution
# ---------------------------------------

plt.figure(figsize=(6,4))

df["sex"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("Sex")
plt.ylabel("Count")

plt.xticks([0,1], ["Female","Male"])

save_plot("gender_distribution.png")

# ---------------------------------------
# Cholesterol Distribution
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.hist(df["chol"], bins=20)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Patients")

save_plot("cholesterol_distribution.png")

# ---------------------------------------
# Correlation Heatmap
# ---------------------------------------

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

save_plot("correlation_heatmap.png")