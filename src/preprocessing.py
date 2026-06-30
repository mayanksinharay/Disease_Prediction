import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(scale=True):
    """
    Load the dataset, remove duplicates, split into training/testing
    sets and optionally scale the features.
    """

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------
    df = pd.read_csv("dataset/heart.csv")

    # --------------------------------------------------
    # Remove Duplicate Rows
    # --------------------------------------------------
    df = df.drop_duplicates()

    # --------------------------------------------------
    # Separate Features and Target
    # --------------------------------------------------
    X = df.drop("target", axis=1)
    y = df["target"]

    # --------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = None

    # --------------------------------------------------
    # Feature Scaling (Optional)
    # --------------------------------------------------
    if scale:

        scaler = StandardScaler()

        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    # --------------------------------------------------
    # Return Processed Data
    # --------------------------------------------------
    return X_train, X_test, y_train, y_test, scaler


# --------------------------------------------------
# Testing
# --------------------------------------------------
if __name__ == "__main__":

    X_train, X_test, y_train, y_test, scaler = prepare_data()

    print("=" * 60)
    print("Training Features :", X_train.shape)
    print("Testing Features  :", X_test.shape)
    print("Training Labels   :", y_train.shape)
    print("Testing Labels    :", y_test.shape)
    print("=" * 60)

    print("\nFirst Five Training Samples:\n")
    print(X_train.head())