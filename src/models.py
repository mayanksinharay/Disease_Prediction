from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_models():

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest" : RandomForestClassifier(
    	n_estimators=200,
    	random_state=42,
    	n_jobs=-1
	),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5
            ))
        ])

    }

    return models