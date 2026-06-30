from sklearn.model_selection import GridSearchCV

from preprocessing import prepare_data
from models import get_models
from utils import save_model


# -----------------------------------------
# Load Data
# -----------------------------------------

X_train, X_test, y_train, y_test, _ = prepare_data(scale=False)

models = get_models()

# -----------------------------------------
# Parameter Grids
# -----------------------------------------

parameter_grids = {

    "Random Forest": {

        "n_estimators": [100, 200, 300],

        "max_depth": [None, 5, 10, 20],

        "min_samples_split": [2, 5, 10]

    },

    "SVM": {

        "classifier__C": [0.1, 1, 10],

        "classifier__kernel": ["linear", "rbf"],

        "classifier__gamma": ["scale", "auto"]

    }

}

# -----------------------------------------
# Perform Grid Search
# -----------------------------------------

for model_name, params in parameter_grids.items():

    print("\n" + "=" * 60)
    print(f"Grid Search : {model_name}")
    print("=" * 60)

    model = models[model_name]

    grid = GridSearchCV(

        estimator=model,

        param_grid=params,

        cv=5,

        scoring="accuracy",

        n_jobs=-1,

        verbose=2

    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters")

    print(grid.best_params_)

    print()

    print(f"Best CV Accuracy : {grid.best_score_:.4f}")

    save_model(
        grid.best_estimator_,
        f"best_{model_name.lower().replace(' ','_')}.pkl"
    )

print("\nGrid Search Complete!")