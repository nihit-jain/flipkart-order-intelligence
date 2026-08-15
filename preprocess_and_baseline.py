import pandas as pd
import numpy as np


import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def main() -> None:
    # Load the dataset
    df = pd.read_csv("orders_dataset.csv")

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    # Separate features (X) from target (y)
    X = df.drop(columns=["returned", "order_id"])
    y = df["returned"]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

        # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining set:", X_train.shape)
    print("Test set:", X_test.shape)

        # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Preprocessing for numerical features
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

        # Preprocessing for categorical features
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

        # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Random Forest pipeline
    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
        # Hyperparameter grid
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 10, None],
    }

    # 5-fold GridSearchCV
    grid_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )

    print("\nTraining Random Forest GridSearchCV...")

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "Best cross-validated ROC-AUC:",
        round(grid_search.best_score_, 4),
    )
        # Evaluate best Random Forest on held-out test set
    y_prob_rf = grid_search.predict_proba(X_test)[:, 1]

    rf_test_roc_auc = roc_auc_score(
        y_test,
        y_prob_rf,
    )

    print(
        "Random Forest test ROC-AUC:",
        round(rf_test_roc_auc, 4),
    )

    # Feature importance from the winning Random Forest
    best_rf = grid_search.best_estimator_

    rf_classifier = best_rf.named_steps["classifier"]
    importances = rf_classifier.feature_importances_

    # Get feature names after preprocessing
    feature_names = (
        best_rf.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\nTop 5 Random Forest features:")
    print(
        importance_df.head(5).to_string(index=False)
    )

    # Baseline model: always predicts the majority class
    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )

    # Logistic Regression model
    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train Logistic Regression
    logistic_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_logistic = logistic_model.predict(X_test)
    y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob_logistic)
    print("ROC-AUC:", round(roc_auc, 4))
    # Sweep decision thresholds from 0.10 to 0.90
    thresholds = np.arange(0.10, 0.901, 0.01)

    threshold_results = []

    for threshold in thresholds:
        y_pred_threshold = (y_prob_logistic >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_pred_threshold,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred_threshold,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred_threshold,
            zero_division=0,
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    threshold_df = pd.DataFrame(threshold_results)

    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print("\nBest threshold by F1:")
    print("Threshold:", round(best_row["threshold"], 2))
    print("Precision:", round(best_row["precision"], 4))
    print("Recall:", round(best_row["recall"], 4))
    print("F1-score:", round(best_row["f1"], 4))

    # Evaluate Logistic Regression
    accuracy = accuracy_score(y_test, y_pred_logistic)
    precision = precision_score(
        y_test,
        y_pred_logistic,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred_logistic,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred_logistic,
        zero_division=0,
    )

    print("\nLogistic Regression:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_logistic)

    print("\nLogistic Regression confusion matrix:")
    print(cm)

    # Train the baseline
    baseline_model.fit(X_train, y_train)

    # Make predictions on the untouched test set
    y_pred = baseline_model.predict(X_test)

    # Evaluate the baseline
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nDummyClassifier baseline:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    # Permutation importance on the held-out test set
    permutation = permutation_importance(
        grid_search.best_estimator_,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    permutation_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": permutation.importances_mean,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\nTop 5 permutation importance features:")
    print(
        permutation_df.head(5).to_string(index=False)
    )

    # Subgroup analysis using the tuned Random Forest
    y_pred_rf = (
        grid_search.best_estimator_
        .predict(X_test)
    )

    test_results = X_test.copy()
    test_results["actual"] = y_test.values
    test_results["predicted"] = y_pred_rf

    print("\nSubgroup analysis: product_category")

    for group in test_results["product_category"].unique():
        subset = test_results[
            test_results["product_category"] == group
        ]

        precision = precision_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0,
        )

        recall = recall_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0,
        )

        print(
            f"{group}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}"
        )

    print("\nSubgroup analysis: payment_method")

    for group in test_results["payment_method"].unique():
        subset = test_results[
            test_results["payment_method"] == group
        ]

        precision = precision_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0,
        )

        recall = recall_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0,
        )

        print(
            f"{group}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}"
        )

    joblib.dump(
        grid_search.best_estimator_,
        "models/return_risk_model.pkl",
    )

    print(
        "\nSaved final model to "
        "models/return_risk_model.pkl"
    )

if __name__ == "__main__":
    main()