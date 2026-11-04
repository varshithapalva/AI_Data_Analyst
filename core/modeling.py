
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)


def prepare_features(df, target=None):
    """
    Prepare input features:
    - Exclude identifier columns.
    - Convert dates into year, month and day.
    - Separate numerical and categorical columns.
    - Exclude the prediction target.
    """

    data = df.copy()

    if target is not None:
        if target not in data.columns:
            raise ValueError("Selected target column does not exist.")

        data = data.drop(columns=[target])

    # Convert datetime columns into useful numerical features.
    date_columns = data.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in date_columns:
        data[column + "_Year"] = data[column].dt.year
        data[column + "_Month"] = data[column].dt.month
        data[column + "_Day"] = data[column].dt.day
        data = data.drop(columns=[column])

    # Exclude likely identifiers.
    identifier_words = {
        "id", "index", "identifier", "code"
    }

    columns_to_drop = []

    for column in data.columns:
        parts = column.lower().replace("-", "_").split("_")

        if any(part in identifier_words for part in parts):
            columns_to_drop.append(column)

    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Exclude columns with no useful information.
    useless_columns = [
        column for column in data.columns
        if data[column].nunique(dropna=True) <= 1
    ]

    data = data.drop(columns=useless_columns, errors="ignore")

    # Convert unsupported infinite values to missing values.
    data = data.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely missing.
    data = data.dropna(axis=1, how="all")

    return data


def build_preprocessor(X, scale_numeric=False):
    """Create preprocessing pipelines for numerical and categorical data."""

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable input features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def run_classification(df, target):
    """Train and evaluate a Random Forest classification model."""

    if target not in df.columns:
        raise ValueError("Target column was not found.")

    working_df = df.dropna(subset=[target]).copy()

    if len(working_df) < 20:
        raise ValueError(
            "At least 20 complete records are recommended for classification."
        )

    y = working_df[target]
    X = prepare_features(working_df, target)

    if X.empty:
        raise ValueError("No usable feature columns were found.")

    if y.nunique() < 2:
        raise ValueError("Classification requires at least two classes.")

    # Remove classes with only one example to allow stratified splitting.
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    working_mask = y.isin(valid_classes)
    X = X.loc[working_mask]
    y = y.loc[working_mask]

    if y.nunique() < 2:
        raise ValueError(
            "Not enough examples per category to train a model."
        )

    # Use stratification when each class has at least two records.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.named_steps["model"].classes_
    )

    return {
        "task": "Classification",
        "algorithm": "Random Forest Classifier",
        "target": target,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": matrix.tolist(),
        "classes": [
            str(value)
            for value in model.named_steps["model"].classes_
        ],
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": X.columns.tolist(),
        "model": model
    }


def run_regression(df, target):
    """Train and evaluate a Random Forest regression model."""

    if target not in df.columns:
        raise ValueError("Target column was not found.")

    working_df = df.copy()

    working_df[target] = pd.to_numeric(
        working_df[target],
        errors="coerce"
    )

    working_df = working_df.dropna(subset=[target])

    if len(working_df) < 20:
        raise ValueError(
            "At least 20 complete records are recommended for regression."
        )

    y = working_df[target]
    X = prepare_features(working_df, target)

    if X.empty:
        raise ValueError("No usable feature columns were found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "task": "Regression",
        "algorithm": "Random Forest Regressor",
        "target": target,
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2_score": round(float(r2), 4),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": X.columns.tolist(),
        "actual_values": y_test.tolist(),
        "predicted_values": predictions.tolist(),
        "model": model
    }


def run_clustering(df, number_of_clusters=3):
    """Perform K-Means clustering on the dataset."""

    X = prepare_features(df)

    if len(X) < 10:
        raise ValueError(
            "At least 10 records are recommended for clustering."
        )

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns were found.")

    if not 2 <= number_of_clusters < len(X):
        raise ValueError(
            "Number of clusters must be at least 2 "
            "and less than the number of records."
        )

    preprocessor = build_preprocessor(
        X,
        scale_numeric=True
    )

    transformed = preprocessor.fit_transform(X)

    if transformed.shape[1] == 0:
        raise ValueError("No usable features remain after preprocessing.")

    # Keep clustering practical for larger datasets.
    if len(X) > 10000:
        raise ValueError(
            "This initial clustering version supports up to 10,000 rows."
        )

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(transformed)

    inertia = model.inertia_

    if len(set(labels)) > 1 and len(X) > number_of_clusters:
        silhouette = silhouette_score(
            transformed,
            labels,
            sample_size=min(2000, len(X)),
            random_state=42
        )
    else:
        silhouette = None

    result_df = df.copy()
    result_df["Cluster"] = labels

    return {
        "task": "Clustering",
        "algorithm": "K-Means",
        "number_of_clusters": number_of_clusters,
        "inertia": round(float(inertia), 4),
        "silhouette_score": (
            round(float(silhouette), 4)
            if silhouette is not None
            else None
        ),
        "cluster_counts": {
            str(key): int(value)
            for key, value in
            pd.Series(labels).value_counts().sort_index().items()
        },
        "clustered_data": result_df,
        "model": model,
        "preprocessor": preprocessor
    }
