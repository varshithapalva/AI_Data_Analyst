
import pandas as pd
import numpy as np



def get_column_types(df):
    """Identify numerical, categorical, date and other columns."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # Detect date columns stored as text.
    for col in df.columns:
        if (
            col not in numeric_columns
            and col not in datetime_columns
            and (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            )
        ):
            parsed_dates = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed"
            )

            valid_ratio = parsed_dates.notna().mean()

            if valid_ratio >= 0.80:
                datetime_columns.append(col)

    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns
        and col not in datetime_columns
        and (
            pd.api.types.is_object_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[col])
        )
    ]

    boolean_columns = df.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    other_columns = [
        col for col in df.columns
        if col not in numeric_columns
        and col not in datetime_columns
        and col not in categorical_columns
        and col not in boolean_columns
    ]

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "datetime": datetime_columns,
        "boolean": boolean_columns,
        "other": other_columns
    }

def get_data_quality(df):
    """Calculate data quality information."""

    total_cells = df.shape[0] * df.shape[1]

    missing_values = df.isnull().sum()
    missing_count = int(missing_values.sum())

    duplicate_count = int(df.duplicated().sum())

    if total_cells > 0:
        completeness = (
            (total_cells - missing_count) / total_cells
        ) * 100
    else:
        completeness = 0

    quality = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "total_cells": int(total_cells),
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "completeness": round(completeness, 2),
        "memory_usage_kb": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        )
    }

    return quality


def get_column_profile(df):
    """Generate a profile for every column."""

    profiles = []

    for column in df.columns:
        series = df[column]

        profiles.append({
            "Column": column,
            "Data Type": str(series.dtype),
            "Non-Null Values": int(series.count()),
            "Missing Values": int(series.isnull().sum()),
            "Unique Values": int(series.nunique()),
            "Missing Percentage": round(
                series.isnull().mean() * 100, 2
            )
        })

    return pd.DataFrame(profiles)


def generate_profile(df):
    """Generate the complete dataset profile."""

    if df is None or df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    return {
        "quality": get_data_quality(df),
        "column_types": get_column_types(df),
        "columns": get_column_profile(df),
        "preview": df.head(10)
    }


def get_numeric_summary(df):
    """Generate descriptive statistics for numerical columns."""

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T.round(2)


def get_categorical_summary(df):
    """Generate summary information for categorical columns."""

    categorical_columns = get_column_types(df)["categorical"]

    results = []

    for column in categorical_columns:
        results.append({
            "Column": column,
            "Unique Categories": int(df[column].nunique()),
            "Most Common Value": (
                str(df[column].mode().iloc[0])
                if not df[column].mode().empty
                else "Not available"
            )
        })

    return pd.DataFrame(results)
