
import pandas as pd
import numpy as np

from core.profiling import get_column_types


def clean_dataset(df):
    """
    Automatically clean a dataset.

    Returns:
        cleaned_df: cleaned pandas DataFrame
        actions: list of cleaning operations performed
    """

    if df is None or df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned_df = df.copy()
    actions = []

    # 1. Remove duplicate records.
    duplicate_count = int(cleaned_df.duplicated().sum())

    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        actions.append(
            f"Removed {duplicate_count} duplicate rows."
        )
    else:
        actions.append("No duplicate rows found.")

    # 2. Identify column types.
    column_types = get_column_types(cleaned_df)

    # 3. Convert detected date columns.
    for column in column_types["datetime"]:
        if not pd.api.types.is_datetime64_any_dtype(
            cleaned_df[column]
        ):
            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column],
                errors="coerce",
                format="mixed"
            )

            actions.append(
                f"Converted '{column}' into datetime format."
            )

    # 4. Fill numerical missing values using median.
    for column in column_types["numeric"]:
        missing_count = int(cleaned_df[column].isnull().sum())

        if missing_count > 0:
            median_value = cleaned_df[column].median()

            if pd.notna(median_value):
                cleaned_df[column] = (
                    cleaned_df[column].fillna(median_value)
                )

                actions.append(
                    f"Filled {missing_count} missing values "
                    f"in '{column}' using median."
                )

    # 5. Fill categorical missing values using mode.
    for column in column_types["categorical"]:
        missing_count = int(cleaned_df[column].isnull().sum())

        if missing_count > 0:
            modes = cleaned_df[column].mode(dropna=True)

            if not modes.empty:
                fill_value = modes.iloc[0]
                cleaned_df[column] = (
                    cleaned_df[column].fillna(fill_value)
                )

                actions.append(
                    f"Filled {missing_count} missing values "
                    f"in '{column}' using mode."
                )
            else:
                cleaned_df[column] = (
                    cleaned_df[column].fillna("Unknown")
                )

                actions.append(
                    f"Filled missing values in '{column}' "
                    "with 'Unknown'."
                )

    # 6. Fill missing boolean values.
    for column in column_types["boolean"]:
        missing_count = int(cleaned_df[column].isnull().sum())

        if missing_count > 0:
            cleaned_df[column] = (
                cleaned_df[column].fillna(False)
            )

            actions.append(
                f"Filled {missing_count} missing values "
                f"in '{column}' with False."
            )

    # 7. Clean extra spaces in text columns.
    for column in cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns:

        cleaned_df[column] = cleaned_df[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    actions.append("Removed unnecessary spaces from text values.")

    return cleaned_df, actions


def compare_quality(original_df, cleaned_df):
    """Compare dataset quality before and after cleaning."""

    original_missing = int(original_df.isnull().sum().sum())
    cleaned_missing = int(cleaned_df.isnull().sum().sum())

    original_duplicates = int(original_df.duplicated().sum())
    cleaned_duplicates = int(cleaned_df.duplicated().sum())

    return {
        "original_rows": int(len(original_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "original_missing": original_missing,
        "cleaned_missing": cleaned_missing,
        "original_duplicates": original_duplicates,
        "cleaned_duplicates": cleaned_duplicates,
        "rows_removed": int(len(original_df) - len(cleaned_df)),
        "missing_values_filled": original_missing - cleaned_missing
    }
