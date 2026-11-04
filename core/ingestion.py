
import io
import pandas as pd


SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def load_dataset(uploaded_file):
    """
    Load a CSV or Excel dataset from a Streamlit uploaded file.

    Returns:
        pandas.DataFrame
    """

    if uploaded_file is None:
        raise ValueError("Please upload a dataset.")

    filename = uploaded_file.name.lower()

    if not filename.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Excel file."
        )

    file_bytes = uploaded_file.getvalue()

    try:
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
            except UnicodeDecodeError:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding="latin-1"
                )

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

        elif filename.endswith(".xls"):
            try:
                df = pd.read_excel(io.BytesIO(file_bytes))
            except ImportError as exc:
                raise ValueError(
                    "Reading .xls files requires the xlrd package. "
                    "Run: python -m pip install xlrd"
                ) from exc

        if df.empty:
            raise ValueError("The uploaded file contains no data.")

        if len(df.columns) == 0:
            raise ValueError("The uploaded file has no columns.")

        # Remove accidental spaces from column names.
        df.columns = [str(col).strip() for col in df.columns]

        return df

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            f"Unable to read the uploaded file: {exc}"
        ) from exc


def get_file_size(uploaded_file):
    """Return uploaded file size in KB."""

    return round(len(uploaded_file.getvalue()) / 1024, 2)
