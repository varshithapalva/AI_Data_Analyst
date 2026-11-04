import re

import numpy as np
import pandas as pd
import plotly.express as px

from core.profiling import get_column_types


CHART_TYPES = [
    "Bar Chart",
    "Line Chart",
    "Histogram",
    "Scatter Plot",
    "Box Plot",
    "Pie Chart",
    "Correlation Heatmap",
]


CHART_GUIDE = {
    "Bar Chart": {
        "purpose": "Compare a numerical measure across categories.",
        "questions": [
            "Which product category has the highest sales?",
            "Which region contributes the most profit?",
        ],
        "interpretation": "Taller bars represent larger values. Compare categories to identify the strongest and weakest groups.",
    },
    "Line Chart": {
        "purpose": "Show how a numerical measure changes across an ordered timeline or sequence.",
        "questions": [
            "Are sales increasing or decreasing over time?",
            "When did the business experience a peak or dip?",
        ],
        "interpretation": "The direction and shape of the line reveal trends, rises, falls and possible seasonal patterns.",
    },
    "Histogram": {
        "purpose": "Show how frequently numerical values fall within value ranges.",
        "questions": [
            "What sales range occurs most often?",
            "Are values concentrated or widely spread?",
        ],
        "interpretation": "The tallest bins show the most common ranges. A long tail or multiple peaks may indicate skew or different groups.",
    },
    "Scatter Plot": {
        "purpose": "Explore the relationship between two numerical variables.",
        "questions": [
            "Do higher discounts tend to occur with higher sales?",
            "Is there a visible relationship between profit and sales?",
        ],
        "interpretation": "An upward or downward pattern suggests association. Isolated points may be unusual observations; association does not prove causation.",
    },
    "Box Plot": {
        "purpose": "Compare the median, spread and possible outliers of numerical values.",
        "questions": [
            "Which group has the widest variation in sales?",
            "Are there unusually high or low values?",
        ],
        "interpretation": "The centre line is the median, the box represents the middle half of values, and separate points may be outliers.",
    },
    "Pie Chart": {
        "purpose": "Show how a small number of categories contribute to a whole.",
        "questions": [
            "What share of records belongs to each customer segment?",
            "Which category represents the largest proportion?",
        ],
        "interpretation": "Each slice represents a category's share of the total. Pie charts are clearest when there are only a few categories.",
    },
    "Correlation Heatmap": {
        "purpose": "Summarize the direction and strength of pairwise relationships among numerical columns.",
        "questions": [
            "Which numerical columns move together?",
            "Are any pairs strongly positively or negatively correlated?",
        ],
        "interpretation": "Values close to +1 indicate positive association, values near -1 indicate negative association, and values near 0 indicate little linear association.",
    },
}


_ID_NAME_PATTERN = re.compile(
    r"(^|[_\s-])(id|identifier|index|serial|record_no|row_no|\w*code)([_\s-]|$)|"
    r"(^|[_\s-])(order|customer|product|transaction|invoice)_?(id|number|no)([_\s-]|$)",
    re.IGNORECASE,
)


def is_identifier_column(series, column_name):
    """Return True for columns that look like record keys, not analysis measures."""
    name = str(column_name).strip()
    if _ID_NAME_PATTERN.search(name):
        return True
    # Date/time columns are often unique per row but are meaningful for trends.
    if re.search(r"date|time|timestamp", name, re.IGNORECASE):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) < 5:
        return False

    unique_ratio = non_null.nunique() / len(non_null)
    if not pd.api.types.is_numeric_dtype(series) and unique_ratio >= 0.98:
        # Unique strings are often transaction references or names, not useful groups.
        return True

    if pd.api.types.is_numeric_dtype(series) and unique_ratio >= 0.90:
        # Numeric sequences with nearly one value per row usually behave as row IDs.
        values = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(values) > 1:
            differences = values.sort_values().diff().dropna()
            if not differences.empty and (differences == 1).mean() >= 0.80:
                return True
    return False


def get_usable_columns(df):
    """Return numeric, categorical and datetime columns suitable for charts."""
    if df is None or df.empty:
        return {"numeric": [], "categorical": [], "datetime": []}

    types = get_column_types(df)
    numeric = [
        col for col in types["numeric"]
        if not is_identifier_column(df[col], col)
    ]
    categorical = [
        col for col in types["categorical"]
        if not is_identifier_column(df[col], col)
        and 1 < df[col].nunique(dropna=True) <= 12
    ]
    datetime = [
        col for col in types["datetime"]
        if not is_identifier_column(df[col], col)
    ]
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}


def get_visualization_suggestions(df):
    """Recommend charts using analytical columns rather than likely identifiers."""
    if df is None or df.empty:
        return []

    usable = get_usable_columns(df)
    numeric = usable["numeric"]
    categorical = usable["categorical"]
    datetime = usable["datetime"]
    suggestions = []

    if numeric:
        suggestions.append({
            "type": "Histogram", "x": numeric[0], "y": None,
            "reason": f"See how {numeric[0]} values are distributed.",
        })
        suggestions.append({
            "type": "Box Plot", "x": numeric[0], "y": None,
            "reason": f"Inspect the median, spread and possible outliers in {numeric[0]}.",
        })

    if categorical and numeric:
        suggestions.append({
            "type": "Bar Chart", "x": categorical[0], "y": numeric[0],
            "reason": f"Compare average {numeric[0]} across {categorical[0]} categories.",
        })

    if datetime and numeric:
        suggestions.append({
            "type": "Line Chart", "x": datetime[0], "y": numeric[0],
            "reason": f"Track how {numeric[0]} changes over {datetime[0]}.",
        })

    if len(numeric) >= 2:
        suggestions.append({
            "type": "Scatter Plot", "x": numeric[0], "y": numeric[1],
            "reason": f"Explore whether {numeric[0]} and {numeric[1]} are related.",
        })
        suggestions.append({
            "type": "Correlation Heatmap", "x": None, "y": None,
            "reason": "Review relationships across all usable numerical columns.",
        })

    if categorical:
        suggestions.append({
            "type": "Pie Chart", "x": categorical[0], "y": None,
            "reason": f"View the proportion of records in each {categorical[0]} category.",
        })

    return suggestions


def create_chart(df, chart_type, x=None, y=None, color=None, theme="light"):
    """Create an interactive Plotly chart with validation and theme support."""
    if df is None or df.empty:
        raise ValueError("No dataset available for visualization.")
    if chart_type not in CHART_TYPES:
        raise ValueError("Unsupported chart type. Please choose an individual chart.")

    chart_df = df.copy()
    template = "plotly_dark" if str(theme).lower() == "dark" else "plotly_white"

    if chart_type == "Correlation Heatmap":
        usable_numeric = get_usable_columns(chart_df)["numeric"]
        numeric_df = chart_df[usable_numeric].select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            raise ValueError("At least two usable numerical columns are required.")
        fig = px.imshow(
            numeric_df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Correlation Heatmap", aspect="auto",
        )
    else:
        if x not in chart_df.columns:
            raise ValueError("Please select a valid X-axis column.")
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"] and y not in chart_df.columns:
            raise ValueError("Please select a valid Y-axis column.")
        if color not in chart_df.columns:
            color = None

        if chart_type == "Bar Chart":
            if is_identifier_column(chart_df[x], x):
                raise ValueError("Choose a meaningful category column, not an ID or record key.")
            if not pd.api.types.is_numeric_dtype(chart_df[y]):
                raise ValueError("Bar chart Y-axis must be numerical.")
            grouped = chart_df.groupby(x, dropna=False)[y].mean().reset_index()
            fig = px.bar(grouped, x=x, y=y, title=f"Average {y} by {x}", text_auto=".2s")
        elif chart_type == "Line Chart":
            if not (pd.api.types.is_datetime64_any_dtype(chart_df[x]) or pd.api.types.is_numeric_dtype(chart_df[x])):
                parsed_x = pd.to_datetime(chart_df[x], errors="coerce", format="mixed")
                if parsed_x.notna().mean() >= 0.80:
                    chart_df[x] = parsed_x
                else:
                    raise ValueError("Choose a date or ordered numerical column for the line chart X-axis.")
            if not pd.api.types.is_numeric_dtype(chart_df[y]):
                raise ValueError("Line chart Y-axis must be numerical.")
            grouped = chart_df.groupby(x, dropna=False)[y].mean().reset_index().sort_values(by=x)
            fig = px.line(grouped, x=x, y=y, markers=True, title=f"Average {y} across {x}")
        elif chart_type == "Histogram":
            if not pd.api.types.is_numeric_dtype(chart_df[x]):
                raise ValueError("Histogram requires a numerical X-axis.")
            fig = px.histogram(chart_df, x=x, color=color, title=f"Distribution of {x}", marginal="box")
        elif chart_type == "Scatter Plot":
            if not pd.api.types.is_numeric_dtype(chart_df[x]) or not pd.api.types.is_numeric_dtype(chart_df[y]):
                raise ValueError("Scatter plot requires two numerical columns.")
            fig = px.scatter(chart_df, x=x, y=y, color=color, title=f"{y} vs {x}")
        elif chart_type == "Box Plot":
            if not pd.api.types.is_numeric_dtype(chart_df[x]):
                raise ValueError("Box plot requires a numerical column.")
            fig = px.box(chart_df, y=x, points="outliers", title=f"Distribution and Outliers of {x}")
        elif chart_type == "Pie Chart":
            if is_identifier_column(chart_df[x], x):
                raise ValueError("Choose a category, not an ID or record key, for a pie chart.")
            counts = chart_df[x].fillna("Missing").value_counts().head(12).rename_axis(x).reset_index(name="Count")
            fig = px.pie(counts, names=x, values="Count", hole=0.35, title=f"Distribution of {x}")
        else:
            raise ValueError("Unable to create this chart.")

    fig.update_layout(
        template=template, height=500,
        margin=dict(l=30, r=30, t=70, b=40),
        legend_title_text="Category",
    )
    return fig


def get_chart_explanation(chart_type, x=None, y=None):
    """Return a user-friendly purpose, example questions and interpretation."""
    guide = CHART_GUIDE.get(chart_type)
    if not guide:
        return "This chart provides an overview of your dataset."
    details = [guide["purpose"]]
    if x:
        details.append(f"Selected X/category: {x}.")
    if y:
        details.append(f"Selected measure: {y}.")
    details.append("Questions this chart can help answer: " + "; ".join(guide["questions"]))
    details.append("How to read it: " + guide["interpretation"])
    return "\n\n".join(details)
