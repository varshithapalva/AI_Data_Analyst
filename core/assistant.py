"""Local dataset-aware question answering for LUMENIQ."""
import re
import numpy as np
import pandas as pd


def _normalise(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find_column(df, question):
    q = _normalise(question)
    # Prefer the longest exact column-name match to avoid partial-name confusion.
    matches = [c for c in df.columns if _normalise(c) and re.search(r"\b" + re.escape(_normalise(c)) + r"\b", q)]
    return max(matches, key=lambda c: len(_normalise(c))) if matches else None


def answer_dataset_question(df, question, cleaning_actions=None, model_result=None):
    """Answer common analytical questions using the currently loaded dataframe."""
    if not question or not str(question).strip():
        return "Please type a question about your dataset."
    q = _normalise(question)
    if q in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi lumeniq", "hello lumeniq"}:
        return "Hello! I'm LUMENIQ. Ask me about your data in everyday language—for example, average sales, best-selling products, monthly trends, missing values or unusual records."
    if any(x in q for x in ["thank you", "thanks", "thx"]):
        return "You're welcome! You can ask another question whenever you're ready."
    if q in {"bye", "goodbye", "see you"}:
        return "Goodbye! Your dataset stays available in this session."
    if any(x in q for x in ["what can you do", "help", "what can i ask", "how to use"]):
        return ("You can ask questions naturally, such as:\n"
                "• What is the average Sales / mean revenue?\n• Which product has the highest sales?\n"
                "• Show the top 5 customers by profit.\n• How many orders are there?\n"
                "• Which columns have missing values?\n• Compare sales by region.\n"
                "• What changed after cleaning?\n• Explain my latest model results.\n\n"
                "I calculate answers from the loaded table locally. This assistant supports dataset questions and common analytical intents, not unrestricted general conversation.")
    if df is None or df.empty:
        return "Please upload a dataset first so I can analyse it."

    rows, cols = df.shape
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    col = _find_column(df, question)

    if any(x in q for x in ["overview", "dataset summary", "describe dataset", "dataset shape", "how big", "dataset information"]):
        return f"Your dataset has {rows:,} rows and {cols} columns, with {int(df.isna().sum().sum()):,} missing cells and {int(df.duplicated().sum()):,} duplicate rows."
    if re.search(r"\b(rows|records|observations|entries|orders)\b", q) and any(x in q for x in ["how many", "count", "number", "total"]):
        return f"There are {rows:,} records in the current dataset."
    if "how many columns" in q or "number of columns" in q:
        return f"There are {cols} columns: {', '.join(map(str, df.columns))}."
    if any(x in q for x in ["column names", "list columns", "what columns", "available columns"]):
        return "The dataset columns are: " + ", ".join(map(str, df.columns)) + "."
    if any(x in q for x in ["missing", "null", "empty", "incomplete"]):
        missing = df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
        if missing.empty:
            return "There are no missing values in the current dataset."
        return "Missing values by column:\n" + "\n".join(f"• {c}: {int(n):,}" for c, n in missing.items())
    if any(x in q for x in ["duplicate", "repeated row"]):
        return f"The current dataset contains {int(df.duplicated().sum()):,} duplicate rows."
    if any(x in q for x in ["data type", "column type", "types of column"]):
        return "Column types:\n" + "\n".join(f"• {c}: {t}" for c, t in df.dtypes.items())
    if any(x in q for x in ["cleaning", "cleaned", "what changed", "cleaning action"]):
        return "No cleaning actions have been recorded yet." if not cleaning_actions else "Cleaning actions:\n" + "\n".join(f"• {a}" for a in cleaning_actions)
    if any(x in q for x in ["correlation", "correlated", "relationship between"]):
        if len(numeric) < 2:
            return "At least two numerical columns are needed to calculate correlations."
        from core.visualization import is_identifier_column
        useful_numeric = [c for c in numeric if not is_identifier_column(df[c], c)]
        if len(useful_numeric) < 2:
            return "At least two usable numerical columns are needed; identifier columns are excluded."
        corr = df[useful_numeric].corr(numeric_only=True)
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(useful_numeric) for b in useful_numeric[i+1:] if pd.notna(corr.loc[a,b])]
        if not pairs:
            return "There are not enough valid numeric values to calculate correlations."
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return "Strongest numerical relationships:\n" + "\n".join(f"• {a} and {b}: {v:.3f}" for a,b,v in pairs[:5]) + "\nCorrelation indicates association, not causation."

    # Numeric-column calculations, accepting everyday synonyms.
    if col is not None and col in numeric:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return f"{col} has no usable numeric values."
        if any(x in q for x in ["average", "mean", "avg", "typical"]): return f"The average {col} is {s.mean():,.2f}."
        if any(x in q for x in ["median", "middle value"]): return f"The median {col} is {s.median():,.2f}."
        if any(x in q for x in ["maximum", "highest", "largest", "most", "max"]): return f"The highest {col} value is {s.max():,.2f}."
        if any(x in q for x in ["minimum", "lowest", "smallest", "least", "min"]): return f"The lowest {col} value is {s.min():,.2f}."
        if any(x in q for x in ["total", "sum", "overall"]): return f"The total {col} is {s.sum():,.2f}."
        if any(x in q for x in ["standard deviation", "variation", "spread"]): return f"The standard deviation of {col} is {s.std():,.2f}."
        if any(x in q for x in ["distribution", "describe", "statistics", "stats"]):
            d=s.describe(); return f"Summary for {col}: count {d['count']:.0f}, mean {d['mean']:.2f}, median {s.median():.2f}, minimum {d['min']:.2f}, maximum {d['max']:.2f}, standard deviation {d['std']:.2f}."

    # Category questions: highest/lowest numeric value by a named category column.
    if col is not None and col in categorical and numeric and any(x in q for x in ["popular", "common", "frequent", "highest", "top", "best", "most", "lowest", "least"]):
        metric = next((c for c in numeric if _normalise(c) in q), numeric[0])
        grouped = df.groupby(col, dropna=False)[metric].sum().sort_values(ascending=not any(x in q for x in ["lowest", "least", "worst"]))
        n = 5 if any(x in q for x in ["top 5", "five", "top five"]) else 3
        return f"{('Top' if grouped.index.size else 'Results')} {col} categories by total {metric}:\n" + "\n".join(f"• {k}: {v:,.2f}" for k,v in grouped.head(n).items())
    if col is not None and col in categorical and any(x in q for x in ["count", "how many", "frequency", "distribution", "breakdown"]):
        counts=df[col].value_counts(dropna=False).head(12)
        return f"Record breakdown for {col}:\n" + "\n".join(f"• {k}: {v:,}" for k,v in counts.items())

    if any(x in q for x in ["model", "prediction", "accuracy", "r2", "silhouette", "machine learning"]):
        if not model_result: return "No model result is available yet. Open Predict and run a model first."
        return "Latest model results:\n" + "\n".join(f"• {k.replace('_',' ').title()}: {v}" for k,v in model_result.items() if np.isscalar(v) and k not in {"actual_values", "predicted_values"})

    # Generic fallback offers schema-aware prompts rather than a false answer.
    return ("I couldn't confidently identify the analysis you mean. I can calculate summaries and comparisons from this dataset. "
            f"Available numerical columns: {', '.join(map(str,numeric)) or 'none'}. "
            f"Available category columns: {', '.join(map(str,categorical)) or 'none'}. "
            "Try: 'What is the average [numeric column]?', 'Which [category] has the highest [numeric column]?', or 'Show the distribution of [column]'.")
