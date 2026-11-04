import streamlit as st

# Inject custom CSS for a beautiful purple-to-teal-to-blue gradient header block
st.markdown(
    """
    <style>
    .gradient-header {
        background: linear-gradient(135deg, #7B2CBF 0%, #06B6D4 50%, #2563EB 100%);
        padding: 24px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Example usage in your layout:
st.markdown('<div class="gradient-header"><h1>Welcome to LUMENIQ</h1></div>', unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import plotly.express as px

from core.ingestion import load_dataset, get_file_size
from core.profiling import (
    generate_profile,
    get_numeric_summary,
    get_categorical_summary
)
from core.cleaning import clean_dataset, compare_quality
from core.visualization import (
    CHART_TYPES, CHART_GUIDE, create_chart, get_visualization_suggestions,
    get_chart_explanation, get_usable_columns, is_identifier_column
)
from core.modeling import (
    run_classification,
    run_regression,
    run_clustering
)
from core.assistant import answer_dataset_question
from core.reporting import generate_pdf_report


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LUMENIQ | Intelligent Data Analyst",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* ================================================
   LUMENIQ - COLORFUL PROFESSIONAL THEME
   ================================================ */

/* MAIN BACKGROUND */
.stApp {
    background:
        radial-gradient(
            ellipse at 5% 5%,
            rgba(139, 92, 246, 0.17),
            transparent 35%
        ),
        radial-gradient(
            ellipse at 95% 10%,
            rgba(6, 182, 212, 0.14),
            transparent 32%
        ),
        radial-gradient(
            ellipse at 50% 100%,
            rgba(236, 72, 153, 0.09),
            transparent 35%
        ),
        var(--background-color);

    color: var(--text-color);
}

/* MAIN CONTENT SPACING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HEADER */
.main-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1.5px;

    background: linear-gradient(
        90deg,
        #7c3aed,
        #4f46e5,
        #0891b2,
        #0d9488
    );

    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0;
}

.main-subtitle {
    font-size: 1.05rem;
    opacity: 0.82;
    margin-top: 5px;
    margin-bottom: 28px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg,
        rgba(91, 33, 182, 0.18),
        rgba(79, 70, 229, 0.13),
        rgba(8, 145, 178, 0.13)
    );

    border-right: 1px solid rgba(139, 92, 246, 0.25);
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

/* SIDEBAR HEADINGS */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-color);
    font-weight: 750;
}

/* SECTION HEADINGS */
h1, h2, h3 {
    letter-spacing: -0.4px;
}

h2 {
    font-weight: 800;
}

h3 {
    font-weight: 700;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(124, 58, 237, 0.15),
        rgba(79, 70, 229, 0.09),
        rgba(6, 182, 212, 0.10)
    );

    border: 1px solid rgba(139, 92, 246, 0.30);
    border-radius: 18px;

    padding: 19px 17px;

    box-shadow:
        0 5px 18px rgba(79, 70, 229, 0.08);

    transition: transform 0.2s ease,
                box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);

    box-shadow:
        0 9px 24px rgba(79, 70, 229, 0.16);
}

div[data-testid="stMetricLabel"] {
    font-weight: 650;
}

div[data-testid="stMetricValue"] {
    font-weight: 800;
}

/* TABS */
div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 11px 11px 0 0;
    font-weight: 700;
    padding: 10px 15px;
    transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[role="tab"]:hover {
    background: rgba(139, 92, 246, 0.13);
    color: #8b5cf6;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #8b5cf6;
    border-bottom: 3px solid #8b5cf6;
    background: rgba(139, 92, 246, 0.10);
}

/* BUTTONS */
.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: 0.55rem 1.1rem;

    border: 1px solid rgba(124, 58, 237, 0.35);

    background: linear-gradient(
        110deg,
        #7c3aed,
        #6366f1,
        #0891b2
    );

    color: white;

    box-shadow:
        0 4px 12px rgba(99, 102, 241, 0.20);

    transition: all 0.2s ease;
}

/* BUTTON HOVER */
.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #06b6d4;
    color: white;

    box-shadow:
        0 7px 20px rgba(99, 102, 241, 0.32);

    transform: translateY(-2px);
}

/* BUTTON ACTIVE */
.stButton > button:active,
.stDownloadButton > button:active {
    transform: scale(0.98);
}

/* PRIMARY BUTTON */
.stButton > button[kind="primary"] {
    background: linear-gradient(
        110deg,
        #7c3aed,
        #4f46e5,
        #0891b2
    );

    color: white;
    border: none;
}

/* INPUT FIELDS */
div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="textarea"] {
    border-radius: 11px;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 11px;
    border-color: rgba(139, 92, 246, 0.30);
}

/* INPUT FOCUS */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within,
div[data-baseweb="textarea"]:focus-within {
    border-color: #8b5cf6;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.16);
}

/* SELECTED OPTIONS */
div[data-baseweb="popover"] {
    border-radius: 12px;
}

/* INFORMATION BOXES */
div[data-testid="stAlert"] {
    border-radius: 14px;
    border-width: 1px;
}

/* DATAFRAMES */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 14px;
    overflow: hidden;

    box-shadow:
        0 4px 15px rgba(79, 70, 229, 0.06);
}

/* CHAT MESSAGES */
div[data-testid="stChatMessage"] {
    border: 1px solid rgba(139, 92, 246, 0.22);
    border-radius: 16px;
    padding: 12px 15px;
    margin-bottom: 10px;

    background: rgba(139, 92, 246, 0.055);
}

/* CHAT INPUT */
div[data-testid="stChatInput"] {
    border-radius: 15px;
}

/* CHART CONTAINERS */
div[data-testid="stPlotlyChart"] {
    background: rgba(139, 92, 246, 0.035);
    border: 1px solid rgba(139, 92, 246, 0.17);
    border-radius: 15px;
    padding: 8px;
}

/* EXPANDERS */
div[data-testid="stExpander"] {
    border: 1px solid rgba(139, 92, 246, 0.24);
    border-radius: 13px;
    overflow: hidden;
}

/* DIVIDERS */
hr {
    border-color: rgba(139, 92, 246, 0.22);
}

/* CAPTIONS */
.stCaption {
    opacity: 0.78;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"] {
    background: rgba(139, 92, 246, 0.055);
    border: 1px dashed rgba(139, 92, 246, 0.50);
    border-radius: 15px;
    padding: 10px;
}

/* FOOTER */
footer {
    visibility: hidden;
}

/* SMOOTH TRANSITIONS */
.stApp,
section[data-testid="stSidebar"],
div[data-testid="stMetric"] {
    transition: background 0.25s ease,
                color 0.25s ease;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "original_df": None,
    "cleaned_df": None,
    "cleaning_actions": [],
    "dataset_name": None,
    "model_result": None,
    "last_question": "",
    "last_answer": "",
    "chat_history": [],
    "pending_question": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# File uploader reset key
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">LUMENIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Your intelligent workspace for data exploration, automated cleaning and predictive analytics'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR AND UPLOAD
# --------------------------------------------------

with st.sidebar:
    st.title("✨ LUMENIQ")
    st.caption("Understand your data. Discover what matters.")

    st.divider()

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX and XLS",
        key=f"dataset_uploader_{st.session_state.upload_key}"
    )

    if uploaded_file is not None:
        if st.session_state.dataset_name != uploaded_file.name:
            try:
                new_df = load_dataset(uploaded_file)

                st.session_state.original_df = new_df.copy()
                st.session_state.cleaned_df = new_df.copy()
                st.session_state.cleaning_actions = []
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.model_result = None
                st.session_state.last_question = ""
                st.session_state.last_answer = ""
                st.session_state.chat_history = []
                st.session_state.pending_question = None

                st.success("Dataset uploaded successfully!")

            except Exception as exc:
                st.error(str(exc))

    if st.session_state.dataset_name:
        st.divider()
        st.success(f"Loaded: {st.session_state.dataset_name}")

        if st.button("Remove Dataset", use_container_width=True):
            for key, value in defaults.items():
                st.session_state[key] = value

            # Reset the file uploader widget
            st.session_state.chat_history = []
            st.session_state.pending_question = None
            st.session_state.upload_key += 1

            st.rerun()

    st.divider()
    st.caption("LUMENIQ | Local Data Analysis")


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if st.session_state.cleaned_df is None:

    st.info("👈 Upload a CSV or Excel dataset from the sidebar to get started.")

    st.markdown("### What can LUMENIQ do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧹 Automated Cleaning")
        st.write(
            "Identify missing values, remove duplicates "
            "and prepare datasets for analysis."
        )

    with col2:
        st.subheader("📈 Data Exploration")
        st.write(
            "Explore data through interactive charts, "
            "summaries and visual insights."
        )

    with col3:
        st.subheader("🤖 Predictive Analytics")
        st.write(
            "Perform classification, regression "
            "and clustering using machine learning."
        )

    st.stop()


# --------------------------------------------------
# DATA REFERENCES
# --------------------------------------------------

original_df = st.session_state.original_df
df = st.session_state.cleaned_df
cleaning_actions = st.session_state.cleaning_actions


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

tabs = st.tabs([
    "🏠 Overview",
    "🧹 Prepare Data",
    "📊 Explore Data",
    "🤖 Predict",
    "💬 AI Assistant",
    "📄 Reports"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with tabs[0]:

    st.header("Dataset Overview")

    st.write(
        "Get a quick understanding of your uploaded dataset "
        "and its current data quality."
    )

    profile = generate_profile(df)
    quality = profile["quality"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Completeness", f"{quality['completeness']}%")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Information")

    st.dataframe(
        profile["columns"],
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Numeric Summary")
        numeric_summary = get_numeric_summary(df)

        if not numeric_summary.empty:
            st.dataframe(numeric_summary, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    with col2:
        st.subheader("Categorical Summary")
        categorical_summary = get_categorical_summary(df)

        if not categorical_summary.empty:
            st.dataframe(
                categorical_summary,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No categorical columns available.")


# ==================================================
# TAB 2: PREPARE DATA
# ==================================================

with tabs[1]:

    st.header("Automated Data Preparation")

    st.write(
        "LUMENIQ can automatically remove duplicate records, "
        "handle missing values and prepare columns."
    )

    st.subheader("Original Dataset Preview")
    st.caption("This is the dataset exactly as uploaded. LUMENIQ keeps this original copy so you can compare it with the cleaned version.")
    st.dataframe(original_df.head(20), use_container_width=True)

    raw_col1, raw_col2, raw_col3 = st.columns(3)

    raw_col1.metric("Original Rows", f"{len(original_df):,}")
    raw_col2.metric("Original Missing Values", f"{int(original_df.isnull().sum().sum()):,}")
    raw_col3.metric("Original Duplicate Rows", f"{int(original_df.duplicated().sum()):,}")

    if st.button("✨ Clean My Dataset", type="primary"):

        try:
            cleaned, actions = clean_dataset(original_df)

            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_actions = actions
            st.session_state.model_result = None

            st.success("Dataset cleaning completed!")
            st.rerun()

        except Exception as exc:
            st.error(f"Cleaning failed: {exc}")

    st.subheader("Current Cleaning Status")

    if cleaning_actions:
        for action in cleaning_actions:
            st.write(f"✓ {action}")
    else:
        st.info("Cleaning has not been run yet.")

    if cleaning_actions:

        st.subheader("Before and After Comparison")
        comparison = compare_quality(original_df, df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows Removed", comparison["rows_removed"])
        col2.metric("Missing Values Filled", comparison["missing_values_filled"])
        col3.metric("Duplicates Remaining", comparison["cleaned_duplicates"])

        comparison_data = pd.DataFrame({
            "Metric": ["Rows", "Missing Values", "Duplicate Rows"],
            "Before Cleaning": [
                comparison["original_rows"],
                comparison["original_missing"],
                comparison["original_duplicates"]
            ],
            "After Cleaning": [
                comparison["cleaned_rows"],
                comparison["cleaned_missing"],
                comparison["cleaned_duplicates"]
            ],
        })

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Cleaned Dataset Preview")
        st.caption("This is the working dataset after the recorded cleaning operations.")

        st.dataframe(df.head(20), use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prepared Dataset",
            data=csv_data,
            file_name="LUMENIQ_Cleaned_Dataset.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "The original dataset is shown above. No cleaning has been "
            "applied yet, so LUMENIQ has not created a before-and-after "
            "comparison. Click 'Clean My Dataset' to prepare the data "
            "and view the comparison."
        )


# ==================================================
# TAB 3: EXPLORE DATA
# ==================================================

with tabs[2]:

    st.header("Interactive Data Exploration")

    st.write(
        "Explore patterns with recommended charts and explanations "
        "of what each chart helps you understand."
    )

    usable = get_usable_columns(df)

    numeric_columns = usable["numeric"]
    categorical_columns = usable["categorical"]
    datetime_columns = usable["datetime"]

    st.subheader("Recommended Visualizations")

    suggestions = get_visualization_suggestions(df)

    if suggestions:

        for idx, suggestion in enumerate(suggestions):

            rec_col, action_col = st.columns([5, 1])

            with rec_col:
                st.markdown(
                    f"**{suggestion['type']}** — {suggestion['reason']}"
                )

                st.caption(
                    "Suggested fields: "
                    + (
                        suggestion['x']
                        or "All usable numerical columns"
                    )
                    + (
                        " + " + suggestion['y']
                        if suggestion['y']
                        else ""
                    )
                )

            with action_col:

                if st.button(
                    "Use chart",
                    key=f"use_recommendation_{idx}",
                    use_container_width=True
                ):

                    st.session_state["selected_chart_type"] = suggestion["type"]
                    st.session_state["chart_type_selector"] = suggestion["type"]
                    st.session_state["recommended_chart_x"] = suggestion["x"]
                    st.session_state["recommended_chart_y"] = suggestion["y"]

                    st.rerun()

    else:
        st.info(
            "No suitable chart recommendations could be generated. "
            "Check whether your dataset contains useful numerical "
            "or categorical columns."
        )

    st.divider()

    chart_default = st.session_state.get(
        "selected_chart_type",
        CHART_TYPES[0]
    )

    chart_type = st.selectbox(
        "Choose Visualization",
        CHART_TYPES,
        index=(
            CHART_TYPES.index(chart_default)
            if chart_default in CHART_TYPES
            else 0
        ),
        key="chart_type_selector"
    )

    guide = CHART_GUIDE[chart_type]

    st.info(
        f"**What this chart shows:** {guide['purpose']}\n\n"
        "**Questions it can answer:** "
        + " | ".join(guide["questions"])
    )

    x = None
    y = None
    color = None

    if chart_type in ["Histogram", "Box Plot"]:

        if numeric_columns:

            default_x = st.session_state.get("recommended_chart_x")

            x = st.selectbox(
                "Select Numeric Column",
                numeric_columns,
                index=(
                    numeric_columns.index(default_x)
                    if default_x in numeric_columns
                    else 0
                )
            )

        else:
            st.warning(
                "No suitable numerical columns are available. "
                "Identifier-like columns are excluded from recommendations."
            )

    elif chart_type == "Pie Chart":

        if categorical_columns:

            default_x = st.session_state.get("recommended_chart_x")

            x = st.selectbox(
                "Select Category",
                categorical_columns,
                index=(
                    categorical_columns.index(default_x)
                    if default_x in categorical_columns
                    else 0
                )
            )

        else:
            st.warning(
                "No low-cardinality categorical columns are available "
                "for a meaningful pie chart."
            )

    elif chart_type == "Correlation Heatmap":

        st.caption(
            "LUMENIQ uses usable numerical columns and excludes "
            "likely identifiers from this chart."
        )

    elif chart_type == "Bar Chart":

        if categorical_columns and numeric_columns:

            default_x = st.session_state.get("recommended_chart_x")
            default_y = st.session_state.get("recommended_chart_y")

            x = st.selectbox(
                "Horizontal axis — categories",
                categorical_columns,
                index=(
                    categorical_columns.index(default_x)
                    if default_x in categorical_columns
                    else 0
                )
            )

            y = st.selectbox(
                "Vertical axis — value to compare",
                numeric_columns,
                index=(
                    numeric_columns.index(default_y)
                    if default_y in numeric_columns
                    else 0
                )
            )

        else:
            st.warning(
                "A bar chart needs a meaningful categorical column "
                "and a numerical measure."
            )

    elif chart_type == "Line Chart":

        x_options = datetime_columns + numeric_columns

        if x_options and numeric_columns:

            default_x = st.session_state.get("recommended_chart_x")

            x = st.selectbox(
                "Horizontal axis — date or sequence",
                x_options,
                index=(
                    x_options.index(default_x)
                    if default_x in x_options
                    else 0
                )
            )

            y_options = [
                col for col in numeric_columns if col != x
            ]

            if y_options:

                default_y = st.session_state.get("recommended_chart_y")

                y = st.selectbox(
                    "Vertical axis — value over time",
                    y_options,
                    index=(
                        y_options.index(default_y)
                        if default_y in y_options
                        else 0
                    )
                )

            else:
                st.warning(
                    "Choose a different X-axis column so the line chart "
                    "has a separate numerical measure."
                )

        else:
            st.warning(
                "A line chart needs an ordered date/numerical column "
                "and a numerical measure."
            )

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            default_x = st.session_state.get("recommended_chart_x")

            x = st.selectbox(
                "Horizontal axis — first measure",
                numeric_columns,
                index=(
                    numeric_columns.index(default_x)
                    if default_x in numeric_columns
                    else 0
                )
            )

            y_options = [
                col for col in numeric_columns if col != x
            ]

            default_y = st.session_state.get("recommended_chart_y")

            y = st.selectbox(
                "Vertical axis — second measure",
                y_options,
                index=(
                    y_options.index(default_y)
                    if default_y in y_options
                    else 0
                )
            )

        else:
            st.warning(
                "A scatter plot requires at least two "
                "suitable numerical columns."
            )

    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"] and categorical_columns:

        st.caption(
            "Compare by category (optional): colour the chart by a group "
            "such as Region or Customer Segment. This separates groups "
            "visually; it does not change the underlying calculation."
        )

        selected_color = st.selectbox(
            "Compare by category (optional)",
            ["None"] + categorical_columns,
            help=(
                "Choose a category to assign different colours "
                "to groups in the chart. Leave as None for a "
                "single-series chart."
            )
        )

        color = (
            None
            if selected_color == "None"
            else selected_color
        )

    st.caption(
        "**How to interpret it:** "
        + guide["interpretation"]
    )

    if st.button("Generate Visualization", type="primary"):

        try:

            theme_base = st.get_option("theme.base") or "light"

            chart = create_chart(
                df,
                chart_type,
                x=x,
                y=y,
                color=color,
                theme=theme_base
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

            st.markdown("**Chart explanation**")

            st.write(
                get_chart_explanation(
                    chart_type,
                    x=x,
                    y=y
                )
            )

        except Exception as exc:
            st.error(f"Visualization error: {exc}")

    st.divider()

    st.subheader("Quick Insights")

    if numeric_columns:

        selected_metric = st.selectbox(
            "Choose a numeric column for a quick summary",
            numeric_columns,
            key="quick_metric"
        )

        series = pd.to_numeric(
            df[selected_metric],
            errors="coerce"
        ).dropna()

        if not series.empty:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Average",
                f"{series.mean():,.2f}"
            )

            col2.metric(
                "Maximum",
                f"{series.max():,.2f}"
            )

            col3.metric(
                "Minimum",
                f"{series.min():,.2f}"
            )


# ==================================================
# TAB 4: PREDICT
# ==================================================

with tabs[3]:

    st.header("Machine Learning Workspace")

    st.write(
        "Tell LUMENIQ what you want to discover. It will recommend "
        "a suitable analysis approach using your dataset and "
        "display its evaluation results."
    )

    task = st.selectbox(
        "Choose Your Analysis Goal",
        [
            "Classification",
            "Regression",
            "Clustering"
        ]
    )

    model_guides = {

        "Classification": (
            "Random Forest Classifier",
            "Predicts a category or class, such as customer segment. "
            "It combines many decision trees and uses their votes, "
            "which can capture non-linear patterns in tabular data. "
            "Useful for churn-risk categories, fraud flags and "
            "customer grouping labels. Metrics: accuracy, precision, "
            "recall and F1-score."
        ),

        "Regression": (
            "Random Forest Regressor",
            "Predicts a numerical value, such as sales or profit. "
            "It averages predictions from multiple decision trees "
            "and can model non-linear relationships in tabular data. "
            "Useful for demand, revenue or price estimation. "
            "Metrics: MAE, RMSE and R²."
        ),

        "Clustering": (
            "K-Means",
            "Groups records into a chosen number of clusters based "
            "on feature similarity, without a target label. "
            "It is useful for exploratory customer or product "
            "segmentation. The silhouette score gives an indication "
            "of how separated the groups are; it does not prove "
            "that clusters are meaningful business segments."
        )
    }

    algorithm_name, algorithm_explanation = model_guides[task]

    st.info(
        f"**LUMENIQ recommends: {task}**\n\n"
        f"{algorithm_explanation}\n\n"
        "You choose the business goal; LUMENIQ selects the "
        "built-in algorithm for that task. You do not need "
        "to choose an algorithm name."
    )

    target = None
    number_of_clusters = 3

    if task == "Classification":

        target_options = [
            col for col in df.columns
            if df[col].nunique(dropna=True) >= 2
            and df[col].nunique(dropna=True) <= max(
                20,
                int(len(df) * 0.20)
            )
            and not is_identifier_column(df[col], col)
        ]

        if target_options:

            target_counts = df[
                target_options[0]
            ].value_counts(dropna=True)

            default_index = (
                target_options.index("Customer_Segment")
                if "Customer_Segment" in target_options
                else 0
            )

            st.caption(
                "Choose an outcome you genuinely want to predict, "
                "such as a customer segment or order status. "
                "LUMENIQ filters likely identifiers from this list."
            )

            target = st.selectbox(
                "What category would you like to predict?",
                target_options,
                index=default_index
            )

            class_counts = df[
                target
            ].value_counts(dropna=True)

            if (class_counts < 2).any():

                st.warning(
                    "Some categories have fewer than two records. "
                    "The model may not be trainable until the target "
                    "has enough examples in each category."
                )

        else:

            st.warning(
                "No suitable target column was found. "
                "Classification needs a meaningful category column "
                "with repeated examples; unique IDs are excluded."
            )

    elif task == "Regression":

        regression_targets = [
            col for col in df.select_dtypes(
                include="number"
            ).columns.tolist()
            if not is_identifier_column(df[col], col)
        ]

        if regression_targets:

            default_index = (
                regression_targets.index("Sales")
                if "Sales" in regression_targets
                else 0
            )

            st.caption(
                "Choose a measurable outcome such as Sales, "
                "Revenue, Profit or Units_Sold—not an order "
                "number or record ID."
            )

            target = st.selectbox(
                "What numerical value would you like to predict?",
                regression_targets,
                index=default_index
            )

        else:

            st.warning(
                "No numeric target column was found."
            )

    else:

        number_of_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=min(10, max(2, len(df) - 1)),
            value=min(3, max(2, len(df) - 1))
        )

    if st.button("🚀 Run Machine Learning", type="primary"):

        try:

            with st.spinner(
                "Training model. Please wait..."
            ):

                if task == "Classification":

                    if target is None:
                        raise ValueError(
                            "Please select a suitable target column."
                        )

                    result = run_classification(
                        df,
                        target
                    )

                elif task == "Regression":

                    if target is None:
                        raise ValueError(
                            "Please select a suitable target column."
                        )

                    result = run_regression(
                        df,
                        target
                    )

                else:

                    result = run_clustering(
                        df,
                        number_of_clusters
                    )

                st.session_state.model_result = result

            st.success(
                "Machine learning analysis completed!"
            )

        except Exception as exc:

            st.session_state.model_result = None

            st.error(
                f"Model execution failed: {exc}"
            )

    result = st.session_state.model_result

    if result:

        st.divider()

        st.subheader("Model Results")

        st.write(
            f"**Task:** {result['task']}"
        )

        st.write(
            f"**Algorithm:** {result['algorithm']}"
        )

        if result["task"] == "Classification":

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{result['accuracy']:.2%}"
            )

            col2.metric(
                "Precision",
                f"{result['precision']:.2%}"
            )

            col3.metric(
                "Recall",
                f"{result['recall']:.2%}"
            )

            col4.metric(
                "F1 Score",
                f"{result['f1_score']:.2%}"
            )

            st.subheader("Confusion Matrix")

            matrix = pd.DataFrame(
                result["confusion_matrix"],
                index=result["classes"],
                columns=result["classes"]
            )

            st.dataframe(
                matrix,
                use_container_width=True
            )

            st.caption(
                "The confusion matrix compares actual and predicted classes."
            )

        elif result["task"] == "Regression":

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "R² Score",
                f"{result['r2_score']:.4f}"
            )

            col2.metric(
                "RMSE",
                f"{result['rmse']:,.2f}"
            )

            col3.metric(
                "MAE",
                f"{result['mae']:,.2f}"
            )

            prediction_df = pd.DataFrame({
                "Actual": result["actual_values"],
                "Predicted": result["predicted_values"]
            })

            st.subheader(
                "Actual vs Predicted Values"
            )

            fig = px.scatter(
                prediction_df,
                x="Actual",
                y="Predicted",
                title="Actual Values Compared with Predictions",
                color_discrete_sequence=["#8b5cf6"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                prediction_df.head(20),
                use_container_width=True
            )

        elif result["task"] == "Clustering":

            col1, col2 = st.columns(2)

            col1.metric(
                "Number of Clusters",
                result["number_of_clusters"]
            )

            col2.metric(
                "Silhouette Score",
                (
                    f"{result['silhouette_score']:.4f}"
                    if result["silhouette_score"] is not None
                    else "Not available"
                )
            )

            st.subheader("Cluster Distribution")

            cluster_counts = pd.DataFrame(
                list(result["cluster_counts"].items()),
                columns=["Cluster", "Records"]
            )

            fig = px.bar(
                cluster_counts,
                x="Cluster",
                y="Records",
                title="Number of Records in Each Cluster",
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                result["clustered_data"].head(20),
                use_container_width=True
            )

        st.info(
            "Model results are based on the uploaded dataset and "
            "the selected train-test split or clustering configuration. "
            "They should be validated before business use."
        )


# ==================================================
# TAB 5: AI ASSISTANT
# ==================================================

with tabs[4]:

    st.header("LUMENIQ Data Assistant")

    st.write(
        "Ask questions about your uploaded dataset, data quality, "
        "cleaning operations and machine learning results."
    )

    st.info(
        "Ask in everyday language. LUMENIQ calculates supported "
        "answers directly from your loaded dataset on this device; "
        "your raw dataset is not sent to an external API. "
        "It understands common data-analysis questions, "
        "not unrestricted general conversation."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button(
        "Clear Conversation",
        key="clear_lumeniq_chat"
    ):

        st.session_state.chat_history = []
        st.session_state.pending_question = None

        st.rerun()

    st.subheader("Try asking")

    examples = [
        "Hello LUMENIQ",
        "Give me a dataset overview",
        "Which columns contain missing values?",
        "What is the average Sales?",
        "Which product has the highest Sales?",
        "Show the distribution of Product",
        "Which columns are correlated?",
        "What cleaning actions were performed?",
        "Explain the latest model results."
    ]

    example_cols = st.columns(2)

    for index, example in enumerate(examples):

        if example_cols[index % 2].button(
            example,
            key=f"chat_example_{index}",
            use_container_width=True
        ):

            st.session_state.pending_question = example
            st.rerun()

    question = st.chat_input(
        "Ask about your dataset in everyday language..."
    )

    question_to_answer = (
        question
        or st.session_state.pending_question
    )

    if question_to_answer:

        st.session_state.pending_question = None

        st.session_state.chat_history.append({
            "role": "user",
            "content": question_to_answer
        })

        answer = answer_dataset_question(
            df,
            question_to_answer,
            cleaning_actions,
            st.session_state.model_result
        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()


# ==================================================
# TAB 6: REPORTS
# ==================================================

with tabs[5]:

    st.header("Download Analysis Report")

    st.write(
        "Generate a PDF report containing dataset quality, "
        "cleaning operations, column information and model results."
    )

    st.subheader("Report Contents")

    st.write("✓ Dataset overview")
    st.write("✓ Data quality information")
    st.write("✓ Before and after cleaning comparison")
    st.write("✓ Cleaning operations")
    st.write("✓ Column information")
    st.write("✓ Latest machine learning results, if available")
    st.write("✓ Interpretation, limitations and privacy notes")

    if st.button(
        "Generate PDF Report",
        type="primary"
    ):

        try:

            pdf_bytes = generate_pdf_report(
                df=df,
                original_df=original_df,
                cleaning_actions=cleaning_actions,
                model_result=st.session_state.model_result,
                dataset_name=st.session_state.dataset_name
            )

            st.download_button(
                label="⬇️ Download LUMENIQ PDF Report",
                data=pdf_bytes,
                file_name="LUMENIQ_Analysis_Report.pdf",
                mime="application/pdf"
            )

        except Exception as exc:

            st.error(
                f"Report generation failed: {exc}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "LUMENIQ | Intelligent Data Analysis | Automated Exploration "
    "and Predictive Analytics"
)
