# LUMENIQ — Intelligent Data Analysis Workspace

## Abstract
LUMENIQ is a local-first, Streamlit-based application that guides users through structured-data exploration, data preparation, visualisation, baseline machine learning and report generation. It is intended to make common analytical workflows more approachable to users who may not have a technical background.

## Modules
1. **Data ingestion:** loads CSV and Excel datasets.
2. **Profiling:** reports dataset dimensions, column types, missing values, duplicates and summaries.
3. **Preparation:** retains the original dataset and creates a separately cleaned working copy.
4. **Visualisation:** recommends charts based on usable numerical, categorical and date columns and provides chart explanations.
5. **Prediction:** provides classification, regression and clustering using Random Forest and K-Means algorithms.
6. **Dataset assistant:** handles common natural-language-style questions by matching analytical intents and calculating answers locally.
7. **Reporting:** creates a downloadable PDF summary.

## Distinctive project combination
The project combines local dataset question answering, identifier-aware chart recommendations, guided visualisation explanations, original-versus-cleaned comparison, three baseline ML task types and PDF reporting in one application. Individual capabilities also exist in other analytics products; the contribution is their integration into this guided workflow.

## Evaluation and limitations
The application should be tested with multiple datasets, including small, imbalanced and messy data. Rule-based question interpretation has limits and may not understand every phrasing. Model metrics do not guarantee business performance. Users should inspect cleaning actions, target suitability and model evaluation before relying on results.

## Future scope
Potential extensions include optional local/hosted LLM-based question interpretation, richer follow-up context, forecasting, anomaly detection, feature-importance explanations, configurable aggregation and user-controlled cleaning policies.
