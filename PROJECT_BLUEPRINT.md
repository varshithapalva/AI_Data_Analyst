# LUMENIQ Project Blueprint

## Project title
LUMENIQ — Intelligent Data Analysis Workspace

## Problem statement
People who are not trained in programming or data science often find it difficult to inspect raw datasets, identify data-quality issues, choose useful visualisations and interpret analytical results. LUMENIQ brings these tasks together in a guided interface.

## Objectives
- Load common tabular data formats.
- Profile and prepare datasets while preserving the uploaded original.
- Recommend meaningful visualisations and explain their purpose.
- Provide baseline classification, regression and clustering workflows.
- Answer common questions about the active dataset using local calculations.
- Export a consolidated PDF report.

## Architecture
Streamlit interface → ingestion and profiling → cleaning → visualisation / machine learning / local question answering → PDF reporting.

## Algorithms
- Random Forest Classifier: predicts categorical outcomes.
- Random Forest Regressor: predicts numerical outcomes.
- K-Means: groups records by feature similarity without a target label.

## Important limitations
The assistant is rule-based and is not a general-purpose LLM. Automatic cleaning applies predefined rules and should be reviewed. Model scores are baseline estimates and may be affected by dataset size, imbalance, leakage and feature quality. Correlation is not causation.
