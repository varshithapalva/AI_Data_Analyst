# LUMENIQ — Intelligent Data Analysis Workspace

LUMENIQ is a local-first Streamlit application for exploring structured datasets, preparing data, creating visualisations, running baseline machine-learning tasks, asking dataset-aware questions and exporting a PDF report.

## Main capabilities
- Upload CSV, XLSX and XLS files.
- Inspect dataset shape, column types, missing values and duplicate records.
- Preserve the uploaded original and create a separate cleaned working dataset.
- Recommend and generate interactive charts with explanations.
- Run classification (Random Forest), regression (Random Forest) and clustering (K-Means).
- Ask common dataset questions using a local rule-based analysis assistant.
- Export a PDF summary.

## Setup on Windows
Open this folder in VS Code, then run in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## How to use
1. Upload a CSV or Excel dataset from the sidebar.
2. Review the Overview tab.
3. In Prepare Data, inspect the original data and select **Clean My Dataset** when ready.
4. In Explore Data, choose a recommendation or chart, review its explanation, and optionally compare categories using colour grouping.
5. In Predict, choose the business goal: category prediction, numeric prediction or pattern discovery. The app uses its built-in algorithm for the selected task; the user does not need to choose an algorithm by name.
6. In the Data Assistant, ask supported questions about the loaded dataset.
7. Generate a PDF in Reports.

## Privacy and limitations
Dataset calculations and the question-answering engine run locally in this application. The assistant is a dataset-aware rule-based system, not an unrestricted large language model; it supports common analytical intents and may ask for clearer wording when it cannot confidently interpret a question. Machine-learning results are baseline estimates and depend on dataset size, quality, target suitability and feature leakage. Validate results before making business decisions.

## Project structure
```text
LUMENIQ/
├── app.py
├── core/
│   ├── ingestion.py
│   ├── profiling.py
│   ├── cleaning.py
│   ├── visualization.py
│   ├── modeling.py
│   ├── assistant.py
│   └── reporting.py
├── data/
├── tests/
├── .streamlit/config.toml
└── requirements.txt
```
