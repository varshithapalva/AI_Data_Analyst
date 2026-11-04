
from io import BytesIO
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


def generate_pdf_report(
    df,
    original_df=None,
    cleaning_actions=None,
    model_result=None,
    dataset_name="Dataset"
):
    """
    Generate a PDF report for the LUMENIQ AI Data Analyst project.
    Returns PDF content as bytes.
    """

    if df is None or df.empty:
        raise ValueError("Cannot generate a report for an empty dataset.")

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=6
    )

    story = []

    # Title
    story.append(Paragraph("LUMENIQ - AI DATA ANALYST", title_style))
    story.append(
        Paragraph("Automated Data Exploration and Predictive Analytics Report",
                  subtitle_style)
    )

    story.append(Paragraph(
        f"<b>Dataset:</b> {dataset_name}<br/>"
        f"<b>Generated on:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        body_style
    ))

    # Dataset overview
    story.append(Paragraph("1. Dataset Overview", heading_style))

    overview_data = [
        ["Metric", "Value"],
        ["Number of Rows", f"{len(df):,}"],
        ["Number of Columns", f"{len(df.columns):,}"],
        ["Total Missing Values", f"{int(df.isnull().sum().sum()):,}"],
        ["Duplicate Rows", f"{int(df.duplicated().sum()):,}"],
        ["Numeric Columns", str(len(df.select_dtypes(include="number").columns))],
        ["Categorical Columns", str(len(df.select_dtypes(include=["object", "category", "string"]).columns))]
    ]

    overview_table = Table(
        overview_data,
        colWidths=[3.0 * inch, 3.0 * inch]
    )

    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F7FB")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(overview_table)

    # Data quality before and after cleaning
    if original_df is not None:
        story.append(Paragraph("2. Data Quality Comparison", heading_style))

        comparison_data = [
            ["Quality Metric", "Before Cleaning", "After Cleaning"],
            [
                "Rows",
                str(len(original_df)),
                str(len(df))
            ],
            [
                "Missing Values",
                str(int(original_df.isnull().sum().sum())),
                str(int(df.isnull().sum().sum()))
            ],
            [
                "Duplicate Rows",
                str(int(original_df.duplicated().sum())),
                str(int(df.duplicated().sum()))
            ]
        ]

        comparison_table = Table(
            comparison_data,
            colWidths=[2.0 * inch, 2.0 * inch, 2.0 * inch]
        )

        comparison_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))

        story.append(comparison_table)

    # Cleaning actions
    story.append(Paragraph("3. Data Cleaning Operations", heading_style))

    if cleaning_actions:
        for action in cleaning_actions:
            story.append(Paragraph(f"- {action}", body_style))
    else:
        story.append(Paragraph(
            "No cleaning actions were recorded.",
            body_style
        ))

    # Column information
    story.append(Paragraph("4. Column Information", heading_style))

    column_data = [["Column Name", "Data Type", "Missing Values", "Unique Values"]]

    for column in df.columns:
        column_data.append([
            str(column),
            str(df[column].dtype),
            str(int(df[column].isnull().sum())),
            str(int(df[column].nunique(dropna=True)))
        ])

    column_table = Table(
        column_data,
        colWidths=[2.1 * inch, 1.2 * inch, 1.2 * inch, 1.5 * inch],
        repeatRows=1
    )

    column_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(column_table)

    # Machine learning results
    story.append(Paragraph("5. Machine Learning Results", heading_style))

    if model_result:
        story.append(Paragraph(
            f"<b>Task:</b> {model_result.get('task', 'Not specified')}<br/>"
            f"<b>Algorithm:</b> {model_result.get('algorithm', 'Not specified')}",
            body_style
        ))

        display_keys = [
            "target",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "r2_score",
            "mse",
            "rmse",
            "mae",
            "silhouette_score",
            "inertia",
            "number_of_clusters",
            "training_rows",
            "testing_rows"
        ]

        metric_data = [["Metric", "Result"]]

        for key in display_keys:
            if key in model_result and model_result[key] is not None:
                metric_data.append([
                    key.replace("_", " ").title(),
                    str(model_result[key])
                ])

        if len(metric_data) > 1:
            metric_table = Table(
                metric_data,
                colWidths=[3.0 * inch, 3.0 * inch]
            )

            metric_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("PADDING", (0, 0), (-1, -1), 7)
            ]))

            story.append(metric_table)
    else:
        story.append(Paragraph(
            "No machine learning model has been executed for this report.",
            body_style
        ))

    # Limitations and privacy
    story.append(Paragraph("6. Interpretation and Limitations", heading_style))

    story.append(Paragraph(
        "Machine learning results depend on the dataset, selected features, "
        "data quality, and train-test split. Evaluation scores should not be "
        "treated as guarantees of future performance. Correlation does not "
        "establish causation. Clustering results describe groupings in the "
        "provided data and may not represent naturally distinct populations.",
        body_style
    ))

    story.append(Paragraph("7. Data Privacy", heading_style))

    story.append(Paragraph(
        "LUMENIQ is designed around local dataset processing. This report "
        "contains dataset summaries and analysis results rather than a "
        "complete copy of the uploaded dataset.",
        body_style
    ))

    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "Generated by LUMENIQ - AI Data Analyst",
        subtitle_style
    ))

    document.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
