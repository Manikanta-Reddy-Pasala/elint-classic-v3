import os
from collections import Counter
from typing import List, Tuple, Dict, Generator, Any, Optional
import pandas as pd
import pytest
from _pytest.config import Config
from _pytest.reports import TestReport
from _pytest.nodes import Item
from _pytest.runner import CallInfo
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for PDF rendering, no display needed
import matplotlib.pyplot as plt
from _pytest.config import Config
from _pytest.reports import TestReport
from _pytest.nodes import Item
from _pytest.runner import CallInfo

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class ParameterizedSuiteReporter:
    """Tracks tests across a parameterized loop, categorizes errors into functional buckets,

    and builds a comprehensive PDF summary document after the final loop execution run.
    """

    def __init__(self, suite_name: str) -> None:
        self.suite_name: str = suite_name
        self.custom_title: Optional[str] = None
        self.custom_description: Optional[str] = None
        self.total_runs: int = 0
        self.passed: int = 0
        self.failed: int = 0

        # Exact Refactored Categorized Error Buckets
        self.errors_one_not_detected: List[str] = []
        self.errors_total_count_failed: List[str] = []
        self.errors_unknown_or_other: List[str] = []
        self.errors_other_emitters_detected: List[str] = []
        self.errors_total_count_not_detected_warning: List[str] = []
        self.final_confusion_matrix: Optional[pd.DataFrame] = None

    def record_run(self, passed: bool, error_details: Optional[List[str]] = None) -> None:
        """error_details may be a list of independent messages, each categorized separately."""
        self.total_runs += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        if error_details:
            if isinstance(error_details, str):
                self._categorize_error(error_details)
            else:
                for msg in error_details:
                    self._categorize_error(msg)

    def _categorize_error(self, err_text: str) -> None:
        """Deduces functional exception classification using targeted assertpy signature searches."""
        lower_txt = err_text.lower()

        # --- CATEGORY 0 (NEW): OTHER EMITTERS DETECTED WARNING ---
        # Matches the custom warning raised in tests/test_pdw_generator.py when the
        # emitters table contains more entries than emitter_1, emitter_2 and 'unknown_ai'.
        if "system detected other known emitters" in lower_txt or \
                "other emitters detected except" in lower_txt:
            self.errors_other_emitters_detected.append(err_text)
            return

        # --- CATEGORY 0b (NEW): NOT DETECTING FULL NUMBER OF EMITTERS WARNING ---
        # Matches the NON-BLOCKING warning you raised in the testcase via
        # request.node.stash["custom_db_warning"]:
        #   "AI model failed to detect total number of emitters(controlled + unknown, random emitters)
        #    Expected {n} with standard deviation of +- {std} emitters Got only {count}"
        # Note: this must come BEFORE Category 2, because Category 2's
        # "failed to detect total number" substring would otherwise swallow it
        # and report it as a hard failure instead of a warning.
        if "ai model failed to detect total number of emitters" in lower_txt:
            self.errors_total_count_not_detected_warning.append(err_text)
            return


        # --- CATEGORY 1: ONE KNOWN EMITTER WAS NOT DETECTED ---
        # Matches: "AI model failed to detect known emitter..." OR "...specifications: Pulse Width... was not detected by AI model"
        if "failed to detect known emitter" in lower_txt or "specifications:" in lower_txt or "was not detected by ai model" in lower_txt:
            self.errors_one_not_detected.append(err_text)

        # --- CATEGORY 2: TOTAL NUMBER OF EMITTERS WAS NOT DETECTED ---
        # Matches: "failed to detect total number of emitters..." OR "failed to detected minimum number of emitters..." OR "expected number of emitters to be between..."
        elif ("failed to detect total number" in lower_txt or
              "failed to detected minimum number" in lower_txt or
              "expected number of emitters to be between" in lower_txt or
              "outside the expected standard deviation" in lower_txt):
            self.errors_total_count_failed.append(err_text)

        # --- CATEGORY 3: ANY OTHER UNKNOWN ERRORS ---
        # Fallback for unexpected python/pytest infrastructure failures
        else:
            self.errors_unknown_or_other.append(err_text)
        print(f"[DBG bucket] one_not_detected={len(self.errors_one_not_detected)} "
              f"total_count={len(self.errors_total_count_failed)} "
              f"unknown={len(self.errors_unknown_or_other)} "
              f"other_emitters={len(self.errors_other_emitters_detected)}", flush=True)
    def attach_final_dataframe(self, df: pd.DataFrame) -> None:
        self.final_confusion_matrix = df.copy()

    def render_confusion_matrix_heatmap(self, df: pd.DataFrame, out_path: str) -> None:
        """Render the confusion-matrix DataFrame as a matplotlib heatmap (PNG).

        df is a percentage matrix (rows = true label, cols = predicted label), attached
        via attach_final_dataframe(). Cell values are already in the 0-100 range, so they
        are parsed as-is (no fraction scaling). Uses POSITIONAL (iloc) access for the
        label column so it is immune to whatever the index/column names actually are.
        """
        df_reset = df.reset_index().copy()

        # POSITIONAL: first column = row labels, remaining = data
        row_labels = df_reset.iloc[:, 0].astype(str).tolist()
        data_df = df_reset.iloc[:, 1:]

        # Drop any 'total' column by name (only if present)
        data_df = data_df.drop(columns=[c for c in data_df.columns if str(c).lower() == "total"])
        data_cols = list(data_df.columns)

        def to_pct(val):
            s = str(val).strip()
            try:
                if s.endswith("%"):
                    return float(s[:-1])
                return float(s)           # values are already 0-100 percentages
            except ValueError:
                return 0.0                # crash-proof: non-numeric -> 0

        numeric_matrix = data_df.map(to_pct)                # pandas 2.2.2 -> .map()

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(numeric_matrix.values, cmap="viridis", aspect="auto")

        ax.set_xticks(range(len(data_cols)))
        ax.set_yticks(range(len(numeric_matrix)))
        ax.set_xticklabels(data_cols, rotation=45, ha="right")
        ax.set_yticklabels(row_labels)

        ax.set_title("Confusion Matrix Evaluation Output (%)", fontsize=12, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of rows")

        vmax = np.nanmax(numeric_matrix.values)
        for i in range(numeric_matrix.shape[0]):
            for j in range(numeric_matrix.shape[1]):
                val = numeric_matrix.iat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > vmax * 0.6 else "black",
                        fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def generate_pdf(self) -> str:
        """Compiles the telemetry metrics, charts, grouped exception maps, and matrices into the final sheet document."""
        display_title = self.custom_title if self.custom_title else "Parameterized Test Run Summary"
        safe_name = "".join([c if c.isalnum() else "_" for c in self.suite_name])
        filename = f"final_summary_{safe_name}.pdf"

        doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'Title', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor("#1A365D"),
            spaceAfter=10
        )
        description_style = ParagraphStyle(
            'DescText', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor("#4A5568"),
            spaceAfter=15
        )
        section_style = ParagraphStyle(
            'Section', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor("#2D3748"),
            spaceBefore=15, spaceAfter=6
        )
        bucket_header_style = ParagraphStyle(
            'BucketHeader', parent=styles['Heading3'], fontSize=10, leading=14, textColor=colors.HexColor("#7B341E"),
            spaceBefore=10, spaceAfter=4
        )
        failure_text_style = ParagraphStyle(
            'FailureText', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor("#991B1B"),
            wordWrap='CJK'
        )

        story: List[Any] = []
        story.append(Paragraph(display_title, title_style))
        if self.custom_description:
            story.append(Paragraph(self.custom_description, description_style))

        story.append(Paragraph(f"<b>Base Test Scope:</b> {self.suite_name}", styles['Normal']))
        story.append(Spacer(1, 10))

        # 1. Global Statistics Grid
        story.append(Paragraph("Global Execution Statistics", section_style))
        summary_data = [
            [Paragraph("<b>Metric</b>", styles['Normal']), Paragraph("<b>Count</b>", styles['Normal'])],
            ["Total Test Scenarios Checked", str(self.total_runs)],
            ["Passed Case Scenarios", str(self.passed)],
            ["Failed Case Scenarios", str(self.failed)]
        ]

        t = Table(summary_data, colWidths=[270, 270])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#E2E8F0")),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor("#1A365D")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TEXTCOLOR', (1, 2), (1, 2), colors.green),
            ('TEXTCOLOR', (1, 3), (1, 3), colors.red if self.failed > 0 else colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)

        # 2. GROUPED FUNCTIONAL FAILURES SECTION
        has_any_errors = bool(
            self.errors_one_not_detected
            or self.errors_total_count_failed
            or self.errors_unknown_or_other
            or self.errors_total_count_not_detected_warning

        )

        if has_any_errors:
            story.append(Paragraph("Functional Error Classification Summary", section_style))
            story.append(Spacer(1, 4))

            def render_error_bucket(bucket_title: str, error_list: List[str], is_warning: bool = False) -> None:
                if not error_list:
                    return

                if is_warning:
                    header_style = ParagraphStyle(
                        'WarnHeader',
                        parent=styles['Heading3'],
                        fontSize=10, leading=14,
                        textColor=colors.HexColor("#92400E"),
                        spaceBefore=10, spaceAfter=4,
                    )
                    story.append(Paragraph(f"<b>⚠️ {bucket_title}</b>", header_style))
                else:
                    story.append(Paragraph(f"<b>📂 {bucket_title}</b>", bucket_header_style))

                counts = Counter(error_list).most_common(5)
                for idx, (msg, freq) in enumerate(counts, 1):
                    story.append(Paragraph(f"Variant Log Case #{idx} (Occurrences: {freq})", styles['Normal']))
                    story.append(Spacer(1, 2))

                    err_para = Paragraph(msg, failure_text_style)
                    err_card_table = Table([[err_para]], colWidths=[540])

                    if is_warning:
                        err_card_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#FCD34D")),
                            ('PADDING', (0, 0), (-1, -1), 8),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ]))
                    else:
                        err_card_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEE2E2")),
                            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#FCA5A5")),
                            ('PADDING', (0, 0), (-1, -1), 8),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ]))

                    story.append(err_card_table)
                    story.append(Spacer(1, 6))

            # Warnings first, then hard-failure buckets
            render_error_bucket(
                "Other Emitters Detected (Non-Blocking Warning)",
                self.errors_other_emitters_detected,
                is_warning=True,
            )
            render_error_bucket(
                "Not Detecting Full Number of Emitters (Non-Blocking Warning)",
                self.errors_total_count_not_detected_warning,
                is_warning=True,
            )
            # Render each bucket sequentially down the document canvas
            render_error_bucket("Category 1: One Known Emitter Was Not Detected", self.errors_one_not_detected)
            render_error_bucket("Category 2: Total Number of Emitters Deviation Failures",
                                self.errors_total_count_failed)
            render_error_bucket("Category 3: Any Other Unknown Errors / System Tracebacks",
                                self.errors_unknown_or_other)

            # 3. Confusion Matrix Section (text table + heatmap)
            if self.final_confusion_matrix is not None:
                story.append(Paragraph("Final Confusion Matrix Evaluation Output (%)", section_style))
                story.append(Spacer(1, 4))

                # --- 3a. Text table (exact values, kept for reference/searchability) ---
                df_reset = self.final_confusion_matrix.reset_index()
                matrix_data = [[Paragraph(f"<b>{col}</b>", styles['Normal']) for col in df_reset.columns]]

                for _, row in df_reset.iterrows():
                    matrix_data.append([str(val) if not isinstance(val, float) else f"{val:.2f}%" for val in row])

                col_count = len(df_reset.columns)
                col_widths = [540 / col_count] * col_count

                ct = Table(matrix_data, colWidths=col_widths)
                ct.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F7FAFC")),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ('PADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(ct)
                story.append(Spacer(1, 12))  # breathing room between table and heatmap

                # --- 3b. Heatmap image (visual view of the same data) ---
                heatmap_png = os.path.join(
                    os.path.dirname(os.path.abspath(filename)),
                    f"confusion_heatmap_{safe_name}.png"
                )
                self.render_confusion_matrix_heatmap(self.final_confusion_matrix, heatmap_png)

                if os.path.exists(heatmap_png):
                    story.append(Paragraph("Heatmap View", section_style))
                    story.append(Spacer(1, 4))
                    # Fit the image to the usable page width (letter 612pt − 36pt×2 margins = 540pt)
                    img = Image(heatmap_png, width=540, height=340)
                    story.append(img)

            doc.build(story)
            return os.path.abspath(filename)