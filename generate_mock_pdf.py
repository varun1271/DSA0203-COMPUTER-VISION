import os
import cv2
import reportlab
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

TASKS_12 = [
    ("task01_face_detection.py", "Task 01: Real-Time Face Detection", "task01_face_detection_result.jpg", "Detects faces using multi-scale Haar Cascade classifiers."),
    ("task02_expression.py", "Task 02: Facial Expression & Emotion Recognition", "task02_expression_result.jpg", "Classifies emotions (Happy, Surprised, Neutral) via landmark ratio analysis."),
    ("task03_ocr.py", "Task 03: Optical Character Recognition (OCR)", "task03_ocr_result.jpg", "Extracts text from live camera frames using EasyOCR / adaptive contours."),
    ("task04_motion_alert.py", "Task 04: Real-Time Motion Security Intelligence", "task04_motion_alert_result.jpg", "Performs frame differencing, trajectory tracking, and security alerts."),
    ("task05_drawing.py", "Task 05: Virtual Air Canvas / Air Drawing", "task05_drawing_result.jpg", "Tracks finger tip in 2D space to draw neon glow lines on camera stream."),
    ("task06_edge_detection.py", "Task 06: Interactive Real-Time Edge Scanner", "task06_edge_detection_result.jpg", "Applies Canny, Sobel, and Laplacian holographic filters on live video."),
    ("task07_tracking.py", "Task 07: Object Tracking (CSRT/KCF/MIL)", "task07_tracking_result.jpg", "Locks onto user-selected ROI and tracks position vectors across frames."),
    ("task08_feature_matching.py", "Task 08: Feature Detection & Matching (ORB/SIFT)", "task08_feature_matching_result.jpg", "Matches keypoints between target reference and camera feed using RANSAC."),
    ("task09_license_plate.py", "Task 09: Automatic License Plate Recognition (ALPR)", "task09_license_plate_result.jpg", "Locates rectangular plate geometry and isolates license plate characters."),
    ("task10_background_subtraction.py", "Task 10: Background Subtraction & Virtual BG", "task10_background_subtraction_result.jpg", "Separates foreground subject from background using MOG2 / KNN subtractors."),
    ("task11_face_counting.py", "Task 11: Live Face Counter & Crowd Analytics", "task11_face_counting_result.jpg", "Tracks active & peak face count with spatial density zone partitioning."),
    ("task12_image_to_video.py", "Task 12: Real-Time Stream Recorder & Studio HUD", "task12_result.jpg", "Encodes live video into MP4 with timecode counter & audio spectrum HUD.")
]

class TwelvePageCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Landscape header & footer lines (Width = 841.89, Height = 595.27)
        self.setStrokeColor(colors.HexColor("#0284C7"))
        self.setLineWidth(1.5)
        self.line(20, 575, 821, 575)

        self.drawString(20, 582, "COMPUTER VISION MINI PROJECTS - 12-PAGE MOCK CODEBOOK (CODE & MOCK OUTPUT SIDE-BY-SIDE)")

        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(20, 20, 821, 20)

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(821, 10, page_str)
        self.drawString(20, 10, "DSA0203 Computer Vision Suite | 12 Tasks Mock Showcase")
        self.restoreState()

def format_code_lines_to_html(lines, start_idx=1):
    formatted = []
    for line_num, line in enumerate(lines, start=start_idx):
        line_str = line.rstrip("\n\r")
        indent_count = len(line_str) - len(line_str.lstrip(" "))
        indent_spaces = "&nbsp;" * indent_count
        trimmed_line = line_str.lstrip(" ")
        
        escaped_line = (trimmed_line.replace("&", "&amp;")
                                    .replace("<", "&lt;")
                                    .replace(">", "&gt;"))

        line_formatted = f"<font color='#94A3B8'>{line_num:02d}&nbsp;</font>{indent_spaces}{escaped_line}"
        formatted.append(line_formatted)
    return "<br/>".join(formatted)

def create_mock_codebook_pdf(output_filename="CV_Mini_Projects_Mock_Codebook.pdf"):
    pdf_path = os.path.abspath(output_filename)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        leftMargin=20,
        rightMargin=20,
        topMargin=25,
        bottomMargin=25
    )

    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TaskTitleStyle',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#0F172A")
    )

    desc_style = ParagraphStyle(
        'TaskDescStyle',
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#334155")
    )

    print(f"[PDF] Generating 12-page Mock Side-by-Side Codebook...")

    for file_idx, (filename, title, output_img_name, description) in enumerate(TASKS_12, start=1):
        if not os.path.exists(filename):
            print(f"[WARNING] File not found: {filename}")
            continue

        with open(filename, "r", encoding="utf-8") as f:
            code_lines = f.readlines()

        total_lines = len(code_lines)
        output_img_path = os.path.join("output", output_img_name)

        # Build Code Column (Single or 2-Subcolumn for long files > 80 lines)
        if total_lines > 80:
            half = (total_lines + 1) // 2
            col1_lines = code_lines[:half]
            col2_lines = code_lines[half:]

            max_lines = max(len(col1_lines), len(col2_lines))
            leading = 390.0 / max_lines
            leading = min(leading, 7.5)
            leading = max(leading, 4.0)
            font_size = min(6.8, leading * 0.82)

            code_style = ParagraphStyle(
                f'CodeStyle_Split_{file_idx}',
                fontName='Courier',
                fontSize=font_size,
                leading=leading,
                textColor=colors.HexColor("#0F172A"),
                backColor=colors.HexColor("#F8FAFC"),
                borderPadding=(1, 2, 1, 2),
                borderColor=colors.HexColor("#CBD5E1"),
                borderWidth=0.5,
                borderRadius=2
            )

            p1 = Paragraph(format_code_lines_to_html(col1_lines, 1), code_style)
            p2 = Paragraph(format_code_lines_to_html(col2_lines, half + 1), code_style)
            
            code_container = Table([[p1, p2]], colWidths=[240, 240])
            code_container.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
        else:
            leading = 440.0 / max(total_lines, 1)
            leading = min(leading, 7.8)
            leading = max(leading, 4.5)
            font_size = min(7.2, leading * 0.82)

            code_style = ParagraphStyle(
                f'CodeStyle_Single_{file_idx}',
                fontName='Courier',
                fontSize=font_size,
                leading=leading,
                textColor=colors.HexColor("#0F172A"),
                backColor=colors.HexColor("#F8FAFC"),
                borderPadding=(2, 3, 2, 3),
                borderColor=colors.HexColor("#CBD5E1"),
                borderWidth=0.5,
                borderRadius=3
            )

            code_container = Paragraph(format_code_lines_to_html(code_lines, 1), code_style)

        # -------------------------------------------------------------
        # RIGHT COLUMN: MOCK OUTPUT DISPLAY CARD (Width = 295 pt)
        # -------------------------------------------------------------
        right_elements = []
        right_elements.append(Paragraph(f"<b>{file_idx:02d}. {title}</b>", title_style))
        right_elements.append(Spacer(1, 2))
        right_elements.append(Paragraph(f"<b>File:</b> <code>{filename}</code> | <b>Total Lines:</b> {total_lines}<br/>{description}", desc_style))
        right_elements.append(Spacer(1, 6))

        if os.path.exists(output_img_path):
            try:
                img_widget = Image(output_img_path, width=280, height=175)
                right_elements.append(img_widget)
                right_elements.append(Spacer(1, 3))
                right_elements.append(Paragraph("<font size=7 color='#0284C7'><b>[MOCK / DEMO OUTPUT DISPLAY SNAPSHOT]</b></font>", desc_style))
            except Exception as e:
                print(f"[NOTICE] Error loading image {output_img_path}: {e}")

        # Assemble Side-by-Side Table Layout
        side_table = Table([[code_container, right_elements]], colWidths=[485, 295])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (1, 0), (1, 0), 10),
            ('RIGHTPADDING', (0, 0), (0, 0), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))

        story.append(side_table)

        # PageBreak so each of the 12 tasks occupies exactly 1 page
        if file_idx < 12:
            story.append(PageBreak())

    doc.build(story, canvasmaker=TwelvePageCanvas)
    print(f"[SUCCESS] 12-Page Mock Codebook PDF created successfully at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_mock_codebook_pdf()
