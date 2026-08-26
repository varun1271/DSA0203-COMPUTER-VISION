import os
import cv2
import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

FILES_WITH_METADATA = [
    ("task01_face_detection.py", "Task 01: Real-Time Face Detection", "task01_face_detection_result.jpg"),
    ("task02_expression.py", "Task 02: Facial Expression & Emotion Recognition", "task02_expression_result.jpg"),
    ("task03_ocr.py", "Task 03: Optical Character Recognition (OCR)", "task03_ocr_result.jpg"),
    ("task04_motion_alert.py", "Task 04: Real-Time Motion Security Intelligence", "task04_motion_alert_result.jpg"),
    ("task05_drawing.py", "Task 05: Virtual Air Canvas / Air Drawing", "task05_drawing_result.jpg"),
    ("task06_edge_detection.py", "Task 06: Interactive Real-Time Edge Scanner", "task06_edge_detection_result.jpg"),
    ("task07_tracking.py", "Task 07: Object Tracking (CSRT/KCF/MIL)", "task07_tracking_result.jpg"),
    ("task08_feature_matching.py", "Task 08: Feature Detection & Matching (ORB/SIFT)", "task08_feature_matching_result.jpg"),
    ("task09_license_plate.py", "Task 09: Automatic License Plate Recognition (ALPR)", "task09_license_plate_result.jpg"),
    ("task10_background_subtraction.py", "Task 10: Background Subtraction & Virtual BG", "task10_background_subtraction_result.jpg"),
    ("task11_face_counting.py", "Task 11: Live Face Counter & Crowd Analytics", "task11_face_counting_result.jpg"),
    ("task12_image_to_video.py", "Task 12: Real-Time Stream Recorder & Studio HUD", "task12_result.jpg"),
    ("utils.py", "Utility Helpers & Video Stream Core", None),
    ("run_all.py", "Master Projects Runner", None)
]

def prepare_task12_snapshot():
    output_dir = "output"
    video_path = os.path.join(output_dir, "task12_output_video.mp4")
    target_img_path = os.path.join(output_dir, "task12_result.jpg")

    if os.path.exists(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.imwrite(target_img_path, frame)
                print(f"[INFO] Extracted task 12 frame to: {target_img_path}")
            cap.release()
        except Exception as e:
            print(f"[NOTICE] Could not extract task 12 frame: {e}")

class OnePageCodeCanvas(canvas.Canvas):
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
        self.setFillColor(colors.HexColor("#555555"))
        
        # Top line header
        self.setStrokeColor(colors.HexColor("#00AA88"))
        self.setLineWidth(1)
        self.line(30, 812, 565, 812)

        # Header Title
        self.drawString(30, 820, "COMPUTER VISION REAL-TIME WEBCAM CODEBOOK & LIVE OUTPUTS")

        # Footer Line
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(30, 28, 565, 28)

        # Footer Text
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(565, 16, page_str)
        self.drawString(30, 16, "DSA0203 - Computer Vision Mini Projects Suite")
        self.restoreState()

def create_codebook_pdf(output_filename="CV_Mini_Projects_Codebook.pdf"):
    prepare_task12_snapshot()
    pdf_path = os.path.abspath(output_filename)
    
    # A4 Page Dimensions: 595.27 x 841.89
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=35,
        bottomMargin=35
    )

    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#0A2540")
    )

    filename_style = ParagraphStyle(
        'FileNameStyle',
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#007755")
    )

    print(f"[PDF] Compiling codebook with live outputs ({len(FILES_WITH_METADATA)} pages)...")

    for file_idx, (filename, title, output_img_name) in enumerate(FILES_WITH_METADATA, start=1):
        if not os.path.exists(filename):
            print(f"[WARNING] File not found: {filename}")
            continue

        with open(filename, "r", encoding="utf-8") as f:
            code_lines = f.readlines()

        total_lines = len(code_lines)
        output_img_path = os.path.join("output", output_img_name) if output_img_name else None
        has_output_image = output_img_path and os.path.exists(output_img_path)

        # Available vertical space calculation for 1-page fit
        # Total height = 841 - 70 = 771 pt
        header_height = 110.0 if has_output_image else 35.0
        available_code_height = 760.0 - header_height

        # Calculate exact line height (leading) and font size to guarantee 1-page fit
        leading = available_code_height / max(total_lines, 1)
        leading = min(leading, 10.0)
        leading = max(leading, 5.0)  # Dynamic scale for large files
        font_size = min(8.0, leading * 0.85)

        code_style = ParagraphStyle(
            f'CodeStyle_{file_idx}',
            fontName='Courier',
            fontSize=font_size,
            leading=leading,
            textColor=colors.HexColor("#1A1A1A"),
            backColor=colors.HexColor("#F8F9FA"),
            borderPadding=(3, 5, 3, 5),
            borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.5,
            borderRadius=3
        )

        # Build Header Block (Title + Filename + Live Output Thumbnail if available)
        header_text_p = Paragraph(f"<b>{file_idx:02d}. {title}</b><br/><font size=8 color='#007755'>File: <code>{filename}</code> | Total Lines: {total_lines}</font>", title_style)

        if has_output_image:
            # Create thumbnail image widget (Width ~ 160pt, Height ~ 95pt)
            try:
                img_widget = Image(output_img_path, width=150, height=95)
                header_table = Table([[header_text_p, img_widget]], colWidths=[370, 160])
                header_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(header_table)
            except Exception as e:
                story.append(header_text_p)
                print(f"[NOTICE] Error inserting image {output_img_path}: {e}")
        else:
            story.append(header_text_p)

        story.append(Spacer(1, 4))

        # Format code text with line numbers and indentation
        formatted_code_lines = []
        for line_num, line in enumerate(code_lines, start=1):
            line_str = line.rstrip("\n\r")
            indent_count = len(line_str) - len(line_str.lstrip(" "))
            indent_spaces = "&nbsp;" * indent_count
            trimmed_line = line_str.lstrip(" ")
            
            escaped_line = (trimmed_line.replace("&", "&amp;")
                                        .replace("<", "&lt;")
                                        .replace(">", "&gt;"))

            line_formatted = f"<font color='#888888'>{line_num:02d}&nbsp;&nbsp;</font>{indent_spaces}{escaped_line}"
            formatted_code_lines.append(line_formatted)

        code_text = "<br/>".join(formatted_code_lines)
        story.append(Paragraph(code_text, code_style))

        # PageBreak after each file so every project code + output image occupies 1 page
        if file_idx < len(FILES_WITH_METADATA):
            story.append(PageBreak())

    doc.build(story, canvasmaker=OnePageCodeCanvas)
    print(f"[SUCCESS] Complete Codebook PDF with Outputs created at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_codebook_pdf()
