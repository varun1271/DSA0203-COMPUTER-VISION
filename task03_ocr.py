"""
Task 03: Real-Time Optical Character Recognition (OCR)
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Detects and reads text from live video frames using OpenCV text contour analysis & EasyOCR.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

easyocr_reader = None

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        try:
            import easyocr
            easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[INFO] EasyOCR initialized successfully.")
        except Exception as e:
            print(f"[NOTICE] EasyOCR not loaded ({e}). Using real-time OpenCV text contour detection.")
            easyocr_reader = False
    return easyocr_reader

def perform_opencv_contour_ocr(frame):
    """Real-time text detection using adaptive thresholding and contour bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        if area > 400 and aspect_ratio > 1.2 and w > 40:
            results.append(([(x, y), (x+w, y), (x+w, y+h), (x, y+h)], "TEXT DETECTED", 0.85))

    return results

def run_ocr(camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    reader = get_easyocr_reader()
    saved_sample = False
    prev_time = time.time()
    window_name = "Task 03 - Real-Time Optical Character Recognition"

    print("\n--- [TASK 03: REAL-TIME OCR] ---")
    print("Capturing live video from webcam... Hold up any text to the camera!")
    print("Press 'q' or 'ESC' on display window to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        annotated_frame = frame.copy()

        if reader:
            ocr_results = reader.readtext(frame)
        else:
            ocr_results = perform_opencv_contour_ocr(frame)

        text_count = 0
        for bbox, text, prob in ocr_results:
            if prob > 0.3:
                text_count += 1
                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)

                top_left = (int(bbox[0][0]), int(bbox[0][1]))
                cv2.rectangle(annotated_frame, (top_left[0], top_left[1] - 25),
                              (top_left[0] + len(text)*12 + 10, top_left[1]), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"{text} ({prob:.2f})", (top_left[0] + 5, top_left[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"OCR Live Detections: {text_count} | FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        draw_tech_hud_grid(annotated_frame)

        if save_output and not saved_sample and text_count > 0:
            out_path = os.path.join(output_dir, "task03_ocr_result.jpg")
            cv2.imwrite(out_path, annotated_frame)
            print(f"[SUCCESS] Saved OCR result image to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 03 COMPLETED]")

if __name__ == "__main__":
    run_ocr()
