"""
Task 09: Real-Time Automatic License Plate Recognition (ALPR)
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Scans live camera stream for rectangular plate contours and performs thresholding in a while True loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def detect_license_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    plate_contour = None
    plate_crop = None
    plate_bbox = None

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            if 1.8 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                plate_contour = approx
                plate_bbox = (x, y, w, h)
                plate_crop = frame[y:y+h, x:x+w]
                break

    annotated = frame.copy()
    h_f, w_f, _ = frame.shape

    if plate_bbox is not None:
        x, y, w, h = plate_bbox
        cv2.drawContours(annotated, [plate_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cv2.putText(annotated, "LICENSE PLATE DETECTED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if plate_crop is not None:
            try:
                crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, crop_thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_thresh_bgr = cv2.cvtColor(crop_thresh, cv2.COLOR_GRAY2BGR)

                res_w, res_h = 160, 45
                resized_thresh = cv2.resize(crop_thresh_bgr, (res_w, res_h))
                annotated[h_f - res_h - 20:h_f - 20, 20:20 + res_w] = resized_thresh
                cv2.rectangle(annotated, (20, h_f - res_h - 20), (20 + res_w, h_f - 20), (0, 255, 255), 1)
                cv2.putText(annotated, "PLATE SEGMENTATION", (20, h_f - res_h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            except Exception:
                pass

    return annotated, plate_bbox

def run_license_plate_rec(camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    saved_sample = False
    prev_time = time.time()
    window_name = "Task 09 - Real-Time ALPR License Plate Scanner"

    print("\n--- [TASK 09: REAL-TIME ALPR LICENSE PLATE SCANNER] ---")
    print("Capturing live video from webcam...")
    print("Hold up any card or vehicle plate to camera!")
    print("Press 'q' or 'ESC' on display window to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        processed, bbox = detect_license_plate(frame)
        h, w, _ = frame.shape

        cv2.rectangle(processed, (0, 0), (w, 45), (15, 15, 20), -1)
        cv2.line(processed, (0, 45), (w, 45), (0, 255, 200), 2)
        cv2.putText(processed, "REAL-TIME ALPR LICENSE PLATE SCANNER", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        draw_tech_hud_grid(processed)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(processed, f"FPS: {fps:.1f}", (w - 100, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if save_output and not saved_sample and bbox is not None:
            out_path = os.path.join(output_dir, "task09_license_plate_result.jpg")
            cv2.imwrite(out_path, processed)
            print(f"[SUCCESS] Saved ALPR result snapshot to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, processed)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 09 COMPLETED]")

if __name__ == "__main__":
    run_license_plate_rec()
