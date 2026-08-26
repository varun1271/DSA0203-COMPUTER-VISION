"""
Task 11: Real-Time Live Face Counter & Crowd Analytics
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Counts live faces in webcam feed and displays real-time statistics in a while True loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def run_face_counter(camera_idx=0, save_output=True, max_threshold=5):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    saved_sample = False
    prev_time = time.time()
    max_detected_so_far = 0
    window_name = "Task 11 - Real-Time Live Face Counter"

    print("\n--- [TASK 11: REAL-TIME LIVE FACE COUNTER] ---")
    print("Capturing live video from webcam...")
    print("Press 'q' or 'ESC' on display window to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h_f, w_f, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        face_count = len(faces)
        if face_count > max_detected_so_far:
            max_detected_so_far = face_count

        annotated = frame.copy()

        # Spatial Grid Lines
        cv2.line(annotated, (w_f // 2, 0), (w_f // 2, h_f), (50, 50, 60), 1)
        cv2.line(annotated, (0, h_f // 2), (w_f, h_f // 2), (50, 50, 60), 1)

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.rectangle(annotated, (x, y - 25), (x + 100, y), (20, 20, 25), -1)
            cv2.putText(annotated, f"PERSON #{idx:02d}", (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

        banner_color = (15, 15, 20) if face_count <= max_threshold else (0, 0, 150)
        cv2.rectangle(annotated, (0, 0), (w_f, 45), banner_color, -1)
        cv2.line(annotated, (0, 45), (w_f, 45), (0, 255, 200), 2)

        status_str = f"LIVE FACE COUNTER | ACTIVE: {face_count} | PEAK: {max_detected_so_far}"
        cv2.putText(annotated, status_str, (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        draw_tech_hud_grid(annotated)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(annotated, f"FPS: {fps:.1f}", (15, h_f - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if save_output and not saved_sample:
            out_path = os.path.join(output_dir, "task11_face_counting_result.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"[SUCCESS] Saved crowd counter snapshot to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[TASK 11 COMPLETED] Max People Count Detected: {max_detected_so_far}")

if __name__ == "__main__":
    run_face_counter()
