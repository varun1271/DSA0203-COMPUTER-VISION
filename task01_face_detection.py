"""
Task 01: Real-Time Face Detection
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Processes live frames and detects faces using OpenCV Haar Cascades in a real-time while loop.
"""

import cv2
import time
import os
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def run_face_detection(camera_idx=0, save_output=True):
    # Load Haar cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Capture real-time video input from default webcam
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    saved_sample = False
    prev_time = time.time()

    print("\n--- [TASK 01: REAL-TIME FACE DETECTION] ---")
    print("Capturing live video from webcam...")
    print("Press 'q' or 'ESC' on display window to exit.\n")

    window_name = "Task 01 - Real-Time Face Detection"

    # Real-time processing loop
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read live frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Real-time multi-scale face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw bounding boxes around detected live faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate live FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        # Display info banner & tech HUD
        cv2.putText(frame, f"Live Faces Detected: {len(faces)} | FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        draw_tech_hud_grid(frame)

        # Save single output snapshot if requested
        if save_output and not saved_sample and len(faces) > 0:
            out_path = os.path.join(output_dir, "task01_face_detection_result.jpg")
            cv2.imwrite(out_path, frame)
            print(f"[SUCCESS] Saved live detection result snapshot to: {out_path}")
            saved_sample = True

        # Display real-time output
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 01 COMPLETED]")

if __name__ == "__main__":
    run_face_detection()
