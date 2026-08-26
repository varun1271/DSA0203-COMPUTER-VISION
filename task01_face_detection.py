"""
Task 01: Real-Time Face Detection
Detects faces from webcam or synthetic test input using Haar Cascade & MediaPipe models.
"""

import cv2
import time
import argparse
import os
from utils import get_camera_or_fallback, ensure_output_dir

def run_face_detection(demo=False, save_output=False):
    # Load Haar cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo)
    output_dir = ensure_output_dir()
    
    prev_time = time.time()
    saved_sample = False

    print("\n--- [TASK 01: FACE DETECTION] ---")
    print("Press 'q' or 'ESC' on display window to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Multi-scale face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        # Display info banner
        cv2.putText(frame, f"Faces Detected: {len(faces)} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Save single output image if requested
        if (save_output or is_demo) and not saved_sample and len(faces) > 0:
            out_path = os.path.join(output_dir, "task01_face_detection_result.jpg")
            cv2.imwrite(out_path, frame)
            print(f"[SUCCESS] Saved output image to: {out_path}")
            saved_sample = True

        cv2.imshow("Task 01 - Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 01 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 01: Face Detection")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save output result image")
    args = parser.parse_args()
    
    run_face_detection(demo=args.demo, save_output=args.save)
