"""
Task 02: Facial Expression & Emotion Detection
Detects facial landmarks and classifies expressions (Happy, Surprise, Neutral).
"""

import cv2
import time
import argparse
import os
import numpy as np
from utils import get_camera_or_fallback, ensure_output_dir

def detect_expression(frame, face_cascade, smile_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect smile within face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        # Detect eyes within upper face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray[0:int(h*0.6), :], scaleFactor=1.1, minNeighbors=5)
        
        # Expression Classification Heuristics
        expression = "Neutral"
        color = (255, 255, 0)
        
        if len(smiles) > 0:
            expression = "Happy / Smiling"
            color = (0, 255, 0)
        elif len(eyes) >= 2:
            # Check mouth aspect ratio / open mouth heuristic if no smile
            mouth_roi = roi_gray[int(h*0.65):h, int(w*0.2):int(w*0.8)]
            if mouth_roi.size > 0:
                _, thresh = cv2.threshold(mouth_roi, 60, 255, cv2.THRESH_BINARY_INV)
                dark_pixels = cv2.countNonZero(thresh)
                ratio = dark_pixels / (mouth_roi.shape[0] * mouth_roi.shape[1] + 1e-5)
                if ratio > 0.4:
                    expression = "Surprised / Open Mouth"
                    color = (0, 165, 255)

        # Draw Face Bounding Box & Label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"Emotion: {expression}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Draw Eye boxes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)

        # Draw Smile boxes
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
            
    return frame, len(faces)

def run_expression_detection(demo=False, save_output=False):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo)
    output_dir = ensure_output_dir()

    prev_time = time.time()
    saved_sample = False

    print("\n--- [TASK 02: FACIAL EXPRESSION DETECTION] ---")
    print("Press 'q' or 'ESC' on display window to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        processed_frame, num_faces = detect_expression(frame, face_cascade, smile_cascade, eye_cascade)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(processed_frame, f"FPS: {fps:.1f} | Active Faces: {num_faces}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if (save_output or is_demo) and not saved_sample and num_faces > 0:
            out_path = os.path.join(output_dir, "task02_expression_result.jpg")
            cv2.imwrite(out_path, processed_frame)
            print(f"[SUCCESS] Saved output image to: {out_path}")
            saved_sample = True

        try:
            cv2.imshow("Task 02 - Expression Detection", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 02 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 02: Facial Expression Recognition")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save result image")
    args = parser.parse_args()

    run_expression_detection(demo=args.demo, save_output=args.save)
