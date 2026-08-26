"""
Task 04: Real-Time Motion Alert & Security Intelligence System
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Processes live frames for motion detection, tracking, and security alerts in a while True loop.
"""

import cv2
import time
import os
import datetime
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def run_motion_alert(camera_idx=0, save_output=True, min_area=600):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    first_frame = None
    saved_sample = False
    prev_time = time.time()
    motion_event_count = 0
    motion_history = []
    heatmap = None
    window_name = "Task 04 - Real-Time Motion Security Intelligence"

    print("\n--- [TASK 04: REAL-TIME MOTION SECURITY INTELLIGENCE] ---")
    print("Capturing live video from webcam...")
    print("Press 'q' or 'ESC' on display window to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, c = frame.shape
        if heatmap is None or heatmap.shape[:2] != (h, w):
            heatmap = np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = blur
            continue

        # Frame Differencing
        frame_delta = cv2.absdiff(first_frame, blur)
        thresh = cv2.threshold(frame_delta, 22, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)

        # Accumulate motion heatmap
        heatmap = cv2.addWeighted(heatmap, 0.94, thresh.astype(np.float32) / 255.0, 0.06, 0)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        active_centroids = []
        annotated = frame.copy()

        # Render Motion Heatmap tint overlay
        heatmap_norm = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        annotated = cv2.addWeighted(annotated, 0.82, heatmap_color, 0.18, 0)

        total_moving_area = 0
        for c_contour in contours:
            area = cv2.contourArea(c_contour)
            if area < min_area:
                continue

            total_moving_area += area
            (x, y, bw, bh) = cv2.boundingRect(c_contour)
            cx, cy = x + bw // 2, y + bh // 2
            active_centroids.append((cx, cy))

            # Target Box & Reticle
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
            cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

            cv2.putText(annotated, f"MOTION [{area:.0f}px]", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            motion_detected = True

        if active_centroids:
            motion_history.append(active_centroids[0])
            if len(motion_history) > 30:
                motion_history.pop(0)

        for i in range(1, len(motion_history)):
            cv2.line(annotated, motion_history[i-1], motion_history[i], (0, 255, 255), 2)

        threat_level = min(100, int((total_moving_area / (w * h * 0.05)) * 100))
        cv2.rectangle(annotated, (0, 0), (w, 45), (15, 15, 20), -1)
        cv2.line(annotated, (0, 45), (w, 45), (0, 255, 200), 2)

        curr_time_str = datetime.datetime.now().strftime("%H:%M:%S")

        if motion_detected:
            motion_event_count += 1
            cv2.putText(annotated, f"LIVE MOTION ALERT [{curr_time_str}] | THREAT: {threat_level}%", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, f"SYSTEM MONITORING - ALL CLEAR [{curr_time_str}]", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        draw_tech_hud_grid(annotated)

        # Periodically update reference frame
        first_frame = cv2.addWeighted(first_frame, 0.95, blur, 0.05, 0)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if save_output and not saved_sample and motion_detected:
            out_path = os.path.join(output_dir, "task04_motion_alert_result.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"[SUCCESS] Saved motion alert snapshot to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[TASK 04 COMPLETED] Total Motion Events Detected: {motion_event_count}")

if __name__ == "__main__":
    run_motion_alert()
