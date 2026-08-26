"""
Task 07: Real-Time Object Tracking & Target Lock
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Tracks user-selected object in real-time video stream using CSRT/KCF/MIL algorithms in a while True loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def create_tracker(tracker_type="csrt"):
    tracker_type = tracker_type.lower()
    if tracker_type == "csrt":
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
    elif tracker_type == "kcf":
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
            
    try:
        return cv2.TrackerMIL_create()
    except Exception:
        return None

class CentroidTracker:
    def __init__(self):
        self.bbox = None

    def init(self, frame, bbox):
        self.bbox = [int(v) for v in bbox]

    def update(self, frame):
        if self.bbox is None:
            return False, None
        x, y, w, h = self.bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        search_margin = 40
        x1, y1 = max(0, x - search_margin), max(0, y - search_margin)
        x2, y2 = min(frame.shape[1], x + w + search_margin), min(frame.shape[0], y + h + search_margin)
        roi = blur[y1:y2, x1:x2]
        
        if roi.size > 0:
            _, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                self.bbox = (cx - w//2, cy - h//2, w, h)
                
        return True, tuple(self.bbox)

def draw_target_lock_reticle(frame, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    corner_len = min(w, h) // 4
    cv2.line(frame, (x, y), (x + corner_len, y), (0, 255, 0), 2)
    cv2.line(frame, (x, y), (x, y + corner_len), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), (0, 255, 0), 2)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), (0, 255, 0), 2)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), (0, 255, 0), 2)

    radius = int(min(w, h) * 0.45)
    cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 1)
    cv2.line(frame, (cx - radius - 10, cy), (cx - 5, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx + 5, cy), (cx + radius + 10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - radius - 10), (cx, cy - 5), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy + 5), (cx, cy + radius + 10), (0, 255, 0), 1)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

def run_object_tracking(tracker_type="csrt", camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    tracker = create_tracker(tracker_type)
    if tracker is None:
        tracker = CentroidTracker()

    tracking_initialized = False
    bbox = None
    saved_sample = False
    prev_time = time.time()
    trajectory_points = []
    window_name = "Task 07 - Real-Time Object Tracking"

    print(f"\n--- [TASK 07: REAL-TIME OBJECT TRACKING ({tracker_type.upper()})] ---")
    print("Capturing live video from webcam...")
    print("Select an object ROI on the webcam feed using your mouse, then press SPACE or ENTER!")
    print("Press 's' to re-select ROI, 'q' or 'ESC' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_h, frame_w, _ = frame.shape

        if not tracking_initialized:
            try:
                roi = cv2.selectROI(window_name, frame, False, True)
                if roi[2] > 0 and roi[3] > 0:
                    bbox = roi
                    tracker.init(frame, bbox)
                    tracking_initialized = True
                else:
                    cv2.putText(frame, "Select ROI and press SPACE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow(window_name, frame)
                    if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]:
                        break
                    continue
            except cv2.error:
                break

        success, bbox = tracker.update(frame)
        annotated = frame.copy()

        cv2.rectangle(annotated, (0, 0), (frame_w, 45), (15, 15, 20), -1)
        cv2.line(annotated, (0, 45), (frame_w, 45), (0, 255, 200), 2)

        if success and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = x + w // 2, y + h // 2
            trajectory_points.append((cx, cy))
            if len(trajectory_points) > 40:
                trajectory_points.pop(0)

            draw_target_lock_reticle(annotated, x, y, w, h)

            for i in range(1, len(trajectory_points)):
                cv2.line(annotated, trajectory_points[i-1], trajectory_points[i], (0, 255, 0), 2)

            cv2.putText(annotated, f"TARGET LOCKED [{tracker_type.upper()}] | POS: ({cx},{cy})", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "TARGET LOST - Press 's' to Select New Object", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        draw_tech_hud_grid(annotated)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(annotated, f"FPS: {fps:.1f}", (15, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if save_output and not saved_sample and success:
            out_path = os.path.join(output_dir, "task07_tracking_result.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"[SUCCESS] Saved tracking result snapshot to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            tracking_initialized = False
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 07 COMPLETED]")

if __name__ == "__main__":
    run_object_tracking()
