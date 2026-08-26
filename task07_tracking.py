"""
Task 07: Tactical Drone Target Lock & Trajectory Tracker
Military/Sci-Fi Object Tracking HUD with Target Lock Reticle, Velocity Vectors, Trajectory Pathing, and Radar Sweeps.
"""

import cv2
import time
import argparse
import os
import numpy as np
from utils import get_camera_or_fallback, ensure_output_dir, draw_tech_hud_grid

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

def draw_target_lock_reticle(frame, x, y, w, h, frame_count=0):
    cx, cy = x + w // 2, y + h // 2
    
    # Outer lock-on brackets
    corner_len = min(w, h) // 4
    cv2.line(frame, (x, y), (x + corner_len, y), (0, 255, 0), 2)
    cv2.line(frame, (x, y), (x, y + corner_len), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), (0, 255, 0), 2)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), (0, 255, 0), 2)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), (0, 255, 0), 2)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), (0, 255, 0), 2)

    # Rotating target lock circle
    radius = int(min(w, h) * 0.45)
    cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 1)
    
    # Crosshair lines
    cv2.line(frame, (cx - radius - 10, cy), (cx - 5, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx + 5, cy), (cx + radius + 10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - radius - 10), (cx, cy - 5), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy + 5), (cx, cy + radius + 10), (0, 255, 0), 1)

    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

def run_object_tracking(tracker_type="csrt", demo=False, save_output=False):
    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo)
    output_dir = ensure_output_dir()

    print(f"\n--- [TASK 07: TACTICAL DRONE TARGET LOCK ({tracker_type.upper()})] ---")
    print("Press 's' to Select ROI manually, 'q' or 'ESC' to quit.\n")

    tracker = create_tracker(tracker_type)
    if tracker is None:
        tracker = CentroidTracker()

    tracking_initialized = False
    bbox = None
    saved_sample = False
    prev_time = time.time()
    trajectory_points = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        frame_h, frame_w, _ = frame.shape

        if not tracking_initialized:
            if is_demo:
                bbox = (frame_w // 2 - 75, frame_h // 2 - 90, 150, 180)
            else:
                try:
                    roi = cv2.selectROI("Task 07 - Tactical Lock ROI", frame, False, True)
                    cv2.destroyWindow("Task 07 - Tactical Lock ROI")
                    bbox = roi if (roi[2] > 0 and roi[3] > 0) else (100, 100, 150, 150)
                except cv2.error:
                    bbox = (100, 100, 150, 150)

            tracker.init(frame, bbox)
            tracking_initialized = True

        success, bbox = tracker.update(frame)

        annotated = frame.copy()

        # Top HUD Banner
        cv2.rectangle(annotated, (0, 0), (frame_w, 45), (15, 15, 20), -1)
        cv2.line(annotated, (0, 45), (frame_w, 45), (0, 255, 200), 2)

        if success and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = x + w // 2, y + h // 2
            trajectory_points.append((cx, cy))
            if len(trajectory_points) > 40:
                trajectory_points.pop(0)

            draw_target_lock_reticle(annotated, x, y, w, h, frame_count)

            # Draw Velocity Vector Arrow
            if len(trajectory_points) >= 5:
                dx = trajectory_points[-1][0] - trajectory_points[-5][0]
                dy = trajectory_points[-1][1] - trajectory_points[-5][1]
                velocity = np.sqrt(dx**2 + dy**2)
                cv2.arrowedLine(annotated, (cx, cy), (cx + dx * 2, cy + dy * 2), (0, 255, 255), 2, tipLength=0.3)
                cv2.putText(annotated, f"VEL: {velocity:.1f} px/s", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Trajectory Trail
            for i in range(1, len(trajectory_points)):
                cv2.line(annotated, trajectory_points[i-1], trajectory_points[i], (0, 255, 0), 2)

            cv2.putText(annotated, f"TARGET LOCKED [{tracker_type.upper()}] | POS: ({cx},{cy})", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "TARGET LOST! SEARCHING RADAR SWEEP...", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Radar Sweep Animation
            radar_angle = (frame_count * 5) % 360
            rad = np.radians(radar_angle)
            rx, ry = frame_w // 2, frame_h // 2
            r_len = 100
            ex, ey = int(rx + r_len * np.cos(rad)), int(ry + r_len * np.sin(rad))
            cv2.circle(annotated, (rx, ry), r_len, (0, 0, 255), 1)
            cv2.line(annotated, (rx, ry), (ex, ey), (0, 0, 255), 2)

        draw_tech_hud_grid(annotated)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(annotated, f"FPS: {fps:.1f}", (15, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if (save_output or is_demo) and not saved_sample and success:
            out_path = os.path.join(output_dir, "task07_tracking_result.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"[SUCCESS] Saved tactical tracking result image to: {out_path}")
            saved_sample = True

        try:
            cv2.imshow("Task 07 - Tactical Drone Target Lock", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                tracking_initialized = False
            elif key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 07 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 07: Tactical Drone Target Lock")
    parser.add_argument("--tracker", type=str, default="csrt", choices=["csrt", "kcf", "mil"], help="Tracker algorithm")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save result image")
    args = parser.parse_args()

    run_object_tracking(tracker_type=args.tracker, demo=args.demo, save_output=args.save)
