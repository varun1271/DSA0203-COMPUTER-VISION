"""
Task 04: Tactical Motion Alert & Security Intelligence System
Sci-Fi Tactical Motion Detection with Trajectory Tracking, Intrusion Zones, Threat Gauges, and Motion Heatmaps.
"""

import cv2
import time
import argparse
import os
import datetime
import numpy as np
from utils import get_camera_or_fallback, ensure_output_dir, draw_tech_hud_grid

def run_motion_alert(demo=False, save_output=False, min_area=600):
    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo)
    output_dir = ensure_output_dir()

    print("\n--- [TASK 04: TACTICAL MOTION & SECURITY INTELLIGENCE] ---")
    print("Features: Trajectory Tracking | Motion Heatmaps | Threat Gauge | Intrusion Zone Grid")
    print("Press 'q' or 'ESC' on display window to quit.\n")

    first_frame = None
    saved_sample = False
    prev_time = time.time()
    motion_event_count = 0
    motion_history = []  # Stores centroid motion trajectory history
    heatmap = None

    while cap.isOpened():
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

        # Draw Perimeter Security Line (Red Zone Boundary)
        zone_y = int(h * 0.4)
        cv2.line(annotated, (0, zone_y), (w, zone_y), (0, 0, 255), 1)
        cv2.putText(annotated, "SECURED PERIMETER BOUNDARY", (w - 240, zone_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        total_moving_area = 0

        for c_contour in contours:
            area = cv2.contourArea(c_contour)
            if area < min_area:
                continue

            total_moving_area += area
            (x, y, bw, bh) = cv2.boundingRect(c_contour)
            cx, cy = x + bw // 2, y + bh // 2
            active_centroids.append((cx, cy))

            # Tactical Target Box & Corner Crosshairs
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            # Target Lock Reticle
            cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
            cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

            cv2.putText(annotated, f"TARGET [{area:.0f}px]", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            motion_detected = True

        # Update centroid trajectory trails
        if active_centroids:
            motion_history.append(active_centroids[0])
            if len(motion_history) > 30:
                motion_history.pop(0)

        # Render Trajectory Motion Trail (Cyberpunk cyan path)
        for i in range(1, len(motion_history)):
            pt1 = motion_history[i - 1]
            pt2 = motion_history[i]
            thickness = int(np.sqrt(30 / (len(motion_history) - i + 1)) * 2)
            cv2.line(annotated, pt1, pt2, (255, 255, 0), max(1, thickness))

        # Dynamic Threat Level Gauge (0-100%)
        threat_level = min(100, int((total_moving_area / (w * h * 0.05)) * 100))
        
        # Sleek Top HUD Header Dashboard
        cv2.rectangle(annotated, (0, 0), (w, 55), (15, 15, 20), -1)
        cv2.line(annotated, (0, 55), (w, 55), (0, 255, 200), 2)

        curr_time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if motion_detected:
            motion_event_count += 1
            cv2.putText(annotated, f"CRITICAL ALERT: MOTION DETECTED [{curr_time_str}]", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, f"SYSTEM MONITORING - ALL CLEAR [{curr_time_str}]", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Threat Level Bar
        gauge_x, gauge_y, gauge_w, gauge_h = w - 210, 15, 180, 15
        cv2.rectangle(annotated, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (50, 50, 50), -1)
        bar_fill = int((threat_level / 100.0) * gauge_w)
        bar_color = (0, 255, 0) if threat_level < 30 else ((0, 255, 255) if threat_level < 70 else (0, 0, 255))
        cv2.rectangle(annotated, (gauge_x, gauge_y), (gauge_x + bar_fill, gauge_y + gauge_h), bar_color, -1)
        cv2.rectangle(annotated, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (255, 255, 255), 1)
        cv2.putText(annotated, f"THREAT: {threat_level}%", (gauge_x - 100, gauge_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        draw_tech_hud_grid(annotated)

        # Periodically update reference frame
        first_frame = cv2.addWeighted(first_frame, 0.95, blur, 0.05, 0)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(annotated, f"FPS: {fps:.1f} | TRAJECTORY POINTS: {len(motion_history)}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if (save_output or is_demo) and not saved_sample and motion_detected:
            out_path = os.path.join(output_dir, "task04_motion_alert_result.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"[SUCCESS] Saved tactical motion alert image to: {out_path}")
            saved_sample = True

        try:
            cv2.imshow("Task 04 - Tactical Motion Intelligence", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print(f"[TASK 04 COMPLETED] Total Motion Events Detected: {motion_event_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 04: Tactical Motion Intelligence")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save alert result image")
    args = parser.parse_args()

    run_motion_alert(demo=args.demo, save_output=args.save)
