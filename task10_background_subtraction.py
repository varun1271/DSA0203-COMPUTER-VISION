"""
Task 10: Real-Time Background Subtraction & Virtual Background
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Applies MOG2/KNN background subtraction & virtual background blending in a real-time while loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def create_synthwave_portal_bg(h, w):
    """Generates a Synthwave grid background for live camera replacement."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        v = int(20 + 80 * (y / h))
        bg[y, :] = (int(v * 1.2), int(v * 0.3), int(v * 0.8))

    horizon = int(h * 0.55)
    cv2.line(bg, (0, horizon), (w, horizon), (255, 0, 200), 2)
    for x in range(-w, w * 2, 40):
        cv2.line(bg, (w // 2, horizon), (x, h), (255, 0, 150), 1)
    for y_g in range(horizon, h, 20):
        cv2.line(bg, (0, y_g), (w, y_g), (0, 255, 255), 1)
    cv2.circle(bg, (w // 2, horizon - 40), 70, (0, 200, 255), -1)
    return bg

def run_background_subtraction(method="mog2", camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    if method.lower() == "knn":
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    else:
        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    mode = 1
    saved_sample = False
    prev_time = time.time()
    synthwave_bg = None
    window_name = "Task 10 - Real-Time Background Subtraction"

    print(f"\n--- [TASK 10: REAL-TIME BACKGROUND SUBTRACTION ({method.upper()})] ---")
    print("Capturing live video from webcam...")
    print("Keyboard Controls: '1' -> Virtual Background | '2' -> Optical Stealth | '3' -> Foreground Mask | 'q'/'ESC' -> Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape
        if synthwave_bg is None or synthwave_bg.shape[:2] != (h, w):
            synthwave_bg = create_synthwave_portal_bg(h, w)

        fg_mask = subtractor.apply(frame)
        _, fg_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        fg_blur = cv2.GaussianBlur(fg_clean, (11, 11), 0)
        alpha = (fg_blur.astype(np.float32) / 255.0)[..., np.newaxis]

        if mode == 1:
            display_frame = (frame.astype(np.float32) * alpha + synthwave_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        elif mode == 2:
            t = time.time() * 5
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + 10 * np.sin(map_y / 10.0 + t)).astype(np.float32)
            map_y = (map_y + 10 * np.cos(map_x / 10.0 + t)).astype(np.float32)
            camo_bg = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            display_frame = (frame.astype(np.float32) * alpha + camo_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        else:
            display_frame = cv2.cvtColor(fg_clean, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(display_frame, (0, 0), (w, 45), (15, 15, 20), -1)
        cv2.line(display_frame, (0, 45), (w, 45), (0, 255, 200), 2)
        cv2.putText(display_frame, "REAL-TIME BACKGROUND SUBTRACTION & VIRTUAL BG", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        draw_tech_hud_grid(display_frame)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if save_output and not saved_sample:
            out_path = os.path.join(output_dir, "task10_background_subtraction_result.jpg")
            cv2.imwrite(out_path, display_frame)
            print(f"[SUCCESS] Saved background subtraction snapshot to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 10 COMPLETED]")

if __name__ == "__main__":
    run_background_subtraction()
