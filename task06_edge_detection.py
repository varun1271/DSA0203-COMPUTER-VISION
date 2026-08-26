"""
Task 06: Real-Time Interactive Edge Detection & Scanner
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Processes live frames for Canny/Sobel/Laplacian edge filtering in a real-time while loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def nothing(x):
    pass

def apply_cyberpunk_edge(frame, method="neon_canny", low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "neon_canny":
        edges = cv2.Canny(blur, low_thresh, high_thresh)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        _, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.where(edges > 0, 255, 0).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif method == "thermal_sobel":
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        mag_norm = np.clip(mag / (mag.max() + 1e-5) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)

    elif method == "laplacian_matrix":
        lap = cv2.Laplacian(blur, cv2.CV_64F)
        lap_abs = cv2.convertScaleAbs(lap)
        return cv2.applyColorMap(lap_abs, cv2.COLORMAP_OCEAN)

    else:
        edges = cv2.Canny(blur, low_thresh, high_thresh)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def run_edge_detection(camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    window_name = "Task 06 - Real-Time Edge Scanner"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Low Thresh", window_name, 50, 255, nothing)
        cv2.createTrackbar("High Thresh", window_name, 150, 255, nothing)
    except cv2.error:
        pass

    current_mode = "neon_canny"
    saved_sample = False
    prev_time = time.time()

    print("\n--- [TASK 06: REAL-TIME EDGE SCANNER] ---")
    print("Capturing live video from webcam...")
    print("Keyboard Controls: '1' -> Neon Canny | '2' -> Sobel | '3' -> Laplacian | 'q'/'ESC' -> Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        low_val, high_val = 50, 150
        try:
            low_val = cv2.getTrackbarPos("Low Thresh", window_name)
            high_val = cv2.getTrackbarPos("High Thresh", window_name)
        except cv2.error:
            pass

        edge_map = apply_cyberpunk_edge(frame, method=current_mode, low_thresh=low_val, high_thresh=high_val)
        h, w, _ = frame.shape
        combined = np.hstack((frame, edge_map))

        cv2.rectangle(combined, (0, 0), (w * 2, 45), (15, 15, 20), -1)
        cv2.line(combined, (0, 45), (w * 2, 45), (0, 255, 200), 2)
        cv2.putText(combined, "RAW OPTICAL FEED", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(combined, f"EDGE SCANNER MODE: {current_mode.upper()}", (w + 15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(combined, f"FPS: {fps:.1f} | L-TH: {low_val} H-TH: {high_val}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        draw_tech_hud_grid(combined)

        if save_output and not saved_sample:
            out_path = os.path.join(output_dir, "task06_edge_detection_result.jpg")
            cv2.imwrite(out_path, combined)
            print(f"[SUCCESS] Saved edge scan result to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_mode = "neon_canny"
        elif key == ord('2'):
            current_mode = "thermal_sobel"
        elif key == ord('3'):
            current_mode = "laplacian_matrix"
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 06 COMPLETED]")

if __name__ == "__main__":
    run_edge_detection()
