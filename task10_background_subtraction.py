"""
Task 10: Cyberpunk Holographic Stealth & Virtual Background System
MOG2/KNN Background Subtraction, Predator Active Camouflage Ripple, Synthwave Hologram Portals & Feathered Blending.
"""

import cv2
import time
import argparse
import os
import numpy as np
from utils import get_camera_or_fallback, ensure_output_dir, draw_tech_hud_grid

def create_synthwave_portal_bg(h, w):
    """Generates a high-tech Synthwave neon grid portal background."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        # Deep violet gradient
        v = int(20 + 80 * (y / h))
        bg[y, :] = (int(v * 1.2), int(v * 0.3), int(v * 0.8))

    # Horizon line
    horizon = int(h * 0.55)
    cv2.line(bg, (0, horizon), (w, horizon), (255, 0, 200), 2)

    # Perspective Grid lines
    for x in range(-w, w * 2, 40):
        cv2.line(bg, (w // 2, horizon), (x, h), (255, 0, 150), 1)

    for y_g in range(horizon, h, 20):
        cv2.line(bg, (0, y_g), (w, y_g), (0, 255, 255), 1)

    # Glowing Cyber Sun
    cv2.circle(bg, (w // 2, horizon - 40), 70, (0, 200, 255), -1)
    cv2.circle(bg, (w // 2, horizon - 40), 70, (255, 0, 200), 3)

    return bg

def run_background_subtraction(demo=False, save_output=False, method="mog2"):
    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo)
    output_dir = ensure_output_dir()

    print(f"\n--- [TASK 10: CYBERPUNK HOLOGRAPHIC STEALTH ({method.upper()})] ---")
    print("Keyboard Controls:")
    print("  '1' -> Synthwave Hologram Portal Background")
    print("  '2' -> Predator Active Camouflage (Optical Stealth)")
    print("  '3' -> Raw Foreground Mask Telemetry")
    print("Press 'q' or 'ESC' on display window to quit.\n")

    if method.lower() == "knn":
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    else:
        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    mode = 1
    saved_sample = False
    prev_time = time.time()
    synthwave_bg = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape
        if synthwave_bg is None or synthwave_bg.shape[:2] != (h, w):
            synthwave_bg = create_synthwave_portal_bg(h, w)

        fg_mask = subtractor.apply(frame)
        _, fg_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Feathered edge mask for smooth compositing
        fg_blur = cv2.GaussianBlur(fg_clean, (11, 11), 0)
        alpha = (fg_blur.astype(np.float32) / 255.0)[..., np.newaxis]

        if mode == 1:
            # Synthwave Hologram Composite
            display_frame = (frame.astype(np.float32) * alpha + synthwave_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            cv2.putText(display_frame, "MODE: SYNTHWAVE HOLOGRAM PORTAL", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif mode == 2:
            # Predator Active Camouflage Ripple Effect
            t = time.time() * 5
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + 10 * np.sin(map_y / 10.0 + t)).astype(np.float32)
            map_y = (map_y + 10 * np.cos(map_x / 10.0 + t)).astype(np.float32)
            camo_bg = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            
            display_frame = (frame.astype(np.float32) * alpha + camo_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            cv2.putText(display_frame, "MODE: PREDATOR OPTICAL CAMOUFLAGE", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            display_frame = cv2.cvtColor(fg_clean, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_frame, "MODE: FOREGROUND MASK TELEMETRY", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Sleek Top Banner Bar
        cv2.rectangle(display_frame, (0, 0), (w, 45), (15, 15, 20), -1)
        cv2.line(display_frame, (0, 45), (w, 45), (0, 255, 200), 2)
        cv2.putText(display_frame, "CYBERPUNK HOLOGRAPHIC STEALTH SYSTEM", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        draw_tech_hud_grid(display_frame)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if (save_output or is_demo) and not saved_sample:
            out_path = os.path.join(output_dir, "task10_background_subtraction_result.jpg")
            cv2.imwrite(out_path, display_frame)
            print(f"[SUCCESS] Saved holographic stealth result image to: {out_path}")
            saved_sample = True

        try:
            cv2.imshow("Task 10 - Cyberpunk Holographic Stealth", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                mode = 1
            elif key == ord('2'):
                mode = 2
            elif key == ord('3'):
                mode = 3
            elif key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 10 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 10: Holographic Stealth System")
    parser.add_argument("--method", type=str, default="mog2", choices=["mog2", "knn"], help="Subtractor method")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save result image")
    args = parser.parse_args()

    run_background_subtraction(demo=args.demo, save_output=args.save, method=args.method)
