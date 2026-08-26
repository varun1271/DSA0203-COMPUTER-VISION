"""
Task 08: Cyberpunk Matrix Feature Alignment & Homography
ORB / SIFT Keypoint Extraction, Lowe's Ratio Test, Homography Matrix Object Alignment Polygon & Telemetry.
"""

import cv2
import time
import argparse
import os
import numpy as np
from utils import get_camera_or_fallback, create_synthetic_text_frame, ensure_output_dir, draw_tech_hud_grid

def run_feature_matching(demo=False, save_output=False, method="orb"):
    cap, is_demo = get_camera_or_fallback(0, mode="text", force_demo=demo)
    output_dir = ensure_output_dir()

    print(f"\n--- [TASK 08: CYBERPUNK MATRIX FEATURE ALIGNMENT ({method.upper()})] ---")
    print("Press 'c' to Capture New Target Frame from Camera, 'q' or 'ESC' to quit.\n")

    saved_sample = False
    ref_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized = cv2.resize(frame, (450, 340))
        
        if ref_frame is None:
            if is_demo:
                ref_frame = create_synthetic_text_frame(450, 340, text="CYBER VISION 2026")
            else:
                ref_frame = frame_resized.copy()

        matched_result, num_matches, inliers = perform_feature_matching(ref_frame, frame_resized, method=method)
        h, w, _ = ref_frame.shape

        # Sleek Top Telemetry Bar
        cv2.rectangle(matched_result, (0, 0), (w * 2, 45), (15, 15, 20), -1)
        cv2.line(matched_result, (0, 45), (w * 2, 45), (0, 255, 200), 2)

        cv2.putText(matched_result, f"FEATURE ALIGNMENT [{method.upper()}] | MATCHES: {num_matches} | RANSAC INLIERS: {inliers}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        draw_tech_hud_grid(matched_result)

        if (save_output or is_demo) and not saved_sample and num_matches > 0:
            out_path = os.path.join(output_dir, "task08_feature_matching_result.jpg")
            cv2.imwrite(out_path, matched_result)
            print(f"[SUCCESS] Saved feature alignment result image to: {out_path}")
            saved_sample = True

        try:
            cv2.imshow("Task 08 - Matrix Feature Alignment & Homography", matched_result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                ref_frame = frame_resized.copy()
                print("[INFO] Captured new reference image target from live feed.")
            elif key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 08 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 08: Feature Alignment & Homography")
    parser.add_argument("--method", type=str, default="orb", choices=["orb", "sift"], help="Feature detector")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save result image")
    args = parser.parse_args()

    run_feature_matching(demo=args.demo, save_output=args.save, method=args.method)

