"""
Task 05: Real-Time Virtual Air Canvas / Air Drawing
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Draws on live camera feed using MediaPipe hand index finger tracking or color pointer in a while True loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def apply_glow_effect(canvas):
    """Applies a smooth neon glow effect to canvas strokes."""
    glow = cv2.GaussianBlur(canvas, (21, 21), 0)
    return cv2.addWeighted(canvas, 1.0, glow, 0.8, 0)

def run_air_drawing(camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    mp_hands = None
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            mp_hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            print("[INFO] MediaPipe Hand Tracking active.")
    except Exception as e:
        print(f"[NOTICE] Hand Tracking unavailable ({e}).")

    canvas = None
    colors = [(255, 255, 0), (0, 255, 120), (255, 0, 200), (0, 200, 255), (0, 0, 0)]
    color_names = ["CYAN", "GREEN", "MAGENTA", "GOLD", "ERASER"]
    current_color_idx = 0
    brush_thickness = 8
    prev_point = None
    saved_sample = False
    window_name = "Task 05 - Real-Time Virtual Air Canvas"

    print("\n--- [TASK 05: REAL-TIME VIRTUAL AIR CANVAS] ---")
    print("Capturing live video from webcam...")
    print("Point your index finger in front of camera to draw!")
    print("Press 'c' to Clear Canvas, 'q' or 'ESC' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Top Palette Control Toolbar
        toolbar_h = 60
        btn_w = w // len(colors)
        cv2.rectangle(frame, (0, 0), (w, toolbar_h), (20, 20, 25), -1)
        cv2.line(frame, (0, toolbar_h), (w, toolbar_h), (0, 255, 200), 2)

        for i, col in enumerate(colors):
            x1, y1 = i * btn_w, 0
            x2, y2 = (i + 1) * btn_w, toolbar_h
            is_selected = (i == current_color_idx)

            bg_col = (40, 40, 40) if i == len(colors)-1 else tuple([int(c*0.5) for c in col])
            cv2.rectangle(frame, (x1 + 4, 6), (x2 - 4, toolbar_h - 6), bg_col, -1)
            
            border_col = (0, 255, 255) if is_selected else (80, 80, 80)
            cv2.rectangle(frame, (x1 + 4, 6), (x2 - 4, toolbar_h - 6), border_col, 3 if is_selected else 1)
            cv2.putText(frame, color_names[i], (x1 + 15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2 if is_selected else 1)

        draw_point = None
        if mp_hands:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    draw_point = (cx, cy)
                    cv2.circle(frame, draw_point, 12, colors[current_color_idx], -1)
                    cv2.circle(frame, draw_point, 16, (255, 255, 255), 2)

        if draw_point is not None:
            if draw_point[1] < toolbar_h:
                selected_idx = draw_point[0] // btn_w
                if 0 <= selected_idx < len(colors):
                    current_color_idx = selected_idx
                prev_point = None
            else:
                if prev_point is not None:
                    thickness = 30 if current_color_idx == 4 else brush_thickness
                    cv2.line(canvas, prev_point, draw_point, colors[current_color_idx], thickness)
                prev_point = draw_point
        else:
            prev_point = None

        glow_canvas = apply_glow_effect(canvas)
        gray_c = cv2.cvtColor(glow_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_c, 5, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        combined = cv2.bitwise_and(frame, mask)
        combined = cv2.add(combined, glow_canvas)
        draw_tech_hud_grid(combined)

        cv2.putText(combined, f"BRUSH: {color_names[current_color_idx]}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if save_output and not saved_sample and np.sum(canvas) > 0:
            out_path = os.path.join(output_dir, "task05_drawing_result.jpg")
            cv2.imwrite(out_path, combined)
            print(f"[SUCCESS] Saved drawing result to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            print("[INFO] Canvas Cleared.")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 05 COMPLETED]")

if __name__ == "__main__":
    run_air_drawing()
