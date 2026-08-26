"""
Task 05: Cyberpunk Neon Air Canvas / Glow Painting
Virtual air drawing with glowing neon laser particle brush FX, smooth interpolation, and glassmorphism HUD toolbar.
"""

import cv2
import time
import argparse
import os
import numpy as np
from utils import get_camera_or_fallback, ensure_output_dir, draw_tech_hud_grid

def apply_glow_effect(canvas):
    """Applies a smooth cyberpunk neon glow effect to canvas strokes."""
    glow = cv2.GaussianBlur(canvas, (21, 21), 0)
    return cv2.addWeighted(canvas, 1.0, glow, 0.8, 0)

def run_air_drawing(demo=False, save_output=False):
    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo)
    output_dir = ensure_output_dir()

    print("\n--- [TASK 05: CYBERPUNK NEON AIR CANVAS] ---")
    print("Glow Palette: Cyan Laser | Electric Green | Magenta | Yellow | Particle Eraser")
    print("Press 'c' to Clear Canvas, 'q' or 'ESC' to quit.\n")

    mp_hands = None
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            mp_hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            print("[INFO] MediaPipe High-Precision Hand Tracking active.")
    except Exception as e:
        print(f"[NOTICE] Hand Tracking fallback active ({e}).")

    canvas = None
    glow_canvas = None

    # Neon Color Palette (BGR)
    colors = [
        (255, 255, 0),   # Cyan Laser
        (0, 255, 120),   # Electric Green
        (255, 0, 200),   # Magenta
        (0, 200, 255),   # Gold Yellow
        (0, 0, 0)        # Eraser
    ]
    color_names = ["CYAN", "GREEN", "MAGENTA", "GOLD", "ERASER"]
    current_color_idx = 0
    brush_thickness = 8
    prev_point = None
    saved_sample = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Glassmorphism Top Control Toolbar
        toolbar_h = 60
        btn_w = w // len(colors)
        
        # Dark HUD Top Strip
        cv2.rectangle(frame, (0, 0), (w, toolbar_h), (20, 20, 25), -1)
        cv2.line(frame, (0, toolbar_h), (w, toolbar_h), (0, 255, 200), 2)

        for i, col in enumerate(colors):
            x1, y1 = i * btn_w, 0
            x2, y2 = (i + 1) * btn_w, toolbar_h
            is_selected = (i == current_color_idx)

            bg_col = (40, 40, 40) if i == len(colors)-1 else tuple([int(c*0.5) for c in col])
            cv2.rectangle(frame, (x1 + 4, 6), (x2 - 4, toolbar_h - 6), bg_col, -1)
            
            border_col = (0, 255, 255) if is_selected else (80, 80, 80)
            thickness = 3 if is_selected else 1
            cv2.rectangle(frame, (x1 + 4, 6), (x2 - 4, toolbar_h - 6), border_col, thickness)

            text_col = (255, 255, 255) if is_selected else (180, 180, 180)
            cv2.putText(frame, color_names[i], (x1 + 15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2 if is_selected else 1)

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

        # Demo Mode animation stroke
        if draw_point is None:
            t = time.time() * 2.5
            cx = int(w/2 + np.sin(t) * (w/3))
            cy = int(h/2 + np.cos(t * 1.4) * (h/4))
            draw_point = (cx, cy)

        # Check Palette Selection Interaction
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

        # Apply Neon Glow FX
        glow_canvas = apply_glow_effect(canvas)

        # Blend Glow Canvas with camera frame
        gray_c = cv2.cvtColor(glow_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_c, 5, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        combined = cv2.bitwise_and(frame, mask)
        combined = cv2.add(combined, glow_canvas)

        draw_tech_hud_grid(combined)

        cv2.putText(combined, f"ACTIVE BRUSH: {color_names[current_color_idx]} | GLOW FX: ON", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if (save_output or is_demo) and not saved_sample and np.sum(canvas) > 0:
            out_path = os.path.join(output_dir, "task05_drawing_result.jpg")
            cv2.imwrite(out_path, combined)
            print(f"[SUCCESS] Saved neon air canvas result image to: {out_path}")
            saved_sample = True

        try:
            cv2.imshow("Task 05 - Cyberpunk Neon Air Canvas", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                print("[INFO] Canvas Cleared.")
            elif key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 05 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 05: Cyberpunk Neon Air Canvas")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save result image")
    args = parser.parse_args()

    run_air_drawing(demo=args.demo, save_output=args.save)
