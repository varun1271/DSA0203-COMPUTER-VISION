"""
Task 12: Cinematic Video Studio & Cyberpunk Stream Recorder
Encodes video streams into MP4 with SMPTE timecode stamps, audio waveform visualizers, and studio HUDs.
"""

import cv2
import time
import argparse
import os
import glob
import numpy as np
from utils import get_camera_or_fallback, ensure_output_dir, draw_tech_hud_grid

def draw_audio_waveform_bar(frame, frame_count):
    """Draws a dynamic audio frequency waveform visualizer at bottom of frame."""
    h, w, _ = frame.shape
    wave_h = 40
    wave_y = h - wave_h - 10
    
    cv2.rectangle(frame, (10, wave_y), (w - 10, wave_y + wave_h), (15, 15, 20), -1)
    cv2.rectangle(frame, (10, wave_y), (w - 10, wave_y + wave_h), (0, 255, 200), 1)

    num_bars = 40
    bar_w = (w - 30) // num_bars
    for i in range(num_bars):
        # Simulated audio frequency spectrum oscillation
        val = np.abs(np.sin(frame_count * 0.15 + i * 0.3)) * (wave_h - 6)
        bh = int(val)
        bx = 15 + i * bar_w
        by = wave_y + wave_h - 3 - bh
        
        col = (0, 255, 120) if bh < wave_h * 0.6 else (0, 200, 255)
        cv2.rectangle(frame, (bx, by), (bx + bar_w - 2, wave_y + wave_h - 3), col, -1)

def record_stream_to_video(output_video_path, demo=False, max_frames=150, fps=30):
    cap, is_demo = get_camera_or_fallback(0, mode="face", force_demo=demo, max_demo_frames=max_frames)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 640, 480
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"\n--- [TASK 12: CINEMATIC VIDEO STUDIO & RECORDER] ---")
    print(f"Output Video Path: {output_video_path}")
    print("Press 'q' or 'ESC' on display window to stop recording.\n")

    recorded_count = 0
    prev_time = time.time()
    start_timestamp = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized = cv2.resize(frame, (w, h))
        recorded_count += 1

        annotated = frame_resized.copy()

        # Top Studio HUD Header
        cv2.rectangle(annotated, (0, 0), (w, 50), (15, 15, 20), -1)
        cv2.line(annotated, (0, 50), (w, 50), (0, 255, 200), 2)

        # Pulsing Red REC Dot
        pulse = (recorded_count % 30) < 15
        rec_color = (0, 0, 255) if pulse else (0, 0, 100)
        cv2.circle(annotated, (25, 25), 10, rec_color, -1)
        cv2.putText(annotated, "REC", (42, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # SMPTE Timecode Counter [HH:MM:SS:FF]
        elapsed = time.time() - start_timestamp
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        frames = int((elapsed - int(elapsed)) * fps)
        timecode_str = f"00:{mins:02d}:{secs:02d}:{frames:02d}"
        cv2.putText(annotated, f"TC: {timecode_str}", (w - 180, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw Audio Waveform Spectrum
        draw_audio_waveform_bar(annotated, recorded_count)

        draw_tech_hud_grid(annotated)

        # Write frame to video file
        writer.write(annotated)

        curr_time = time.time()
        fps_curr = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(annotated, f"ENCODING: MP4V | FPS: {fps_curr:.1f} | FRAMES: {recorded_count}", (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        try:
            cv2.imshow("Task 12 - Cinematic Video Studio & Recorder", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        except cv2.error:
            pass

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Recorded {recorded_count} cinematic frames into: {output_video_path}")
    print("[TASK 12 COMPLETED]")

def run_image_to_video(demo=False, save_output=False):
    output_dir = ensure_output_dir()
    output_video_path = os.path.join(output_dir, "task12_output_video.mp4")
    record_stream_to_video(output_video_path, demo=demo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 12: Cinematic Video Studio")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save video file")
    args = parser.parse_args()

    run_image_to_video(demo=args.demo, save_output=args.save)
