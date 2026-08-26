import os
import cv2
import numpy as np

def ensure_output_dir(dir_name="output"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def draw_tech_hud_grid(frame):
    """Draws a subtle sci-fi tech grid overlay on a frame."""
    h, w, _ = frame.shape
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 200), 1)
    length = 25
    cv2.line(frame, (15, 15), (15 + length, 15), (0, 255, 255), 2)
    cv2.line(frame, (15, 15), (15, 15 + length), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, 15), (w - 15 - length, 15), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, 15), (w - 15, 15 + length), (0, 255, 255), 2)
    cv2.line(frame, (15, h - 15), (15 + length, h - 15), (0, 255, 255), 2)
    cv2.line(frame, (15, h - 15), (15, h - 15 - length), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, h - 15), (w - 15 - length, h - 15), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, h - 15), (w - 15, h - 15 - length), (0, 255, 255), 2)

def open_webcam(camera_idx=0):
    """Opens physical default webcam using cv2.VideoCapture(0)."""
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)
    return cap
