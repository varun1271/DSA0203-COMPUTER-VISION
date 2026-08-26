import os
import cv2
import numpy as np
import time

def ensure_output_dir(dir_name="output"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def draw_tech_hud_grid(frame):
    """Draws a subtle sci-fi tech grid overlay on a frame."""
    h, w, _ = frame.shape
    # Draw dark translucent HUD border
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 200), 1)
    # Corner target brackets
    length = 25
    cv2.line(frame, (15, 15), (15 + length, 15), (0, 255, 255), 2)
    cv2.line(frame, (15, 15), (15, 15 + length), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, 15), (w - 15 - length, 15), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, 15), (w - 15, 15 + length), (0, 255, 255), 2)
    cv2.line(frame, (15, h - 15), (15 + length, h - 15), (0, 255, 255), 2)
    cv2.line(frame, (15, h - 15), (15, h - 15 - length), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, h - 15), (w - 15 - length, h - 15), (0, 255, 255), 2)
    cv2.line(frame, (w - 15, h - 15), (w - 15, h - 15 - length), (0, 255, 255), 2)

def create_synthetic_face_frame(width=640, height=480, frame_count=0):
    """Generates an HD photorealistic synthetic camera feed with animated subject."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Futuristic dark background gradient
    for y in range(height):
        v = int(25 + 35 * (y / height))
        frame[y, :] = (v + 5, v, v + 15)

    # Grid pattern background
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (35, 45, 55), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (35, 45, 55), 1)
        
    # Moving position for human subject simulation
    shift_x = int(60 * np.sin(frame_count * 0.04))
    shift_y = int(25 * np.cos(frame_count * 0.04))
    center_x = width // 2 + shift_x
    center_y = height // 2 + shift_y

    # Shoulders & Body
    cv2.ellipse(frame, (center_x, center_y + 190), (160, 120), 0, 0, 360, (70, 60, 50), -1)
    cv2.ellipse(frame, (center_x, center_y + 190), (160, 120), 0, 0, 360, (120, 100, 80), 2)

    # Face Structure (natural skin tones)
    cv2.ellipse(frame, (center_x, center_y), (85, 115), 0, 0, 360, (150, 175, 215), -1)
    cv2.ellipse(frame, (center_x, center_y), (85, 115), 0, 0, 360, (100, 120, 160), 2)

    # Hair
    cv2.ellipse(frame, (center_x, center_y - 45), (90, 70), 0, 180, 360, (30, 25, 20), -1)

    # Eyes & Eyebrows
    cv2.circle(frame, (center_x - 30, center_y - 20), 14, (245, 245, 245), -1)
    cv2.circle(frame, (center_x + 30, center_y - 20), 14, (245, 245, 245), -1)
    cv2.circle(frame, (center_x - 30, center_y - 20), 6, (80, 40, 20), -1)
    cv2.circle(frame, (center_x + 30, center_y - 20), 6, (80, 40, 20), -1)
    cv2.circle(frame, (center_x - 28, center_y - 22), 2, (255, 255, 255), -1)
    cv2.circle(frame, (center_x + 32, center_y - 22), 2, (255, 255, 255), -1)

    cv2.line(frame, (center_x - 45, center_y - 40), (center_x - 15, center_y - 38), (40, 30, 25), 3)
    cv2.line(frame, (center_x + 15, center_y - 38), (center_x + 45, center_y - 40), (40, 30, 25), 3)

    # Nose & Mouth
    cv2.line(frame, (center_x, center_y - 5), (center_x - 8, center_y + 22), (110, 130, 170), 2)
    cv2.line(frame, (center_x - 8, center_y + 22), (center_x + 8, center_y + 22), (110, 130, 170), 2)

    smile_open = int(8 * np.abs(np.sin(frame_count * 0.08)))
    cv2.ellipse(frame, (center_x, center_y + 50), (32, 12 + smile_open), 0, 0, 180, (60, 60, 190), -1 if smile_open > 4 else 2)

    # Add realistic digital camera noise / scanline
    scan_y = (frame_count * 4) % height
    cv2.line(frame, (0, scan_y), (width, scan_y), (0, 255, 255), 1)

    return frame

def create_synthetic_text_frame(width=640, height=480, text="CYBERPUNK VISION 2026"):
    """Generates an HD image with distinct text for OCR & Feature Matching testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (20, 20, 25)

    # Glowing dark header block
    cv2.rectangle(frame, (40, 80), (width - 40, 180), (40, 30, 20), -1)
    cv2.rectangle(frame, (40, 80), (width - 40, 180), (0, 255, 255), 2)
    cv2.putText(frame, text, (60, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # Technical data blocks
    cv2.rectangle(frame, (40, 210), (300, 420), (35, 35, 45), -1)
    cv2.rectangle(frame, (40, 210), (300, 420), (100, 100, 150), 1)
    cv2.putText(frame, "SYSTEM ID: 987654", (55, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
    cv2.putText(frame, "STATUS: ONLINE", (55, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, "LAT: 37.7749 N", (55, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, "LON: 122.4194 W", (55, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Tactical diagram
    cv2.circle(frame, (470, 310), 80, (0, 255, 200), 2)
    cv2.circle(frame, (470, 310), 40, (0, 150, 255), 1)
    cv2.line(frame, (470, 210), (470, 410), (0, 255, 200), 1)
    cv2.line(frame, (370, 310), (570, 310), (0, 255, 200), 1)
    
    return frame

def create_synthetic_license_plate_frame(width=640, height=480, plate_str="CA 87-XY 901"):
    """Generates an HD realistic vehicle image with license plate for ALPR testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (40, 45, 50)
    
    # Asphalt Road
    cv2.rectangle(frame, (0, 320), (width, height), (25, 25, 25), -1)
    cv2.line(frame, (0, 400), (width, 400), (255, 255, 255), 2)

    # Vehicle Body (Sleek sports car back)
    cv2.rectangle(frame, (80, 140), (560, 340), (120, 20, 20), -1)
    cv2.rectangle(frame, (80, 140), (560, 340), (60, 10, 10), 3)

    # Rear Windshield
    cv2.polygon = np.array([[140, 140], [500, 140], [450, 80], [190, 80]], np.int32)
    cv2.fillPoly(frame, [cv2.polygon], (20, 20, 25))
    cv2.polylines(frame, [cv2.polygon], True, (100, 100, 100), 2)

    # Tail Lights (Glowing red LEDs)
    cv2.rectangle(frame, (100, 170), (220, 210), (0, 0, 255), -1)
    cv2.rectangle(frame, (420, 170), (540, 210), (0, 0, 255), -1)
    cv2.rectangle(frame, (110, 180), (210, 200), (200, 200, 255), -1)
    cv2.rectangle(frame, (430, 180), (530, 200), (200, 200, 255), -1)

    # Rear Bumper & License Plate Recess
    cv2.rectangle(frame, (200, 240), (440, 310), (15, 15, 15), -1)
    
    # License Plate (White metal plate with dark border)
    plate_x1, plate_y1, plate_x2, plate_y2 = 220, 252, 420, 298
    cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (240, 245, 245), -1)
    cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (20, 20, 20), 3)
    # Blue State Strip
    cv2.rectangle(frame, (plate_x1 + 3, plate_y1 + 3), (plate_x1 + 25, plate_y2 - 3), (180, 50, 0), -1)
    cv2.putText(frame, "USA", (plate_x1 + 5, plate_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.putText(frame, plate_str, (plate_x1 + 32, plate_y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    return frame

class SyntheticCamera:
    """Fallback camera class when physical webcam is not available."""
    def __init__(self, mode="face", max_frames=250):
        self.mode = mode
        self.frame_count = 0
        self.max_frames = max_frames
        self.width = 640
        self.height = 480

    def isOpened(self):
        return self.frame_count < self.max_frames

    def read(self):
        if self.frame_count >= self.max_frames:
            return False, None
        
        self.frame_count += 1
        if self.mode == "face":
            frame = create_synthetic_face_frame(self.width, self.height, self.frame_count)
        elif self.mode == "text":
            frame = create_synthetic_text_frame(self.width, self.height)
        elif self.mode == "plate":
            frame = create_synthetic_license_plate_frame(self.width, self.height)
        else:
            frame = create_synthetic_face_frame(self.width, self.height, self.frame_count)
            
        time.sleep(0.02)
        return True, frame

    def release(self):
        pass

def get_camera_or_fallback(camera_idx=0, mode="face", force_demo=False, max_demo_frames=250):
    """Attempts to open physical webcam silently (using DirectShow on Windows), falling back to SyntheticCamera."""
    if force_demo:
        print(f"[INFO] Running in Demo mode with Synthetic Camera ({mode}).")
        return SyntheticCamera(mode=mode, max_frames=max_demo_frames), True

    # Use cv2.CAP_DSHOW on Windows for fast, silent webcam initialization without MSMF warnings
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("[INFO] Physical webcam initialized successfully.")
            return cap, False
        cap.release()

    print(f"[WARNING] Physical camera (index {camera_idx}) unavailable. Falling back to High-Tech Synthetic Camera.")
    return SyntheticCamera(mode=mode, max_frames=max_demo_frames), True
