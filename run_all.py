"""
CV MINI PROJECTS MASTER RUNNER
Executes any of the 12 Real-Time Computer Vision projects with live webcam input via cv2.VideoCapture(0).
"""

import sys
import subprocess
import argparse

TASKS = [
    ("task01_face_detection.py", "Task 01: Real-Time Face Detection"),
    ("task02_expression.py", "Task 02: Facial Expression & Emotion Detection"),
    ("task03_ocr.py", "Task 03: Optical Character Recognition (OCR)"),
    ("task04_motion_alert.py", "Task 04: Real-Time Motion Alert & Security System"),
    ("task05_drawing.py", "Task 05: Virtual Air Canvas / Air Drawing"),
    ("task06_edge_detection.py", "Task 06: Interactive Real-Time Edge Detection"),
    ("task07_tracking.py", "Task 07: Object Tracking (CSRT/KCF/MIL)"),
    ("task08_feature_matching.py", "Task 08: Feature Detection & Matching (ORB/SIFT)"),
    ("task09_license_plate.py", "Task 09: Automatic License Plate Recognition (ALPR)"),
    ("task10_background_subtraction.py", "Task 10: Background Subtraction & Virtual BG"),
    ("task11_face_counting.py", "Task 11: Live Face Counter"),
    ("task12_image_to_video.py", "Task 12: Real-Time Stream Recorder")
]

def print_menu():
    print("=" * 65)
    print("      COMPUTER VISION REAL-TIME WEBCAM SUITE (cv2.VideoCapture)")
    print("=" * 65)
    for idx, (script, desc) in enumerate(TASKS, start=1):
        print(f"  [{idx:02d}] {desc}")
    print("  [A ] Run ALL 12 Projects Sequentially (Live Webcam)")
    print("  [0 ] Exit")
    print("=" * 65)

def run_task(script_name):
    cmd = [sys.executable, script_name]
    print(f"\n[RUNNING] Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="CV Real-Time Webcam Master Runner")
    parser.add_argument("--all", action="store_true", help="Run all 12 projects sequentially with live webcam")
    parser.add_argument("--task", type=int, choices=range(1, 13), help="Specify task number (1-12)")
    args = parser.parse_args()

    if args.all:
        print("[SUITE] Launching all 12 projects with live webcam feed...")
        for script, desc in TASKS:
            run_task(script)
        print("\n[COMPLETE] All 12 projects executed successfully!")
        return

    if args.task:
        script, desc = TASKS[args.task - 1]
        run_task(script)
        return

    while True:
        print_menu()
        choice = input("Select an option (1-12, A, 0): ").strip().upper()
        if choice == "0":
            print("Exiting Computer Vision Suite. Goodbye!")
            break
        elif choice == "A":
            print("\n[SUITE] Launching all 12 projects sequentially with live webcam...")
            for script, desc in TASKS:
                run_task(script)
            print("\n[COMPLETE] All 12 projects executed successfully!")
        elif choice.isdigit() and 1 <= int(choice) <= 12:
            script, desc = TASKS[int(choice) - 1]
            run_task(script)
        else:
            print("[ERROR] Invalid choice. Please select 1-12, A, or 0.")

if __name__ == "__main__":
    main()
