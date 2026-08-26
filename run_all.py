"""
CV MINI PROJECTS MASTER RUNNER
Run any of the 12 Computer Vision mini-projects interactively or in demo mode.
"""

import sys
import os
import subprocess
import argparse

TASKS = [
    ("task01_face_detection.py", "Task 01: Real-Time Face Detection"),
    ("task02_expression.py", "Task 02: Facial Expression & Emotion Detection"),
    ("task03_ocr.py", "Task 03: Optical Character Recognition (OCR)"),
    ("task04_motion_alert.py", "Task 04: Motion Alert & Security System"),
    ("task05_drawing.py", "Task 05: Virtual Air Canvas / Air Drawing"),
    ("task06_edge_detection.py", "Task 06: Interactive Real-Time Edge Detection"),
    ("task07_tracking.py", "Task 07: Object Tracking (CSRT/KCF/MIL)"),
    ("task08_feature_matching.py", "Task 08: Feature Detection & Matching (ORB/SIFT)"),
    ("task09_license_plate.py", "Task 09: Automatic License Plate Recognition (ALPR)"),
    ("task10_background_subtraction.py", "Task 10: Background Subtraction & Virtual BG"),
    ("task11_face_counting.py", "Task 11: Live Face Counter"),
    ("task12_image_to_video.py", "Task 12: Image to Video Converter & Recorder")
]

def print_menu():
    print("=" * 65)
    print("      COMPUTER VISION REAL-TIME MINI PROJECTS SUITE (2026)")
    print("=" * 65)
    for idx, (script, desc) in enumerate(TASKS, start=1):
        print(f"  [{idx:02d}] {desc}")
    print("  [A ] Run ALL Projects Sequentially (Real-Time Live Camera)")
    print("  [D ] Run ALL Projects in Synthetic Demo Mode")
    print("  [0 ] Exit")
    print("=" * 65)

def run_task(script_name, demo=False):
    cmd = [sys.executable, script_name, "--save"]
    if demo:
        cmd.append("--demo")
    print(f"\n[RUNNING] Executing: {' '.join(cmd)} (Real-Time Camera Mode: {not demo})")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="CV Mini Projects Master Runner")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--all", action="store_true", help="Run all 12 projects sequentially in real-time camera mode")
    parser.add_argument("--task", type=int, choices=range(1, 13), help="Specify task number (1-12)")
    args = parser.parse_args()

    if args.all:
        is_demo = args.demo
        mode_str = "Synthetic Demo Mode" if is_demo else "Real-Time Live Camera Mode"
        print(f"[SUITE] Running all 12 Computer Vision projects ({mode_str})...")
        for script, desc in TASKS:
            run_task(script, demo=is_demo)
        print("\n[COMPLETE] All 12 projects executed successfully!")
        return

    if args.task:
        script, desc = TASKS[args.task - 1]
        run_task(script, demo=args.demo)
        return

    while True:
        print_menu()
        choice = input("Select an option (1-12, A, D, 0): ").strip().upper()
        if choice == "0":
            print("Exiting CV Mini Projects Suite. Goodbye!")
            break
        elif choice == "A":
            print("\n[SUITE] Launching all 12 projects sequentially in Real-Time Live Camera Mode...")
            for script, desc in TASKS:
                run_task(script, demo=False)
            print("\n[COMPLETE] All 12 projects executed successfully!")
        elif choice == "D":
            print("\n[SUITE] Launching all 12 projects sequentially in Synthetic Demo Mode...")
            for script, desc in TASKS:
                run_task(script, demo=True)
            print("\n[COMPLETE] All 12 projects executed successfully!")
        elif choice.isdigit() and 1 <= int(choice) <= 12:
            script, desc = TASKS[int(choice) - 1]
            demo_mode = input("Run in Synthetic Demo mode instead of Live Camera? [y/N]: ").strip().lower() == 'y'
            run_task(script, demo=demo_mode)
        else:
            print("[ERROR] Invalid choice. Please select 1-12, A, D, or 0.")

if __name__ == "__main__":
    main()

