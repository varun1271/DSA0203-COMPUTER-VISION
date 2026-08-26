"""
Task 08: Real-Time Feature Detection & Matching (ORB/SIFT)
Captures real-time video input from default webcam using cv2.VideoCapture(0).
Matches keypoints & homography between target reference and live camera stream in a while True loop.
"""

import cv2
import time
import os
import numpy as np
from utils import open_webcam, ensure_output_dir, draw_tech_hud_grid

def perform_feature_matching(img1, img2, method="orb", max_matches=60):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if method.lower() == "sift" and hasattr(cv2, 'SIFT_create'):
        detector = cv2.SIFT_create(nfeatures=800)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        detector = cv2.ORB_create(nfeatures=800)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return np.hstack((img1, img2)), 0, 0

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0)
    )

    inlier_count = len(good_matches)
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            h, w, _ = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            dst[:, :, 0] += w
            cv2.polylines(matched_img, [np.int32(dst)], True, (0, 255, 255), 3)

            if mask is not None:
                inlier_count = int(np.sum(mask))

    return matched_img, len(good_matches), inlier_count

def run_feature_matching(method="orb", camera_idx=0, save_output=True):
    cap = open_webcam(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {camera_idx}.")
        return

    output_dir = ensure_output_dir()
    saved_sample = False
    ref_frame = None
    window_name = "Task 08 - Real-Time Feature Matching & Homography"

    print(f"\n--- [TASK 08: REAL-TIME FEATURE MATCHING ({method.upper()})] ---")
    print("Capturing live video from webcam...")
    print("Hold up an object to camera and press 'c' to capture reference target!")
    print("Press 'q' or 'ESC' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_resized = cv2.resize(frame, (450, 340))
        if ref_frame is None:
            ref_frame = frame_resized.copy()

        matched_result, num_matches, inliers = perform_feature_matching(ref_frame, frame_resized, method=method)
        h, w, _ = ref_frame.shape

        cv2.rectangle(matched_result, (0, 0), (w * 2, 45), (15, 15, 20), -1)
        cv2.line(matched_result, (0, 45), (w * 2, 45), (0, 255, 200), 2)
        cv2.putText(matched_result, f"REAL-TIME MATCHING [{method.upper()}] | MATCHES: {num_matches} | RANSAC: {inliers}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        draw_tech_hud_grid(matched_result)

        if save_output and not saved_sample and num_matches > 0:
            out_path = os.path.join(output_dir, "task08_feature_matching_result.jpg")
            cv2.imwrite(out_path, matched_result)
            print(f"[SUCCESS] Saved feature matching result image to: {out_path}")
            saved_sample = True

        cv2.imshow(window_name, matched_result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            ref_frame = frame_resized.copy()
            print("[INFO] Captured new reference image target from live camera feed.")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TASK 08 COMPLETED]")

if __name__ == "__main__":
    run_feature_matching()
