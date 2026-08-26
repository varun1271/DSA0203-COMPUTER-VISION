"""
Task 08: Cyberpunk Matrix Feature Alignment & Homography
ORB / SIFT Keypoint Extraction, Lowe's Ratio Test, Homography Matrix Object Alignment Polygon & Telemetry.
"""

import cv2
import time
import argparse
import os
import numpy as np
from utils import create_synthetic_text_frame, ensure_output_dir, draw_tech_hud_grid

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
            if m.distance < 0.72 * n.distance:
                good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    # Compute RANSAC Homography for object localization
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
            
            # Offset x coordinates for side-by-side display
            dst[:, :, 0] += w
            cv2.polylines(matched_img, [np.int32(dst)], True, (0, 255, 255), 3)

            if mask is not None:
                inlier_count = int(np.sum(mask))

    return matched_img, len(good_matches), inlier_count

def run_feature_matching(demo=False, save_output=False, method="orb"):
    output_dir = ensure_output_dir()

    print(f"\n--- [TASK 08: CYBERPUNK MATRIX FEATURE ALIGNMENT ({method.upper()})] ---")

    img1 = create_synthetic_text_frame(450, 340, text="CYBER VISION 2026")
    h, w, _ = img1.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), 18, 0.92)
    img2 = cv2.warpAffine(img1, M, (w, h))

    matched_result, num_matches, inliers = perform_feature_matching(img1, img2, method=method)

    # Sleek Top Telemetry Bar
    cv2.rectangle(matched_result, (0, 0), (w * 2, 45), (15, 15, 20), -1)
    cv2.line(matched_result, (0, 45), (w * 2, 45), (0, 255, 200), 2)

    cv2.putText(matched_result, f"FEATURE ALIGNMENT [{method.upper()}] | MATCHES: {num_matches} | RANSAC INLIERS: {inliers}", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    draw_tech_hud_grid(matched_result)

    if save_output or demo:
        out_path = os.path.join(output_dir, "task08_feature_matching_result.jpg")
        cv2.imwrite(out_path, matched_result)
        print(f"[SUCCESS] Saved feature alignment result image to: {out_path}")

    try:
        cv2.imshow("Task 08 - Matrix Feature Alignment & Homography", matched_result)
        cv2.waitKey(2000 if demo else 0)
    except cv2.error:
        pass

    cv2.destroyAllWindows()
    print("[TASK 08 COMPLETED]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 08: Feature Alignment & Homography")
    parser.add_argument("--method", type=str, default="orb", choices=["orb", "sift"], help="Feature detector")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode")
    parser.add_argument("--save", action="store_true", help="Save result image")
    args = parser.parse_args()

    run_feature_matching(demo=args.demo, save_output=args.save, method=args.method)
