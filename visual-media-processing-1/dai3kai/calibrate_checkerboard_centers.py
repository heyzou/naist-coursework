import csv
import glob
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (3024, 4032)
SQUARE_SIZE_MM = 24.0
MODEL_ROWS = 8
ROW_COUNTS = [6, 5, 6, 5, 6, 5, 6, 5]


def model_points(variant=0):
    points = []
    for row, count in enumerate(ROW_COUNTS):
        xs = range(0, 11, 2) if count == 6 else range(1, 10, 2)
        for x in xs:
            points.append([float(x), float(row), 0.0])
    base = np.asarray(points, np.float32)
    xy = base[:, :2]
    transforms = [
        xy,
        np.column_stack([10.0 - xy[:, 0], xy[:, 1]]),
        np.column_stack([xy[:, 0], 7.0 - xy[:, 1]]),
        np.column_stack([10.0 - xy[:, 0], 7.0 - xy[:, 1]]),
        np.column_stack([xy[:, 1], xy[:, 0]]),
        np.column_stack([7.0 - xy[:, 1], xy[:, 0]]),
        np.column_stack([xy[:, 1], 10.0 - xy[:, 0]]),
        np.column_stack([7.0 - xy[:, 1], 10.0 - xy[:, 0]]),
    ]
    out = np.zeros_like(base)
    out[:, :2] = transforms[variant]
    return out * SQUARE_SIZE_MM


def detect_centers(path, threshold):
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        ratio = w / h
        rectangularity = area / (w * h)
        if not (500 < area < 80000 and 0.35 < ratio < 2.8 and rectangularity > 0.45):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centers.append([2.0 * moments["m10"] / moments["m00"], 2.0 * moments["m01"] / moments["m00"]])
    return np.asarray(centers, np.float32)


def order_center_candidates(points):
    if len(points) != 44:
        return []

    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = [vt, vt[::-1]]
    candidates = []

    for axis in axes:
        for sign_u in (1, -1):
            for sign_v in (1, -1):
                basis = np.vstack([sign_u * axis[0], sign_v * axis[1]]).T
                projected = centered @ basis
                order = np.argsort(projected[:, 1])
                sorted_points = points[order]
                sorted_projected = projected[order]
                rows = []
                start = 0
                score = 0.0
                ok = True
                for count in ROW_COUNTS:
                    row_points = sorted_points[start : start + count]
                    row_projected = sorted_projected[start : start + count]
                    if len(row_points) != count:
                        ok = False
                        break
                    horizontal_order = np.argsort(row_projected[:, 0])
                    row_points = row_points[horizontal_order]
                    row_projected = row_projected[horizontal_order]
                    rows.append(row_points)
                    score += float(np.var(row_projected[:, 1]))
                    start += count
                if not ok:
                    continue
                ordered = np.vstack(rows).astype(np.float32)
                candidates.append((score, ordered))

    candidates.sort(key=lambda item: item[0])
    return [ordered for _, ordered in candidates[:4]]


def calibrate(label, image_specs):
    image_points = []
    used = []

    for filename, threshold in image_specs:
        centers = detect_centers(Path(filename), threshold)
        candidates = order_center_candidates(centers)
        if not candidates:
            continue
        image_points.append(candidates)
        used.append(filename)

    best = None
    from itertools import product

    for choices in product(range(4), repeat=len(image_points)):
        selected_image_points = [candidates[index] for candidates, index in zip(image_points, choices)]
        object_points = [model_points(0) for _ in selected_image_points]
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, selected_image_points, IMAGE_SIZE, None, None
        )
        if best is None or rms < best[0]:
            best = (rms, camera_matrix, dist_coeffs, rvecs, tvecs, object_points, selected_image_points, choices)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs, object_points, selected_image_points, choices = best

    per_view = []
    for objp, imgp, rvec, tvec in zip(object_points, selected_image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected.reshape(-1, 2), cv2.NORM_L2) / len(projected)
        per_view.append(float(error))

    return {
        "label": label,
        "used_images": used,
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.ravel(),
        "mean_error": float(np.mean(per_view)),
        "per_view_errors": per_view,
        "variant": "-".join(str(choice) for choice in choices),
    }


def main():
    sets = {
        "set_a_frontal": [
            ("calibration_checkerboard_04.jpg", 100),
            ("calibration_checkerboard_09.jpg", 70),
            ("calibration_checkerboard_16.jpg", 100),
        ],
        "set_b_oblique": [
            ("calibration_checkerboard_02.jpg", 80),
            ("calibration_checkerboard_07.jpg", 90),
            ("calibration_checkerboard_11.jpg", 60),
            ("calibration_checkerboard_13.jpg", 60),
        ],
        "set_c_mixed": [
            ("calibration_checkerboard_13.jpg", 60),
            ("calibration_checkerboard_14.jpg", 140),
            ("calibration_checkerboard_15.jpg", 130),
        ],
    }

    results = [calibrate(label, specs) for label, specs in sets.items()]

    with open("calibration_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set",
                "num_images",
                "rms_px",
                "mean_reprojection_error_px",
                "fx",
                "fy",
                "cx",
                "cy",
                "k1",
                "k2",
                "p1",
                "p2",
                "k3",
                "images",
                "per_view_errors_px",
                "pattern_variant",
            ]
        )
        for result in results:
            matrix = result["camera_matrix"]
            dist = result["dist_coeffs"]
            writer.writerow(
                [
                    result["label"],
                    len(result["used_images"]),
                    f"{result['rms']:.6f}",
                    f"{result['mean_error']:.6f}",
                    f"{matrix[0, 0]:.6f}",
                    f"{matrix[1, 1]:.6f}",
                    f"{matrix[0, 2]:.6f}",
                    f"{matrix[1, 2]:.6f}",
                    f"{dist[0]:.8f}",
                    f"{dist[1]:.8f}",
                    f"{dist[2]:.8f}",
                    f"{dist[3]:.8f}",
                    f"{dist[4]:.8f}",
                    ";".join(result["used_images"]),
                    ";".join(f"{value:.6f}" for value in result["per_view_errors"]),
                    result["variant"],
                ]
            )

    for result in results:
        matrix = result["camera_matrix"]
        print(result["label"])
        print(" images:", ", ".join(result["used_images"]))
        print(" rms:", f"{result['rms']:.6f}", "mean:", f"{result['mean_error']:.6f}")
        print(matrix)
        print(result["dist_coeffs"])


if __name__ == "__main__":
    main()
