from pathlib import Path
import csv
import json

import cv2
import numpy as np


PATTERN_SIZE = (7, 10)
SQUARE_SIZE_MM = 24.0
IMAGE_ROOT = Path("images")
OUTPUT_ROOT = Path("calibration_outputs")


def image_sort_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def object_points():
    cols, rows = PATTERN_SIZE
    points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * SQUARE_SIZE_MM
    return points


def detect_corners(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return gray.shape[::-1], None

    return gray.shape[::-1], corners.astype(np.float32)


def draw_reprojection(image_path, detected, projected, error_px, output_path):
    image = cv2.imread(str(image_path))
    projected = projected.reshape(-1, 2)
    detected = detected.reshape(-1, 2)

    for detected_point, projected_point in zip(detected, projected):
        dx, dy = detected_point
        px, py = projected_point
        dxy = (int(round(dx)), int(round(dy)))
        pxy = (int(round(px)), int(round(py)))
        cv2.line(image, dxy, pxy, (0, 255, 255), 2)
        cv2.circle(image, dxy, 5, (0, 255, 0), -1)
        cv2.circle(image, pxy, 5, (0, 0, 255), -1)

    cv2.putText(
        image,
        f"green: detected, red: reprojected, RMS error: {error_px:.3f}px",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"green: detected, red: reprojected, RMS error: {error_px:.3f}px",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def calibrate_set(set_dir):
    objp = object_points()
    objpoints = []
    imgpoints = []
    used_images = []
    image_size = None

    for image_path in sorted(set_dir.glob("*.jpg"), key=image_sort_key):
        size, corners = detect_corners(image_path)
        if corners is None:
            print(f"Warning: corners not found in {image_path}")
            continue
        image_size = size
        objpoints.append(objp)
        imgpoints.append(corners)
        used_images.append(image_path)

    if len(imgpoints) < 3:
        raise RuntimeError(f"{set_dir.name}: at least 3 valid images are required")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    per_image = []
    plot_dir = OUTPUT_ROOT / set_dir.name / "reprojection_plots"
    for image_path, obj, corners, rvec, tvec in zip(
        used_images, objpoints, imgpoints, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        residuals = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        distances = np.linalg.norm(residuals, axis=1)
        image_rms = float(np.sqrt(np.mean(distances**2)))
        image_mean = float(np.mean(distances))
        image_max = float(np.max(distances))

        plot_path = plot_dir / f"{image_path.stem}_reprojection.jpg"
        draw_reprojection(image_path, corners, projected, image_rms, plot_path)

        per_image.append(
            {
                "image": image_path.name,
                "rms_px": image_rms,
                "mean_px": image_mean,
                "max_px": image_max,
                "plot": str(plot_path),
            }
        )

    return {
        "set": set_dir.name,
        "num_images": len(used_images),
        "image_size": image_size,
        "opencv_rms_px": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.ravel(),
        "per_image": per_image,
    }


def save_results(results):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_ROOT / "calibration_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set",
                "num_images",
                "opencv_rms_px",
                "mean_image_rms_px",
                "max_image_rms_px",
                "fx",
                "fy",
                "cx",
                "cy",
                "k1",
                "k2",
                "p1",
                "p2",
                "k3",
            ]
        )
        for result in results:
            matrix = result["camera_matrix"]
            dist = result["dist_coeffs"]
            image_rms_values = [item["rms_px"] for item in result["per_image"]]
            writer.writerow(
                [
                    result["set"],
                    result["num_images"],
                    f"{result['opencv_rms_px']:.6f}",
                    f"{np.mean(image_rms_values):.6f}",
                    f"{np.max(image_rms_values):.6f}",
                    f"{matrix[0, 0]:.6f}",
                    f"{matrix[1, 1]:.6f}",
                    f"{matrix[0, 2]:.6f}",
                    f"{matrix[1, 2]:.6f}",
                    f"{dist[0]:.8f}",
                    f"{dist[1]:.8f}",
                    f"{dist[2]:.8f}",
                    f"{dist[3]:.8f}",
                    f"{dist[4]:.8f}",
                ]
            )

    detail_path = OUTPUT_ROOT / "calibration_details.json"
    json_ready = []
    for result in results:
        json_ready.append(
            {
                "set": result["set"],
                "num_images": result["num_images"],
                "image_size": result["image_size"],
                "opencv_rms_px": result["opencv_rms_px"],
                "camera_matrix": result["camera_matrix"].tolist(),
                "dist_coeffs": result["dist_coeffs"].tolist(),
                "per_image": result["per_image"],
            }
        )
    detail_path.write_text(
        json.dumps(json_ready, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary_path, detail_path


def main():
    set_dirs = sorted(path for path in IMAGE_ROOT.glob("Set_*") if path.is_dir())
    if not set_dirs:
        raise RuntimeError(f"No Set_* folders found under {IMAGE_ROOT}")

    results = [calibrate_set(set_dir) for set_dir in set_dirs]
    summary_path, detail_path = save_results(results)

    print(f"Pattern size: {PATTERN_SIZE[0]} x {PATTERN_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE_MM:.1f} mm")
    print(f"Summary: {summary_path}")
    print(f"Details: {detail_path}")
    print()

    for result in results:
        matrix = result["camera_matrix"]
        dist = result["dist_coeffs"]
        image_rms_values = [item["rms_px"] for item in result["per_image"]]
        print(result["set"])
        print(f"  images: {result['num_images']}")
        print(f"  OpenCV RMS: {result['opencv_rms_px']:.6f} px")
        print(f"  mean image RMS: {np.mean(image_rms_values):.6f} px")
        print(f"  max image RMS: {np.max(image_rms_values):.6f} px")
        print(
            "  K: "
            f"fx={matrix[0, 0]:.6f}, fy={matrix[1, 1]:.6f}, "
            f"cx={matrix[0, 2]:.6f}, cy={matrix[1, 2]:.6f}"
        )
        print(
            "  distortion: "
            f"k1={dist[0]:.8f}, k2={dist[1]:.8f}, "
            f"p1={dist[2]:.8f}, p2={dist[3]:.8f}, k3={dist[4]:.8f}"
        )
        print()


if __name__ == "__main__":
    main()
