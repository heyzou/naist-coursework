from pathlib import Path

import cv2
import numpy as np


IMAGE_ROOT = Path("images")
OUTPUT_ROOT = Path("calibration_outputs/dataset_overviews")
THUMB_W = 240
THUMB_H = 320
LABEL_H = 36
MARGIN = 16
COLS = 5


def image_sort_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def make_thumbnail(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)

    h, w = image.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    tile = np.full((THUMB_H + LABEL_H, THUMB_W, 3), 245, dtype=np.uint8)
    x = (THUMB_W - resized.shape[1]) // 2
    y = (THUMB_H - resized.shape[0]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized

    label = path.name
    cv2.putText(
        tile,
        label,
        (10, THUMB_H + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return tile


def make_overview(set_dir):
    files = sorted(set_dir.glob("*.jpg"), key=image_sort_key)
    rows = (len(files) + COLS - 1) // COLS
    canvas_h = MARGIN + rows * (THUMB_H + LABEL_H + MARGIN)
    canvas_w = MARGIN + COLS * (THUMB_W + MARGIN)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for index, path in enumerate(files):
        row = index // COLS
        col = index % COLS
        x = MARGIN + col * (THUMB_W + MARGIN)
        y = MARGIN + row * (THUMB_H + LABEL_H + MARGIN)
        tile = make_thumbnail(path)
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{set_dir.name}_overview.jpg"
    cv2.imwrite(str(output_path), canvas)
    print(output_path)


def main():
    for set_dir in sorted(path for path in IMAGE_ROOT.glob("Set_*") if path.is_dir()):
        make_overview(set_dir)


if __name__ == "__main__":
    main()
