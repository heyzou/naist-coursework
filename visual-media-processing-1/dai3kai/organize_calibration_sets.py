from pathlib import Path
import argparse
import shutil


SETS = {
    "Set_A_3": [
        "S_72114180_0.jpg",
        "S_72114181_0.jpg",
        "S_72114182_0.jpg",
    ],
    "Set_B_7": [
        "S_72114180_0.jpg",
        "S_72114181_0.jpg",
        "S_72114182_0.jpg",
        "S_72114183_0.jpg",
        "S_72114184_0.jpg",
        "S_72114185_0.jpg",
        "S_72114186_0.jpg",
    ],
    "Set_C_11": [
        "S_72114180_0.jpg",
        "S_72114181_0.jpg",
        "S_72114182_0.jpg",
        "S_72114183_0.jpg",
        "S_72114184_0.jpg",
        "S_72114185_0.jpg",
        "S_72114186_0.jpg",
        "S_72114187_0.jpg",
        "S_72114188_0.jpg",
        "S_72114189_0.jpg",
    ],
    "Set_D_15": [
        "S_72114180_0.jpg",
        "S_72114181_0.jpg",
        "S_72114182_0.jpg",
        "S_72114183_0.jpg",
        "S_72114184_0.jpg",
        "S_72114185_0.jpg",
        "S_72114186_0.jpg",
        "S_72114187_0.jpg",
        "S_72114188_0.jpg",
        "S_72114189_0.jpg",
        "S_72114191_0.jpg",
        "S_72114192_0.jpg",
        "S_72114193_0.jpg",
        "S_72114194_0.jpg",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy calibration images into image-set folders."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("images"),
        help="Directory containing the original images. Default: images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images"),
        help="Directory where set folders are created. Default: images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = args.source
    output_dir = args.output

    missing = []
    for set_name, filenames in SETS.items():
        set_dir = output_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)

        expected_count = set_name.rsplit("_", 1)[-1]
        if expected_count.isdigit() and int(expected_count) != len(filenames):
            print(
                f"Warning: {set_name} contains {len(filenames)} listed files, "
                f"but its name suggests {expected_count}."
            )

        for filename in filenames:
            source_path = source_dir / filename
            destination_path = set_dir / filename
            if not source_path.exists():
                missing.append(str(source_path))
                continue
            shutil.copy2(source_path, destination_path)
            print(f"Copied {source_path} -> {destination_path}")

    if missing:
        print("\nMissing source files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
