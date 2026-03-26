from __future__ import annotations

import os
from pathlib import Path

import cv2

from preprocess import PreprocessError, preprocess_path


INPUT_DIR = Path("./data")
OUTPUT_DIR = Path("./output")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(INPUT_DIR.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        try:
            result = preprocess_path(str(image_path))
        except PreprocessError as exc:
            print(f"[SKIP] {image_path.name}: {exc}")
            continue

        out_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(out_path), result.preprocessed_image)
        print(f"[OK] {image_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
