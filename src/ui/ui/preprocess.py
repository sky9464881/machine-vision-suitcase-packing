from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from crop import detect_table_and_crop
from get_perspective_image import get_perspective_img

PREPROCESS_MARKER_SIZE_PX = 100


@dataclass
class PreprocessResult:
    source_path: str | None
    source_image: Any
    perspective_image: Any
    preprocessed_image: Any
    crop_bbox: tuple[int, int, int, int]
    steps: list[str]


class PreprocessError(RuntimeError):
    pass


def preprocess_image(image: Any, *, source_path: str | None = None) -> PreprocessResult:
    """
    입력 이미지 -> 전처리 이미지

    규칙
    1) perspective correction 반드시 성공
    2) crop 반드시 성공
    실패 시 예외 발생
    """
    if image is None:
        raise PreprocessError("입력 이미지가 없습니다.")

    try:
        perspective_image, _pixel_scale, _H, _src_pts = get_perspective_img(
            image,
            ref_marker_size_px=PREPROCESS_MARKER_SIZE_PX,
            debug=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise PreprocessError(f"원근 보정 실패: {exc}") from exc

    try:
        cropped, bbox = detect_table_and_crop(perspective_image)
    except Exception as exc:  # noqa: BLE001
        raise PreprocessError(f"크롭 실패: {exc}") from exc

    if cropped is None or bbox is None:
        raise PreprocessError("크롭 실패: 책상 영역을 찾지 못했습니다.")

    return PreprocessResult(
        source_path=source_path,
        source_image=image,
        perspective_image=perspective_image,
        preprocessed_image=cropped,
        crop_bbox=bbox,
        steps=["perspective", "crop"],
    )



def preprocess_path(image_path: str) -> PreprocessResult:
    image = cv2.imread(image_path)
    if image is None:
        raise PreprocessError(f"이미지를 읽을 수 없습니다: {image_path}")
    return preprocess_image(image, source_path=image_path)
