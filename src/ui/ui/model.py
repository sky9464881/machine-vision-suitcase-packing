from __future__ import annotations

from typing import Any

import cv2
from ultralytics import YOLO

# 모델 없으면 이게 인식 됩니다.
YOLO_MODEL_PATH = "best.pt"

_yolo_model: YOLO | None = None



def get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model



def run_model_on_image(
    image: Any,
    *,
    conf: float = 0.15,
    iou: float = 0.40,
    imgsz: int = 1280,
) -> list[dict[str, Any]]:
    """
    전처리 완료된 이미지를 입력으로 받아 YOLO 추론 결과를 반환한다.
    모든 mask는 입력 image 기준 크기로 맞춘다.
    """
    if image is None:
        raise ValueError("model input image is None")

    h, w = image.shape[:2]
    yolo_model = get_model()

    res = yolo_model(
        image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )[0]

    masks_data = res.masks.data.cpu().numpy() if res.masks is not None else None
    outputs: list[dict[str, Any]] = []

    for i, box in enumerate(res.boxes):
        mask = None
        if masks_data is not None and i < len(masks_data):
            resized_mask = cv2.resize(
                masks_data[i],
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = resized_mask > 0.5

        outputs.append(
            {
                "label": yolo_model.names[int(box.cls[0].item())],
                "confidence": float(box.conf[0].item()),
                "mask": mask,
            }
        )

    return outputs
