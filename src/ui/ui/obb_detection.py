import os
import cv2 as cv
import numpy as np


def get_obb_from_mask(mask):
    if mask is None:
        return None

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv.contourArea)
    rect = cv.minAreaRect(cnt)

    (cx, cy), (w, h), angle = rect
    box = cv.boxPoints(rect)
    box = np.int32(box)

    return (cx, cy), (w, h), angle, box


def extract_upright_object(image, mask, cx, cy, w, h, angle):
    h_img, w_img = image.shape[:2]

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    obj_img = cv.bitwise_and(image, image, mask=mask)

    # 긴 축이 세로가 되도록 회전
    if w < h:
        rotate_angle = angle
        crop_w, crop_h = int(round(w)), int(round(h))
    else:
        rotate_angle = angle + 90
        crop_w, crop_h = int(round(h)), int(round(w))

    M = cv.getRotationMatrix2D((cx, cy), rotate_angle, 1.0)

    rotated_img = cv.warpAffine(
        obj_img,
        M,
        (w_img, h_img),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    rotated_mask = cv.warpAffine(
        mask,
        M,
        (w_img, h_img),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0
    )

    x1 = max(0, int(round(cx - crop_w / 2)))
    y1 = max(0, int(round(cy - crop_h / 2)))
    x2 = min(w_img, int(round(cx + crop_w / 2)))
    y2 = min(h_img, int(round(cy + crop_h / 2)))

    crop_img = rotated_img[y1:y2, x1:x2]
    crop_mask = rotated_mask[y1:y2, x1:x2]

    # mask 기준으로 여백 제거
    ys, xs = np.where(crop_mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        tx1, tx2 = xs.min(), xs.max() + 1
        ty1, ty2 = ys.min(), ys.max() + 1
        crop_img = crop_img[ty1:ty2, tx1:tx2]
        crop_mask = crop_mask[ty1:ty2, tx1:tx2]

    return crop_img, crop_mask


def save_obb_objects(image, detections, save_dir="./output/obb_objects", min_area=30):
    os.makedirs(save_dir, exist_ok=True)

    saved_results = []

    for idx, det in enumerate(detections):
        label = det.get("label", "object")
        conf = det.get("confidence", 0.0)
        mask = det.get("mask", None)

        if mask is None:
            continue

        if mask.dtype != np.uint8:
            mask_u8 = (mask > 0).astype(np.uint8) * 255
        else:
            mask_u8 = mask.copy()

        if cv.countNonZero(mask_u8) < min_area:
            continue

        obb = get_obb_from_mask(mask_u8)
        if obb is None:
            continue

        (cx, cy), (w, h), angle, box = obb

        upright_img, upright_mask = extract_upright_object(
            image=image,
            mask=mask_u8,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            angle=angle
        )

        if upright_img is None or upright_img.size == 0:
            continue

        vis = image.copy()
        cv.drawContours(vis, [box], 0, (0, 255, 0), 2)
        cv.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        safe_label = str(label).replace(" ", "_").replace("/", "_")
        base_name = f"obj_{idx:03d}_{safe_label}_{conf:.2f}"

        obj_path = os.path.join(save_dir, f"{base_name}.png")

        cv.imwrite(obj_path, upright_img)

        saved_results.append({
            "object_id": idx,
            "label": label,
            "confidence": conf,
            "save_path": obj_path,
            "center": (cx, cy),
            "size": (w, h),
            "angle": angle,
        })

    return saved_results