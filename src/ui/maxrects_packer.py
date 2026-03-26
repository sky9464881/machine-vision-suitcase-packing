from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np


# =========================================================
# 데이터 구조
# =========================================================

@dataclass
class Item:
    name: str
    label: str
    confidence: float
    image: np.ndarray       # BGR
    mask: np.ndarray        # uint8, 0 or 255
    width: int
    height: int


@dataclass
class Placement:
    item: Item
    x: int
    y: int
    width: int
    height: int
    rotated: bool


@dataclass
class FreeRect:
    x: int
    y: int
    width: int
    height: int


# =========================================================
# 유틸
# =========================================================

FILENAME_RE = re.compile(r"obj_(\d+)_(.+?)_(\d+(?:\.\d+)?)\.png$", re.IGNORECASE)


def parse_filename(filename: str) -> Tuple[str, float]:
    m = FILENAME_RE.match(filename)
    if not m:
        stem = Path(filename).stem
        return stem, 0.0
    return m.group(2), float(m.group(3))


def build_mask_from_black_background(img_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return mask


def rotate_image_and_mask_90(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return cv.rotate(img, cv.ROTATE_90_CLOCKWISE), cv.rotate(mask, cv.ROTATE_90_CLOCKWISE)


# =========================================================
# MaxRects
# =========================================================

class MaxRectsBinPack:
    def __init__(self, width: int, height: int):
        self.bin_width = int(width)
        self.bin_height = int(height)
        self.free_rects: List[FreeRect] = [FreeRect(0, 0, self.bin_width, self.bin_height)]
        self.used_rects: List[FreeRect] = []

    def insert(self, rect_w: int, rect_h: int, allow_rotate: bool = True) -> Optional[Tuple[int, int, int, int, bool]]:
        best_score = None
        best_index = -1
        best_node = None

        for i, free in enumerate(self.free_rects):
            # not rotated
            if rect_w <= free.width and rect_h <= free.height:
                score = self._score_best_area_fit(rect_w, rect_h, free)
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = i
                    best_node = (free.x, free.y, rect_w, rect_h, False)

            # rotated
            if allow_rotate and rect_h <= free.width and rect_w <= free.height:
                score = self._score_best_area_fit(rect_h, rect_w, free)
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = i
                    best_node = (free.x, free.y, rect_h, rect_w, True)

        if best_node is None:
            return None

        placed = FreeRect(best_node[0], best_node[1], best_node[2], best_node[3])
        self._place_rect(best_index, placed)
        return best_node

    @staticmethod
    def _score_best_area_fit(rect_w: int, rect_h: int, free: FreeRect) -> Tuple[int, int]:
        leftover_h = free.height - rect_h
        leftover_w = free.width - rect_w
        area_fit = free.width * free.height - rect_w * rect_h
        short_side_fit = min(leftover_w, leftover_h)
        return area_fit, short_side_fit

    def _place_rect(self, free_index: int, used: FreeRect) -> None:
        new_free_rects: List[FreeRect] = []
        used_box = used

        for free in self.free_rects:
            if not self._intersects(free, used_box):
                new_free_rects.append(free)
                continue

            # 위
            if used_box.y > free.y and used_box.y < free.y + free.height:
                new_free_rects.append(
                    FreeRect(free.x, free.y, free.width, used_box.y - free.y)
                )

            # 아래
            free_bottom = free.y + free.height
            used_bottom = used_box.y + used_box.height
            if used_bottom < free_bottom:
                new_free_rects.append(
                    FreeRect(free.x, used_bottom, free.width, free_bottom - used_bottom)
                )

            # 왼쪽
            if used_box.x > free.x and used_box.x < free.x + free.width:
                left_x = free.x
                left_w = used_box.x - free.x
                top_y = max(free.y, used_box.y)
                bottom_y = min(free.y + free.height, used_box.y + used_box.height)
                if bottom_y > top_y:
                    new_free_rects.append(
                        FreeRect(left_x, top_y, left_w, bottom_y - top_y)
                    )

            # 오른쪽
            free_right = free.x + free.width
            used_right = used_box.x + used_box.width
            if used_right < free_right:
                right_x = used_right
                right_w = free_right - used_right
                top_y = max(free.y, used_box.y)
                bottom_y = min(free.y + free.height, used_box.y + used_box.height)
                if bottom_y > top_y:
                    new_free_rects.append(
                        FreeRect(right_x, top_y, right_w, bottom_y - top_y)
                    )

        self.free_rects = self._prune_free_list(new_free_rects)
        self.used_rects.append(used)

    @staticmethod
    def _intersects(a: FreeRect, b: FreeRect) -> bool:
        return not (
            b.x >= a.x + a.width or
            b.x + b.width <= a.x or
            b.y >= a.y + a.height or
            b.y + b.height <= a.y
        )

    @staticmethod
    def _contains(a: FreeRect, b: FreeRect) -> bool:
        return (
            b.x >= a.x and
            b.y >= a.y and
            b.x + b.width <= a.x + a.width and
            b.y + b.height <= a.y + a.height
        )

    def _prune_free_list(self, rects: List[FreeRect]) -> List[FreeRect]:
        pruned: List[FreeRect] = []

        # 0 이하 제거
        rects = [r for r in rects if r.width > 0 and r.height > 0]

        for i, a in enumerate(rects):
            contained = False
            for j, b in enumerate(rects):
                if i != j and self._contains(b, a):
                    contained = True
                    break
            if not contained:
                pruned.append(a)

        # 중복 비슷한 것 제거
        unique: List[FreeRect] = []
        seen = set()
        for r in pruned:
            key = (r.x, r.y, r.width, r.height)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


# =========================================================
# 로딩 / 배치 / 렌더링
# =========================================================


def load_items_from_folder(folder_path: str, padding: int = 8) -> List[Item]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"폴더가 없습니다: {folder_path}")

    image_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not image_paths:
        raise ValueError(f"이미지 파일이 없습니다: {folder_path}")

    items: List[Item] = []
    for path in image_paths:
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 읽기 실패: {path}")
            continue

        label, conf = parse_filename(path.name)
        h, w = img.shape[:2]
        mask = build_mask_from_black_background(img)

        items.append(Item(
            name=path.name,
            label=label,
            confidence=conf,
            image=img,
            mask=mask,
            width=w + padding * 2,
            height=h + padding * 2,
        ))

    if not items:
        raise ValueError("유효한 이미지가 없습니다.")

    return items


def estimate_bin_size(items: List[Item], aspect_ratio: Tuple[int, int] = (3, 4)) -> Tuple[int, int]:
    total_area = sum(item.width * item.height for item in items)
    max_w = max(item.width for item in items)
    max_h = max(item.height for item in items)

    ar_w, ar_h = aspect_ratio
    ratio = ar_w / ar_h

    base_h = math.sqrt(total_area / ratio)
    base_w = base_h * ratio

    width = max(int(math.ceil(base_w)), max_w)
    height = max(int(math.ceil(base_h)), max_h)

    return width, height


def try_pack(items: List[Item], bin_width: int, bin_height: int, allow_rotate: bool = True) -> Optional[List[Placement]]:
    packer = MaxRectsBinPack(bin_width, bin_height)
    placements: List[Placement] = []

    # 큰 것부터 넣는 게 보통 안정적
    sorted_items = sorted(items, key=lambda x: x.width * x.height, reverse=True)

    for item in sorted_items:
        result = packer.insert(item.width, item.height, allow_rotate=allow_rotate)
        if result is None:
            return None

        x, y, w, h, rotated = result
        placements.append(Placement(
            item=item,
            x=x,
            y=y,
            width=w,
            height=h,
            rotated=rotated,
        ))

    return placements


def auto_pack(items: List[Item], aspect_ratio: Tuple[int, int] = (3, 4), allow_rotate: bool = True) -> Tuple[int, int, List[Placement]]:
    est_w, est_h = estimate_bin_size(items, aspect_ratio)

    # 조금씩 키워가며 들어가는 최소 크기 찾기
    scales = [1.00, 1.08, 1.15, 1.25, 1.40, 1.60, 1.85, 2.10, 2.40, 2.80, 3.20]
    for scale in scales:
        bw = int(math.ceil(est_w * scale))
        bh = int(math.ceil(est_h * scale))
        placements = try_pack(items, bw, bh, allow_rotate=allow_rotate)
        if placements is not None:
            return bw, bh, placements

    raise RuntimeError("자동 배치 실패: bin 크기를 더 크게 주거나 padding을 줄여보세요.")


def overlay_with_mask(canvas: np.ndarray, obj_img: np.ndarray, obj_mask: np.ndarray, x: int, y: int) -> None:
    h, w = obj_img.shape[:2]
    roi = canvas[y:y+h, x:x+w]

    mask_bool = obj_mask > 0
    roi[mask_bool] = obj_img[mask_bool]



def render_result(
    bin_width: int,
    bin_height: int,
    placements: List[Placement],
    save_path: str,
    background_color: Tuple[int, int, int] = (245, 245, 245),
    suitcase_color: Tuple[int, int, int] = (220, 228, 240),
    border_color: Tuple[int, int, int] = (50, 50, 50),
    padding: int = 8,
) -> np.ndarray:
    canvas = np.full((bin_height + 80, bin_width + 80, 3), background_color, dtype=np.uint8)

    # 캐리어 영역
    sx, sy = 40, 40
    cv.rectangle(canvas, (sx, sy), (sx + bin_width, sy + bin_height), suitcase_color, thickness=-1)
    cv.rectangle(canvas, (sx, sy), (sx + bin_width, sy + bin_height), border_color, thickness=3)

    for p in placements:
        item = p.item
        img = item.image.copy()
        mask = item.mask.copy()

        if p.rotated:
            img, mask = rotate_image_and_mask_90(img, mask)

        obj_h, obj_w = img.shape[:2]
        paste_x = sx + p.x + padding
        paste_y = sy + p.y + padding

        overlay_with_mask(canvas, img, mask, paste_x, paste_y)

        # 디버그용 rect
        cv.rectangle(
            canvas,
            (sx + p.x, sy + p.y),
            (sx + p.x + p.width, sy + p.y + p.height),
            (90, 90, 90),
            thickness=1,
        )

        text = f"{item.label} ({item.confidence:.2f})"
        text_y = max(sy + p.y + 18, 55)
        cv.putText(canvas, text, (sx + p.x + 4, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv.LINE_AA)

    usage_area = sum(p.width * p.height for p in placements)
    usage_ratio = usage_area / float(bin_width * bin_height)
    info = f"bin: {bin_width}x{bin_height} | items: {len(placements)} | fill: {usage_ratio * 100:.2f}%"
    cv.putText(canvas, info, (40, 28), cv.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv.LINE_AA)

    cv.imwrite(save_path, canvas)
    return canvas


# =========================================================
# 실행 함수
# =========================================================


def run_pack(
    input_folder: str,
    output_path: str,
    bin_width: Optional[int] = None,
    bin_height: Optional[int] = None,
    allow_rotate: bool = True,
    padding: int = 8,
) -> Tuple[int, int, List[Placement]]:
    items = load_items_from_folder(input_folder, padding=padding)

    if bin_width is not None and bin_height is not None:
        placements = try_pack(items, bin_width, bin_height, allow_rotate=allow_rotate)
        if placements is None:
            raise RuntimeError("지정한 캐리어 크기에 모든 물체를 배치하지 못했습니다.")
        final_w, final_h = bin_width, bin_height
    else:
        final_w, final_h, placements = auto_pack(items, aspect_ratio=(3, 4), allow_rotate=allow_rotate)

    render_result(final_w, final_h, placements, output_path, padding=padding)
    return final_w, final_h, placements


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="객체 이미지 폴더를 MaxRects로 배치하고 결과 이미지를 저장합니다.")
    parser.add_argument("--input-folder", required=True, help="예: /mnt/data/obb_extracted2/obb/img000")
    parser.add_argument("--output", required=True, help="예: /mnt/data/packed_img000.png")
    parser.add_argument("--bin-width", type=int, default=None, help="캐리어 너비(px)")
    parser.add_argument("--bin-height", type=int, default=None, help="캐리어 높이(px)")
    parser.add_argument("--padding", type=int, default=8, help="각 물체 주변 여유 픽셀")
    parser.add_argument("--no-rotate", action="store_true", help="회전 비허용")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    final_w, final_h, placements = run_pack(
        input_folder=args.input_folder,
        output_path=args.output,
        bin_width=args.bin_width,
        bin_height=args.bin_height,
        allow_rotate=not args.no_rotate,
        padding=args.padding,
    )

    print("[OK] packing complete")
    print(f"input_folder : {args.input_folder}")
    print(f"output       : {args.output}")
    print(f"bin size     : {final_w} x {final_h}")
    print(f"placements   : {len(placements)}")
    for p in placements:
        print(f"- {p.item.name:30s} -> x={p.x:4d}, y={p.y:4d}, w={p.width:4d}, h={p.height:4d}, rotated={p.rotated}")
