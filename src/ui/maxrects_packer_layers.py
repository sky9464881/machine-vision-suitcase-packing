from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from maxrects_packer import (
    Item,
    Placement,
    MaxRectsBinPack,
    estimate_bin_size,
    load_items_from_folder,
    render_result,
)


# =========================================================
# 아르코 마커 기준 cm -> px 변환
# =========================================================

def cm_to_px(cm: float, marker_cm: float = 6.1, marker_px: float = 300.0) -> int:
    """
    실제 길이(cm)를 픽셀(px)로 변환한다.
    기본 기준:
        아르코 마커 6.1 cm == 300 px
    """
    if cm <= 0:
        raise ValueError("cm 값은 0보다 커야 합니다.")
    if marker_cm <= 0 or marker_px <= 0:
        raise ValueError("marker_cm, marker_px는 0보다 커야 합니다.")

    px_per_cm = marker_px / marker_cm
    return int(round(cm * px_per_cm))


def box_cm_to_bin_px(
    box_width_cm: float,
    box_height_cm: float,
    marker_cm: float = 6.1,
    marker_px: float = 300.0,
) -> Tuple[int, int]:
    """
    실제 상자 크기(cm)를 MaxRects에서 사용할 bin 크기(px)로 변환한다.
    """
    bin_w = cm_to_px(box_width_cm, marker_cm=marker_cm, marker_px=marker_px)
    bin_h = cm_to_px(box_height_cm, marker_cm=marker_cm, marker_px=marker_px)
    return bin_w, bin_h


# =========================================================
# 층별 배치
# =========================================================

def split_fittable_items(
    items: List[Item],
    bin_w: int,
    bin_h: int,
    allow_rotate: bool,
) -> Tuple[List[Item], List[Item]]:
    fittable: List[Item] = []
    oversized: List[Item] = []

    for item in items:
        fits = (item.width <= bin_w and item.height <= bin_h) or (
            allow_rotate and item.height <= bin_w and item.width <= bin_h
        )
        if fits:
            fittable.append(item)
        else:
            oversized.append(item)

    return fittable, oversized


def pack_one_layer(
    items: List[Item],
    bin_w: int,
    bin_h: int,
    allow_rotate: bool = True,
) -> Tuple[List[Placement], List[Item]]:
    packer = MaxRectsBinPack(bin_w, bin_h)
    placements: List[Placement] = []
    remaining: List[Item] = []

    # 큰 것부터 배치
    sorted_items = sorted(items, key=lambda x: x.width * x.height, reverse=True)

    for item in sorted_items:
        result = packer.insert(item.width, item.height, allow_rotate=allow_rotate)
        if result is None:
            remaining.append(item)
            continue

        x, y, w, h, rotated = result
        placements.append(
            Placement(
                item=item,
                x=x,
                y=y,
                width=w,
                height=h,
                rotated=rotated,
            )
        )

    return placements, remaining


def build_overview(layer_paths: List[str], output_path: str) -> Optional[str]:
    images = []
    for p in layer_paths:
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is not None:
            images.append((p, img))

    if not images:
        return None

    max_w = max(img.shape[1] for _, img in images)
    gap = 20
    total_h = sum(img.shape[0] for _, img in images) + gap * (len(images) - 1)
    canvas = np.full((total_h, max_w, 3), 255, dtype=np.uint8)

    y = 0
    for idx, (_, img) in enumerate(images, start=1):
        h, w = img.shape[:2]
        canvas[y:y + h, 0:w] = img
        cv.putText(
            canvas,
            f"Layer {idx}",
            (12, y + 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 180),
            2,
            cv.LINE_AA,
        )
        y += h + gap

    cv.imwrite(output_path, canvas)
    return output_path


def run_pack_layers(
    input_folder: str,
    output_dir: str,
    bin_width: Optional[int] = None,
    bin_height: Optional[int] = None,
    box_width_cm: Optional[float] = None,
    box_height_cm: Optional[float] = None,
    marker_cm: float = 6.1,
    marker_px: float = 300.0,
    allow_rotate: bool = True,
    padding: int = 8,
) -> Tuple[int, int, List[List[Placement]], List[str], List[Item]]:
    """
    입력 객체 이미지들을 여러 층으로 MaxRects 배치한다.

    우선순위:
    1) box_width_cm, box_height_cm 이 주어지면 -> marker 기준으로 bin px 계산
    2) 아니면 bin_width, bin_height 직접 사용
    3) 둘 다 없으면 estimate_bin_size 사용

    주의:
    - 객체 크기는 절대 스케일링하지 않는다.
    - PNG 원래 크기를 그대로 사용한다.
    """
    items = load_items_from_folder(input_folder, padding=padding)
    if not items:
        raise RuntimeError("입력 폴더에서 이미지 파일을 찾지 못했습니다.")

    # 실제 상자 크기(cm)가 주어진 경우 -> 아르코 기준으로 픽셀 bin 계산
    if box_width_cm is not None and box_height_cm is not None:
        bin_w, bin_h = box_cm_to_bin_px(
            box_width_cm=box_width_cm,
            box_height_cm=box_height_cm,
            marker_cm=marker_cm,
            marker_px=marker_px,
        )
    elif bin_width is not None and bin_height is not None:
        bin_w, bin_h = int(bin_width), int(bin_height)
    else:
        bin_w, bin_h = estimate_bin_size(items)

    fittable_items, oversized_items = split_fittable_items(items, bin_w, bin_h, allow_rotate)

    if not fittable_items:
        oversized_desc = ", ".join(
            f"{item.label or item.name} ({item.width}x{item.height})" for item in oversized_items
        ) or "없음"
        raise RuntimeError(
            "상자에 들어가는 객체가 없습니다. "
            f"bin={bin_w}x{bin_h}, 제외 대상={oversized_desc}"
        )

    os.makedirs(output_dir, exist_ok=True)

    remaining = fittable_items[:]
    all_layers: List[List[Placement]] = []
    layer_paths: List[str] = []
    layer_idx = 1

    while remaining:
        placed, remaining = pack_one_layer(
            remaining,
            bin_w,
            bin_h,
            allow_rotate=allow_rotate,
        )

        if not placed:
            raise RuntimeError("현재 층에 아무 객체도 배치하지 못했습니다. bin 크기를 늘려야 합니다.")

        all_layers.append(placed)

        layer_path = str(Path(output_dir) / f"layer_{layer_idx:02d}.png")
        render_result(bin_w, bin_h, placed, layer_path, padding=padding)
        layer_paths.append(layer_path)

        layer_idx += 1

    overview_path = str(Path(output_dir) / "layers_overview.png")
    build_overview(layer_paths, overview_path)

    return bin_w, bin_h, all_layers, layer_paths, oversized_items


# =========================================================
# CLI
# =========================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="객체 이미지 폴더를 여러 층으로 MaxRects 배치합니다. "
                    "실제 상자 크기(cm)를 아르코 마커 기준으로 픽셀 bin으로 변환할 수 있습니다."
    )

    parser.add_argument("--input-folder", required=True, help="객체 PNG/JPG 폴더 경로")
    parser.add_argument("--output-dir", required=True, help="층별 결과 이미지 저장 폴더")

    # 직접 bin(px) 지정
    parser.add_argument("--bin-width", type=int, default=None, help="캐리어 너비(px)")
    parser.add_argument("--bin-height", type=int, default=None, help="캐리어 높이(px)")

    # 실제 상자 크기(cm) 지정
    parser.add_argument("--box-width-cm", type=float, default=None, help="실제 상자 가로(cm)")
    parser.add_argument("--box-height-cm", type=float, default=None, help="실제 상자 세로(cm)")

    # 아르코 마커 기준
    parser.add_argument("--marker-cm", type=float, default=6.1, help="아르코 마커 실제 한 변 길이(cm)")
    parser.add_argument("--marker-px", type=float, default=300.0, help="투시변환 후 아르코 마커 한 변 길이(px)")

    parser.add_argument("--padding", type=int, default=8, help="객체 주변 여백(px)")
    parser.add_argument("--no-rotate", action="store_true", help="회전 배치 비활성화")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()

    bw, bh, layers, paths, oversized = run_pack_layers(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        bin_width=args.bin_width,
        bin_height=args.bin_height,
        box_width_cm=args.box_width_cm,
        box_height_cm=args.box_height_cm,
        marker_cm=args.marker_cm,
        marker_px=args.marker_px,
        allow_rotate=not args.no_rotate,
        padding=args.padding,
    )

    print("[OK] layer packing complete")
    print(f"input_folder : {args.input_folder}")
    print(f"output_dir   : {args.output_dir}")
    print(f"bin size(px) : {bw} x {bh}")
    print(f"layers       : {len(layers)}")
    for idx, layer in enumerate(layers, start=1):
        print(f"- layer {idx:02d}: {len(layer)} items")
    for p in paths:
        print(f"  saved: {p}")
