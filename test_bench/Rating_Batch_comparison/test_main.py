import sys
import warnings
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from ultralytics import YOLO

warnings.filterwarnings("ignore")

try:
    from koreanize_matplotlib import koreanize
    koreanize()
except ImportError:
    pass


# ============================================================
# 설정
# ============================================================
EVAL_DATASET_DIR = "eval_dataset"
OUTPUT_DIR = "results"  # None이면 results_YYYYmmdd_HHMMSS 자동 생성
YOLO_X_MODEL_PATH = "yolo11x-seg.pt"
YOLO_N_MODEL_PATH = "yolo11n-seg.pt"

OUT: Path


# ============================================================
# 모델 추론
# ============================================================
def _run_yolo(
    yolo_model: YOLO,
    img_path: str,
    width: int,
    height: int,
    conf: float = 0.15,
    iou: float = 0.40,
    imgsz: int = 1280,
) -> List[dict]:
    """YOLO 모델 추론 결과를 공통 포맷으로 변환한다."""
    result = yolo_model(
        img_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )[0]

    masks_data = result.masks.data.cpu().numpy() if result.masks is not None else None
    outputs: List[dict] = []

    for i, box in enumerate(result.boxes):
        mask = None
        if masks_data is not None:
            resized = cv2.resize(masks_data[i], (width, height), interpolation=cv2.INTER_LINEAR)
            mask = resized > 0.5

        outputs.append(
            {
                "label": yolo_model.names[int(box.cls[0].item())],
                "confidence": float(box.conf[0].item()),
                "mask": mask,
            }
        )
    return outputs


def build_models() -> Dict[str, Callable[[str, int, int], List[dict]]]:
    """평가할 모델들을 등록한다."""
    yolo_x = YOLO(YOLO_X_MODEL_PATH)
    yolo_n = YOLO(YOLO_N_MODEL_PATH)
    yolo_26 = YOLO("yolo26x-seg.pt")
    yolo_bh = YOLO("best_FT_ epoch10_bh.pt")
    yolo_random = YOLO("best_FT_epoch5_full.pt")
    yolo_coco_stopped = YOLO("best_FT_stoopped.pt")
    yolo_coco_bal_full = YOLO("best_FT_coco_full_bal.pt")

    def model_yolo_x(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_x, img_path, width, height)

    def model_yolo_n(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_n, img_path, width, height)
    def model_yolo_1(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_26, img_path, width, height)
    def model_yolo_2(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_bh, img_path, width, height)
    def model_yolo_3(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_random, img_path, width, height)
    def model_yolo_4(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_coco_stopped, img_path, width, height)
    def model_yolo_5(img_path: str, width: int, height: int) -> List[dict]:
        return _run_yolo(yolo_coco_bal_full, img_path, width, height)

    models = {
        "YOLO11x-seg": model_yolo_x,
        "YOLO11n-seg": model_yolo_n,
        "YOLO26-seg": model_yolo_1,
        "YOLO_bh-seg": model_yolo_2,
        "YOLO_random-seg": model_yolo_3,
        "YOLO_coco_stopped-seg": model_yolo_4,
        "YOLO_coco_bal-seg": model_yolo_5,

    }
    print(f"✅ 등록된 모델: {list(models.keys())}")
    return models


# ============================================================
# 평가 유틸
# ============================================================
def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_size_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """겹친 면적 기반 크기 유사도. min(intersection/pred, intersection/gt)"""
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    if intersection <= 0 or area_a <= 0 or area_b <= 0:
        return 0.0
    return float(min(intersection / area_a, intersection / area_b))


def compute_precision_recall(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    precision = float(intersection / pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
    recall = float(intersection / gt_mask.sum()) if gt_mask.sum() > 0 else 0.0
    return precision, recall



def load_gt_masks(gt_dir: Path, width: int, height: int) -> List[dict]:
    """GT 마스크 PNG 파일들을 읽어 리스트로 반환한다."""
    results: List[dict] = []
    for file_path in sorted(gt_dir.glob("*.png")):
        parts = file_path.stem.rsplit("_", 1)
        label = (parts[0] if len(parts) == 2 and parts[1].isdigit() else file_path.stem).replace("_", " ")

        raw = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"  ⚠️  마스크 로드 실패: {file_path}")
            continue

        resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
        results.append({"label": label, "mask": resized > 127})
    return results



def match_and_score(gt_masks: List[dict], detections: List[dict]) -> List[dict]:
    """GT 마스크와 탐지 결과를 매칭하여 평가 결과를 생성한다."""
    detections = [{**det, "_idx": i} for i, det in enumerate(detections)]
    used_indices = set()
    results: List[dict] = []

    for gt in gt_masks:
        gt_label = gt["label"]
        gt_mask = gt["mask"]
        best_iou = -1.0
        best_det: Optional[dict] = None

        # 1순위: 같은 레이블 중 IoU 최대
        for det in detections:
            if det["_idx"] in used_indices or det["label"] != gt_label or det["mask"] is None:
                continue
            iou = compute_iou(gt_mask, det["mask"])
            if iou > best_iou:
                best_iou = iou
                best_det = det

        # 2순위: 전체 탐지 중 IoU 최대 fallback
        if best_det is None:
            for det in detections:
                if det["_idx"] in used_indices or det["mask"] is None:
                    continue
                iou = compute_iou(gt_mask, det["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            if best_det is not None and best_iou <= 0.0:
                best_det = None

        if best_det is not None:
            used_indices.add(best_det["_idx"])
            precision, recall = compute_precision_recall(gt_mask, best_det["mask"])
            size_similarity = compute_size_similarity(gt_mask, best_det["mask"])
            label_correct = int(gt_label == best_det["label"])
            results.append(
                {
                    "gt_label": gt_label,
                    "pred_label": best_det["label"],
                    "confidence": best_det["confidence"],
                    "iou": best_iou,
                    "precision": precision,
                    "recall": recall,
                    "size_similarity": size_similarity,
                    "label_correct": label_correct,
                    "score": label_correct * best_iou * best_det["confidence"],
                    "matched": True,
                }
            )
        else:
            results.append(
                {
                    "gt_label": gt_label,
                    "pred_label": None,
                    "confidence": 0.0,
                    "iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "size_similarity": 0.0,
                    "label_correct": 0,
                    "score": 0.0,
                    "matched": False,
                }
            )

    return results



def get_image_file(data_dir: Path) -> Optional[Path]:
    """폴더에서 평가 대상 이미지 파일 하나를 찾는다."""
    images = [*data_dir.glob("*.jpg"), *[f for f in data_dir.glob("*.png") if "gt_masks" not in str(f)]]
    return images[0] if images else None


# ============================================================
# 시각화
# ============================================================
def draw_confusion_matrix(model_name: str, img_results: List[dict]) -> tuple:
    pair_counts = defaultdict(int)
    gt_label_set = set()
    pred_label_set = set()

    for img_result in img_results:
        for row in img_result["eval_results"]:
            gt = row["gt_label"]
            pred = row["pred_label"] if row["pred_label"] else "(미탐지)"
            pair_counts[(gt, pred)] += 1
            gt_label_set.add(gt)
            pred_label_set.add(pred)

    true_labels = sorted(gt_label_set)
    pred_labels = sorted(pred_label_set - {"(미탐지)"}) + (["(미탐지)"] if "(미탐지)" in pred_label_set else [])

    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for (gt, pred), count in pair_counts.items():
        matrix[true_labels.index(gt), pred_labels.index(pred)] += count

    cm_precision = {}
    cm_recall = {}
    for label in true_labels:
        row_idx = true_labels.index(label)
        tp = matrix[row_idx, pred_labels.index(label)] if label in pred_labels else 0
        row_sum = matrix[row_idx].sum()
        col_sum = matrix[:, pred_labels.index(label)].sum() if label in pred_labels else 0
        cm_recall[label] = float(tp / row_sum) if row_sum > 0 else 0.0
        cm_precision[label] = float(tp / col_sum) if col_sum > 0 else 0.0

    macro_p = float(np.mean(list(cm_precision.values()))) if cm_precision else 0.0
    macro_r = float(np.mean(list(cm_recall.values()))) if cm_recall else 0.0
    macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)) if (macro_p + macro_r) > 0 else 0.0

    fig_h = max(5, len(true_labels) * 0.7 + 2)
    fig_w = max(7, len(pred_labels) * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)

    color_map = {
        "miss": "#e74c3c",
        "correct": "#2ecc71",
        "wrong": "#e67e22",
    }

    for r in range(len(true_labels)):
        for c in range(len(pred_labels)):
            value = matrix[r, c]
            if value == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor="#f8f9fa",
                        edgecolor="#dee2e6",
                        linewidth=0.5,
                    )
                )
                continue

            pred_label = pred_labels[c]
            if pred_label == "(미탐지)":
                facecolor = color_map["miss"]
            elif true_labels[r] == pred_label:
                facecolor = color_map["correct"]
            else:
                facecolor = color_map["wrong"]

            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.85,
                )
            )
            ax.text(c, r, str(value), ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    ax.set_xticks(range(len(pred_labels)))
    ax.set_yticks(range(len(true_labels)))
    ax.set_xticklabels(pred_labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(true_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label (GT)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"[{model_name}] Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(np.arange(-0.5, len(pred_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(true_labels), 1), minor=True)
    ax.grid(which="minor", color="#dee2e6", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=color_map["correct"], label="✅ 정답 (GT==PRED)"),
            mpatches.Patch(facecolor=color_map["wrong"], label="⚠️  오탐 (GT≠PRED)"),
            mpatches.Patch(facecolor=color_map["miss"], label="❌ 미탐지"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.18),
        fontsize=9,
        ncol=3,
    )
    plt.tight_layout()

    filename = f"output_confusion_{model_name.replace(' ', '_')}.png"
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 저장: {OUT / filename}")

    total = sum(pair_counts.values())
    correct = sum(count for (gt, pred), count in pair_counts.items() if gt == pred)
    missed = sum(count for (gt, pred), count in pair_counts.items() if pred == "(미탐지)")
    wrong = total - correct - missed

    if total > 0:
        print(f"  정답(GT==PRED) : {correct}/{total}  ({correct/total:.1%})")
        print(f"  오탐(라벨 틀림): {wrong}/{total}  ({wrong/total:.1%})")
        print(f"  미탐지         : {missed}/{total}  ({missed/total:.1%})")

    return true_labels, cm_precision, cm_recall, macro_p, macro_r, macro_f1



def draw_pr_bar(model_name: str, store: dict) -> None:
    labels = store["true_labels"]
    prec_vals = [store["cm_precision"][label] for label in labels]
    rec_vals = [store["cm_recall"][label] for label in labels]
    f1_vals = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(prec_vals, rec_vals)]
    macro_p = store["macro_p"]
    macro_r = store["macro_r"]
    macro_f1 = store["macro_f1"]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.0), 6))

    bars = [
        ax.bar(x - width, prec_vals, width, label="Precision", color="#9b59b6", alpha=0.87),
        ax.bar(x, rec_vals, width, label="Recall", color="#f39c12", alpha=0.87),
        ax.bar(x + width, f1_vals, width, label="F1 Score", color="#3498db", alpha=0.87),
    ]
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.axhline(macro_p, color="#9b59b6", linestyle="--", linewidth=1.3, label=f"macro Precision:{macro_p:.3f}")
    ax.axhline(macro_r, color="#f39c12", linestyle="--", linewidth=1.3, label=f"macro Recall:{macro_r:.3f}")
    ax.axhline(macro_f1, color="#3498db", linestyle=":", linewidth=1.3, label=f"macro F1:{macro_f1:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("값", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title(f"[{model_name}] Precision / Recall / F1", fontsize=12, fontweight="bold")
    plt.tight_layout()

    filename = f"output_pr_{model_name.replace(' ', '_')}.png"
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 저장: {OUT / filename}")



def draw_pr_heatmap(model_name: str, store: dict) -> None:
    true_labels = store["true_labels"]
    prec_h = [store["cm_precision"][label] for label in true_labels]
    rec_h = [store["cm_recall"][label] for label in true_labels]
    f1_h = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(prec_h, rec_h)]
    data = np.array([prec_h, rec_h, f1_h]).T

    col_labels = ["Precision", "Recall", "F1"]
    n_labels, n_cols = len(true_labels), 3
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f39c12", "#2ecc71"])

    fig, ax = plt.subplots(figsize=(7, max(4, n_labels * 0.55 + 2.5)))
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for r in range(n_labels):
        for c in range(n_cols):
            value = data[r, c]
            ax.text(
                c,
                r,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if value < 0.6 else "black",
            )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(true_labels, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    macro_p = store["macro_p"]
    macro_r = store["macro_r"]
    macro_f1 = store["macro_f1"]
    macro_str = f"Precision:{macro_p:.4f}  Recall:{macro_r:.4f}  F1:{macro_f1:.4f}"
    ax.set_title(f"[{model_name}] Precision / Recall / F1 Heatmap", fontsize=12, fontweight="bold", pad=18)
    fig.text(
        0.5,
        0.01,
        f"Macro 평균  |  {macro_str}",
        ha="center",
        fontsize=9,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    filename = f"output_heatmap_{model_name.replace(' ', '_')}.png"
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 저장: {OUT / filename}")



def draw_label_score_summary(all_results: Dict[str, List[dict]]) -> None:
    label_stats = defaultdict(list)
    for img_results in all_results.values():
        for img_result in img_results:
            for row in img_result["eval_results"]:
                label_stats[row["gt_label"]].append(row)

    labels = sorted(label_stats.keys())
    if not labels:
        return

    iou_values = [float(np.mean([row["iou"] for row in label_stats[label]])) for label in labels]
    prec_values = [float(np.mean([row["precision"] for row in label_stats[label]])) for label in labels]
    rec_values = [float(np.mean([row["recall"] for row in label_stats[label]])) for label in labels]
    size_values = [float(np.mean([row["size_similarity"] for row in label_stats[label]])) for label in labels]

    x = np.arange(len(labels))
    width = 0.20
    fig, ax = plt.subplots(figsize=(max(20, len(labels) * 2.5), 10))

    bars = [
        ax.bar(x - 1.5 * width, iou_values, width, label="IoU", color="#3498db", alpha=0.85),
        ax.bar(x - 0.5 * width, prec_values, width, label="Precision", color="#9b59b6", alpha=0.85),
        ax.bar(x + 0.5 * width, rec_values, width, label="Recall", color="#f39c12", alpha=0.85),
        ax.bar(x + 1.5 * width, size_values, width, label="Size Similarity", color="#2ecc71", alpha=0.85),
    ]
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("값", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Label별 평균 IoU / Precision / Recall / Size Similarity  (전체 모델 통합)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    filename = "output_label_score.png"
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 저장: {OUT / filename}")


# ============================================================
# 실행 파이프라인
# ============================================================
def prepare_paths() -> List[Path]:
    global OUT

    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = "results_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    OUT = Path(output_dir)
    OUT.mkdir(exist_ok=True)

    root = Path(EVAL_DATASET_DIR)
    if not root.exists():
        raise FileNotFoundError(f"'{EVAL_DATASET_DIR}' 폴더가 없습니다.")

    data_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not data_dirs:
        raise ValueError(f"'{EVAL_DATASET_DIR}' 안에 하위 폴더가 없습니다.")

    print(f"📁 출력 폴더 : {OUT.resolve()}")
    print(f"📂 데이터 수 : {len(data_dirs)}개")
    for data_dir in data_dirs:
        images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        masks = list((data_dir / "gt_masks").glob("*.png")) if (data_dir / "gt_masks").exists() else []
        print(f"   📁 {data_dir.name:30} | 이미지:{len(images)}  GT마스크:{len(masks)}")

    return data_dirs



def evaluate_models(models: Dict[str, Callable[[str, int, int], List[dict]]], data_dirs: List[Path]) -> Dict[str, List[dict]]:
    all_results: Dict[str, List[dict]] = {}

    for model_name, model_fn in models.items():
        print(f"\n{'=' * 65}")
        print(f"  🤖 {model_name} 평가 중...")
        print(f"{'=' * 65}")
        img_results: List[dict] = []

        for data_dir in data_dirs:
            gt_mask_dir = data_dir / "gt_masks"
            img_path = get_image_file(data_dir)
            if img_path is None or not gt_mask_dir.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  ⚠️  이미지 로드 실패: {img_path}")
                continue

            height, width = img_bgr.shape[:2]
            detections = model_fn(str(img_path), width, height)
            gt_masks = load_gt_masks(gt_mask_dir, width, height)
            eval_results = match_and_score(gt_masks, detections)

            img_score = float(np.mean([row["score"] for row in eval_results])) if eval_results else 0.0
            correct = sum(1 for row in eval_results if row["gt_label"] == row["pred_label"])

            img_results.append(
                {
                    "name": data_dir.name,
                    "eval_results": eval_results,
                    "img_score": img_score,
                    "correct": correct,
                    "total": len(eval_results),
                }
            )

            print(f"  📁 {data_dir.name}")
            for row in eval_results:
                pred = row["pred_label"] or "미탐지"
                mark = "⭕" if row["gt_label"] == row["pred_label"] else "❌"
                print(
                    f"     {mark} {row['gt_label']:20} → {pred:20} | "
                    f"IoU:{row['iou']:.3f}  score:{row['score']:.3f}"
                )
            print(f"     정답 {correct}/{len(eval_results)}  img_score:{img_score:.4f}\n")

        all_results[model_name] = img_results

    print("✅ 전체 평가 완료")
    return all_results



def summarize_results(all_results: Dict[str, List[dict]]) -> None:
    summary = {}

    for model_name, img_results in all_results.items():
        all_rows = [row for img_result in img_results for row in img_result["eval_results"]]
        correct = sum(1 for row in all_rows if row["gt_label"] == row["pred_label"])
        total = len(all_rows)
        summary[model_name] = {
            "correct": correct,
            "total": total,
            "label_acc": correct / total if total else 0.0,
            "avg_score": float(np.mean([row["score"] for row in all_rows])) if all_rows else 0.0,
            "avg_iou": float(np.mean([row["iou"] for row in all_rows])) if all_rows else 0.0,
            "avg_size": float(np.mean([row["size_similarity"] for row in all_rows])) if all_rows else 0.0,
        }

    ranked = sorted(summary.items(), key=lambda item: (item[1]["label_acc"], item[1]["avg_score"]), reverse=True)

    width_col = 72
    print("\n" + "=" * width_col)
    print(f"  {'모델명':22} {'정답률':>11} {'정답/전체':>9} {'avg score':>12} {'avg size':>10} {'종합 지표':>7}")
    print("=" * width_col)
    for rank, (name, stats) in enumerate(ranked):
        crown = " 👑 " if rank == 0 else f"  {rank + 1} "
        print(
            f" {crown} {name:22} {stats['label_acc']:>12.1%} "
            f"   {stats['correct']:>5}/{stats['total']:<5}   "
            f"{stats['avg_score']:>8.4f} {stats['avg_size']:>11.4f}"
            f"{2*stats['label_acc']+3*stats['avg_score']+5*stats['avg_size']:>11.4f}"
        )
    print("=" * width_col)

    for model_name, img_results in all_results.items():
        all_rows = [row for img_result in img_results for row in img_result["eval_results"]]
        by_label = defaultdict(list)
        for row in all_rows:
            by_label[row["gt_label"]].append(row)

        print(f"\n  [{model_name}] label별 정답률")
        print(f"  {'label':25} {'정답률':>6} {'정답/전체':>10} {'avg score':>14} {'size':>6}")
        print("  " + "-" * 57)
        for label in sorted(by_label):
            rows = by_label[label]
            correct = sum(1 for row in rows if row["gt_label"] == row["pred_label"])
            acc = correct / len(rows)
            avg_score = float(np.mean([row["score"] for row in rows]))
            avg_size = float(np.mean([row["size_similarity"] for row in rows]))
            bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
            print(f"  {label:25} {acc:>8.1%}   {correct:>3}/{len(rows):<3}  {bar}  {avg_score:.4f}   {avg_size:.4f}")

    print("\n✅ 성능 요약 완료")



def run_visualizations(all_results: Dict[str, List[dict]]) -> None:
    cm_store = {}

    for model_name, img_results in all_results.items():
        print(f"\n{'=' * 55}")
        print(f"  🤖 {model_name}")
        print(f"{'=' * 55}")
        true_labels, cm_precision, cm_recall, macro_p, macro_r, macro_f1 = draw_confusion_matrix(model_name, img_results)
        cm_store[model_name] = {
            "true_labels": true_labels,
            "cm_precision": cm_precision,
            "cm_recall": cm_recall,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f1": macro_f1,
        }

    print("\n✅ Confusion Matrix 완료")

    for model_name, store in cm_store.items():
        print(f"\n{'=' * 55}")
        print(f"  🤖 {model_name}")
        print(f"{'=' * 55}")
        draw_pr_bar(model_name, store)

    print("\n✅ PR 막대그래프 완료")

    for model_name, store in cm_store.items():
        print(f"\n{'=' * 55}")
        print(f"  🤖 {model_name}")
        print(f"{'=' * 55}")
        draw_pr_heatmap(model_name, store)

    print("\n✅ Heatmap 완료")
    draw_label_score_summary(all_results)



def main() -> None:
    print(f"Python: {sys.version}")
    data_dirs = prepare_paths()
    models = build_models()
    all_results = evaluate_models(models, data_dirs)
    summarize_results(all_results)
    run_visualizations(all_results)


if __name__ == "__main__":
    main()
