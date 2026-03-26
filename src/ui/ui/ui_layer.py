from __future__ import annotations

import shutil
import sys
import obb_detection
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

# ui1 폴더에서 실행해도 상위 폴더의 packer 모듈을 가져올 수 있도록 경로 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maxrects_packer_layers import run_pack_layers  # noqa: E402


STYLE = """
QWidget {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
QWidget#root { background-color: #F5F5F5; }

QWidget#sidebar {
    background-color: #2C2C2C;
    min-width: 220px; max-width: 220px;
}
QLabel#appTitle { color:#FFFFFF; font-size:18px; font-weight:bold; padding:20px 16px 8px 16px; }
QLabel#appVersion { color:#F5A623; font-size:11px; padding:0px 16px 20px 16px; }
QLabel#menuItem { color:#CCCCCC; font-size:14px; padding:12px 20px; }
QLabel#menuItemActive {
    color:#FFFFFF; font-size:14px; font-weight:bold; padding:12px 20px;
    background-color:#3A3A3A; border-left:3px solid #F5A623;
}
QWidget#topbar { background-color:#FFFFFF; border-bottom:1px solid #E0E0E0; }
QWidget#content { background-color:#F5F5F5; }
QLabel#pageTitle { color:#1A1A1A; font-size:24px; font-weight:bold; }
QWidget#card { background-color:#FFFFFF; border-radius:8px; }

QPushButton#backBtn {
    background-color:#FFFFFF; color:#1A1A1A;
    border:1px solid #DDDDDD; border-radius:6px; font-size:13px; padding:8px 18px;
}
QPushButton#backBtn:hover { background-color:#F5F5F5; }

QPushButton#navBtn {
    background-color:#F5A623; color:#FFFFFF;
    border:none; border-radius:6px; font-size:13px; font-weight:bold; padding:8px 18px;
}
QPushButton#navBtn:hover { background-color:#E09510; }
QPushButton#navBtn:disabled { background-color:#D9D9D9; color:#888888; }

QLabel#imageArea {
    background-color:#F0F0F0;
    border-radius:6px;
    border:2px solid #E8E8E8;
}
"""


class LayerScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setObjectName("root")
        self.setStyleSheet(STYLE)

        self.layer_images: list[np.ndarray] = []
        self.layer_paths: list[str] = []
        self.current_layer_index = 0
        self.cache_root = CURRENT_DIR / "_layer_cache"
        self.bin_info_text = ""
        self.excluded_items_info: list[str] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        title = QLabel("짐 싸기")
        title.setObjectName("appTitle")
        version = QLabel("v1.1.0")
        version.setObjectName("appVersion")
        sb.addWidget(title)
        sb.addWidget(version)

        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet("background:#3D3D3D;")
        sb.addWidget(div)

        for icon, name, active in [
            ("🏠", "홈", False),
            ("📷", "사진 입력", False),
            ("✂️", "세그멘테이션", False),
            ("🗂", "레이어 뷰", True),
        ]:
            lbl = QLabel(f"  {icon}  {name}")
            lbl.setObjectName("menuItemActive" if active else "menuItem")
            sb.addWidget(lbl)

        sb.addStretch()
        settings_lbl = QLabel("  ⚙️  설정")
        settings_lbl.setObjectName("menuItem")
        sb.addWidget(settings_lbl)

        content = QWidget()
        content.setObjectName("content")
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(0)

        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(56)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)
        tb.addStretch()
        admin = QLabel("admin  ▾")
        admin.setStyleSheet("color:#1A1A1A;font-size:14px;")
        tb.addWidget(admin)

        body = QWidget()
        b_layout = QVBoxLayout(body)
        b_layout.setContentsMargins(32, 32, 32, 32)
        b_layout.setSpacing(20)

        header = QHBoxLayout()
        page_title = QLabel("레이어 뷰")
        page_title.setObjectName("pageTitle")
        self.back_button = QPushButton("← 뒤로가기")
        self.back_button.setObjectName("backBtn")
        self.back_button.clicked.connect(self.go_back)
        header.addWidget(page_title)
        header.addStretch()
        header.addWidget(self.back_button)

        card = QWidget()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        self.layer_title = QLabel("레이어 결과")
        self.layer_title.setStyleSheet("font-size:16px;font-weight:bold;color:#1A1A1A;")

        self.image_label = QLabel()
        self.image_label.setObjectName("imageArea")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(420)
        self.image_label.setText("레이어 이미지가 여기에 표시됩니다.")
        self.image_label.setStyleSheet(
            "QLabel#imageArea{background:#F5F5F5;border-radius:6px;"
            "border:2px dashed #DDDDDD;color:#AAAAAA;font-size:14px;}"
        )

        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("← 이전 레이어")
        self.prev_button.setObjectName("navBtn")
        self.prev_button.clicked.connect(self.show_prev_layer)
        self.next_button = QPushButton("다음 레이어 →")
        self.next_button.setObjectName("navBtn")
        self.next_button.clicked.connect(self.show_next_layer)
        self.page_info = QLabel("레이어 0 / 0")
        self.page_info.setStyleSheet("font-size:13px;color:#555555;")
        nav_row.addWidget(self.prev_button)
        nav_row.addStretch()
        nav_row.addWidget(self.page_info)
        nav_row.addStretch()
        nav_row.addWidget(self.next_button)

        legend = QLabel(
            "세그멘테이션에서 남겨둔 객체들을 MaxRects 방식으로 층별 배치한 결과입니다."
        )
        legend.setStyleSheet("font-size:12px;color:#666666;padding:2px 0;")
        legend.setWordWrap(True)

        self.excluded_label = QLabel("제외된 객체 없음")
        self.excluded_label.setStyleSheet(
            "font-size:12px;color:#B04A00;background:#FFF3E8;border:1px solid #FFD5B5;"
            "border-radius:6px;padding:10px;"
        )
        self.excluded_label.setWordWrap(True)
        self.excluded_label.hide()

        card_layout.addWidget(self.layer_title)
        card_layout.addWidget(self.image_label)
        card_layout.addLayout(nav_row)
        card_layout.addWidget(legend)
        card_layout.addWidget(self.excluded_label)

        b_layout.addLayout(header)
        b_layout.addWidget(card)
        b_layout.addStretch()

        c_layout.addWidget(topbar)
        c_layout.addWidget(body)

        root.addWidget(sidebar)
        root.addWidget(content)

        self._refresh_nav_state()

    def go_back(self):
        self.stack.setCurrentIndex(2)

    def _clear_cache_dir(self):
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root, ignore_errors=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _build_object_cutouts(self, base_image: np.ndarray, outputs: list[dict]) -> Path:
        self._clear_cache_dir()
        input_dir = self.cache_root / "objects"
        input_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for idx, obj in enumerate(outputs, start=1):
            mask = obj.get("mask")
            if mask is None:
                continue

            mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()

            # 1) mask에서 OBB 계산
            obb = obb_detection.get_obb_from_mask(mask_u8)
            if obb is None:
                continue

            (cx, cy), (w, h), angle, box = obb

            # 2) OBB 기준으로 회전 정렬된 객체 추출
            upright_img, upright_mask = obb_detection.extract_upright_object(
                image=base_image,
                mask=mask_u8,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                angle=angle,
            )

            if upright_img is None or upright_img.size == 0:
                continue

            label = str(obj.get("label", f"obj{idx}")).replace("/", "_").replace(" ", "_")
            conf = float(obj.get("confidence", 0.0))
            save_name = f"obj_{idx:03d}_{label}_{conf:.2f}.png"
            cv2.imwrite(str(input_dir / save_name), upright_img)
            saved_count += 1

        if saved_count == 0:
            raise RuntimeError("레이어 배치에 사용할 객체 이미지를 만들지 못했습니다.")

        return input_dir

    def _load_rendered_layers(self, layer_paths: list[str]) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        for path in layer_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        return images

    def _update_excluded_label(self):
        if not self.excluded_items_info:
            self.excluded_label.hide()
            self.excluded_label.setText("제외된 객체 없음")
            return

        joined = ", ".join(self.excluded_items_info)
        self.excluded_label.setText(
            f"상자에 들어가지 않아 제외된 객체 {len(self.excluded_items_info)}개: {joined}"
        )
        self.excluded_label.show()

    def _refresh_nav_state(self):
        total = len(self.layer_images)
        current = self.current_layer_index + 1 if total else 0
        self.page_info.setText(f"레이어 {current} / {total}")
        self.prev_button.setEnabled(total > 1 and self.current_layer_index > 0)
        self.next_button.setEnabled(total > 1 and self.current_layer_index < total - 1)
        if total:
            suffix = f" ({self.bin_info_text})" if self.bin_info_text else ""
            self.layer_title.setText(f"레이어 {current} 최적화 배치 결과{suffix}")
        else:
            self.layer_title.setText("레이어 결과")

    def _show_current_layer(self):
        if not self.layer_images:
            self.image_label.setText("표시할 레이어 이미지가 없습니다.")
            self.image_label.setPixmap(QPixmap())
            self._update_excluded_label()
            self._refresh_nav_state()
            return

        img = self.layer_images[self.current_layer_index]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pixmap.scaled(720, 460, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._refresh_nav_state()

    def show_prev_layer(self):
        if self.current_layer_index > 0:
            self.current_layer_index -= 1
            self._show_current_layer()

    def show_next_layer(self):
        if self.current_layer_index < len(self.layer_images) - 1:
            self.current_layer_index += 1
            self._show_current_layer()

    def set_data(self, base_image, outputs):
        if base_image is None:
            self.layer_images = []
            self.layer_paths = []
            self.current_layer_index = 0
            self.bin_info_text = ""
            self.excluded_items_info = []
            self.image_label.setText("표시할 이미지가 없습니다.")
            self.image_label.setPixmap(QPixmap())
            self._refresh_nav_state()
            return

        try:
            input_dir = self._build_object_cutouts(base_image, outputs)
            output_dir = self.cache_root / "packed_layers"
            output_dir.mkdir(parents=True, exist_ok=True)

            bin_w, bin_h, _, layer_paths, excluded_items = run_pack_layers(
                input_folder=str(input_dir),
                output_dir=str(output_dir),
                box_width_cm=45,
                box_height_cm=35,
                marker_cm=6.1,
                marker_px=108.65,
                allow_rotate=True,
                padding=8,
            )

            layer_images = self._load_rendered_layers(layer_paths)
            if not layer_images:
                raise RuntimeError("층별 결과 이미지를 불러오지 못했습니다.")

            self.layer_paths = layer_paths
            self.layer_images = layer_images
            self.current_layer_index = 0
            self.bin_info_text = f"bin: {bin_w} x {bin_h}px / 45 x 35cm / marker 6.1:{108.65}"
            self.excluded_items_info = [
                f"{item.label}({item.width}x{item.height}px)" for item in excluded_items
            ]
            self._update_excluded_label()
            self._show_current_layer()

            if excluded_items:
                QMessageBox.information(
                    self,
                    "일부 객체 제외됨",
                    "상자 크기보다 큰 객체는 제외하고 배치했습니다.\n\n"
                    + "제외된 객체:\n"
                    + "\n".join(
                        f"- {item.label} ({item.width}x{item.height}px)" for item in excluded_items
                    ),
                )

        except Exception as exc:  # noqa: BLE001
            self.layer_images = []
            self.layer_paths = []
            self.current_layer_index = 0
            self.bin_info_text = ""
            self.excluded_items_info = []
            self.image_label.setText("레이어 이미지를 생성하지 못했습니다.")
            self.image_label.setPixmap(QPixmap())
            self._refresh_nav_state()
            QMessageBox.warning(self, "레이어 생성 실패", f"층별 최적화 배치 이미지를 만드는 중 오류가 발생했습니다.\n{exc}")
