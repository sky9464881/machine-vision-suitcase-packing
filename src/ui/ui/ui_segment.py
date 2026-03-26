from __future__ import annotations

from copy import deepcopy

import cv2
from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model import run_model_on_image
from preprocess import PreprocessError, preprocess_path


LOW_CONF_THRESHOLD = 0.50

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

QPushButton#deleteBtn {
    background-color:#FFFFFF; color:#CC3333;
    border:1px solid #FFCCCC; border-radius:6px; font-size:13px; padding:8px 18px;
}
QPushButton#deleteBtn:hover { background-color:#FFF5F5; }

QPushButton#nextBtn {
    background-color:#F5A623; color:#FFFFFF;
    border:none; border-radius:6px; font-size:14px; font-weight:bold; padding:10px 28px;
}
QPushButton#nextBtn:hover { background-color:#E09510; }
QPushButton#nextBtn:pressed { background-color:#C8840E; }

QListWidget {
    border:1px solid #E8E8E8; border-radius:6px;
    background-color:#FAFAFA; font-size:13px;
    outline: none;
}
QListWidget::item { padding:10px 14px; border-bottom:1px solid #F0F0F0; }
QListWidget::item:hover { background-color:#FFF8EE; border-left:3px solid #F5A623; }
QListWidget::item:selected { background-color:#FFF0D0; color:#1A1A1A; border-left:3px solid #F5A623; }

QLabel#imageArea {
    background-color:#F0F0F0;
    border-radius:6px;
    border:2px dashed #DDDDDD;
}
QLabel#infoText {
    color:#777777;
    font-size:12px;
    padding:2px 0;
}
"""


class ClickableImageLabel(QLabel):
    rightClicked = Signal(QPoint)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.rightClicked.emit(event.pos())
        super().mousePressEvent(event)


class SegmentScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.selected_path: str | None = None
        self.display_image = None
        self.original_outputs: list[dict] = []
        self.filtered_outputs: list[dict] = []
        self.last_display_pixmap_size: tuple[int, int] | None = None
        self.display_target_w = 700
        self.display_target_h = 420
        self.preprocess_steps: list[str] = []
        self.preprocess_warnings: list[str] = []

        self.setObjectName("root")
        self.setStyleSheet(STYLE)

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
            ("✂️", "세그멘테이션", True),
            ("🗂", "레이어 뷰", False),
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
        page_title = QLabel("세그멘테이션")
        page_title.setObjectName("pageTitle")
        self.back_btn = QPushButton("← 뒤로가기")
        self.back_btn.setObjectName("backBtn")
        self.back_btn.clicked.connect(self.go_back)
        header.addWidget(page_title)
        header.addStretch()
        header.addWidget(self.back_btn)

        card = QWidget()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        self.image_label = ClickableImageLabel()
        self.image_label.setObjectName("imageArea")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(380)
        self.image_label.setText("이미지를 선택하면 여기에 표시됩니다.")
        self.image_label.setStyleSheet(
            "QLabel#imageArea{background:#F5F5F5;border-radius:6px;"
            "border:2px dashed #DDDDDD;color:#AAAAAA;font-size:14px;}"
        )
        self.image_label.rightClicked.connect(self.delete_mask_by_click)

        self.preprocess_info_label = QLabel("전처리 정보가 여기에 표시됩니다.")
        self.preprocess_info_label.setObjectName("infoText")
        self.preprocess_info_label.setWordWrap(True)

        list_header = QLabel("감지된 객체 목록")
        list_header.setStyleSheet("color:#555;font-size:13px;font-weight:bold;padding:4px 0;")

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(180)
        self.list_widget.setMouseTracking(True)
        self.list_widget.viewport().setMouseTracking(True)
        self.list_widget.viewport().installEventFilter(self)

        bottom = QHBoxLayout()
        self.delete_btn = QPushButton("🗑  선택 항목 삭제")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.clicked.connect(self.delete_selected_item)
        self.next_btn = QPushButton("레이어 뷰 →")
        self.next_btn.setObjectName("nextBtn")
        self.next_btn.clicked.connect(self.next_screen)
        hint = QLabel("우클릭으로 이미지에서 직접 삭제 가능")
        hint.setStyleSheet("color:#AAAAAA;font-size:12px;")

        bottom.addWidget(self.delete_btn)
        bottom.addWidget(hint)
        bottom.addStretch()
        bottom.addWidget(self.next_btn)

        card_layout.addWidget(self.image_label)
        card_layout.addWidget(self.preprocess_info_label)
        card_layout.addWidget(list_header)
        card_layout.addWidget(self.list_widget)
        card_layout.addLayout(bottom)

        b_layout.addLayout(header)
        b_layout.addWidget(card)
        b_layout.addStretch()

        c_layout.addWidget(topbar)
        c_layout.addWidget(body)

        root.addWidget(sidebar)
        root.addWidget(content)

    def go_back(self):
        self.stack.setCurrentIndex(1)

    def reset_state(self):
        self.display_image = None
        self.original_outputs = []
        self.filtered_outputs = []
        self.preprocess_steps = []
        self.preprocess_warnings = []
        self.last_display_pixmap_size = None
        self.list_widget.clear()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("이미지를 선택하면 여기에 표시됩니다.")
        self.preprocess_info_label.setText("전처리 정보가 여기에 표시됩니다.")

    def set_image(self, path: str):
        self.selected_path = path
        self.list_widget.clear()

        try:
            preprocess_result = preprocess_path(path)
        except PreprocessError as exc:
            QMessageBox.warning(self, "전처리 실패", f"전처리에 실패했습니다.\n{exc}")
            self.reset_state()
            self.go_back()
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "전처리 실패", f"전처리 중 알 수 없는 오류가 발생했습니다.\n{exc}")
            self.reset_state()
            self.go_back()
            return

        try:
            outputs = run_model_on_image(preprocess_result.preprocessed_image)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "모델 실행 실패", f"모델 실행에 실패했습니다.\n{exc}")
            self.reset_state()
            self.go_back()
            return

        self.display_image = preprocess_result.preprocessed_image
        self.preprocess_steps = preprocess_result.steps
        self.preprocess_warnings = []
        self.original_outputs = deepcopy(outputs)
        self.filtered_outputs = deepcopy(outputs)
        self.refresh_info_text()
        self.refresh_list()
        self.show_all_masks()

    def refresh_info_text(self):
        steps = " → ".join(self.preprocess_steps) if self.preprocess_steps else "없음(원본 사용)"
        if self.display_image is not None:
            h, w = self.display_image.shape[:2]
            size_text = f"표시 이미지 크기: {w}x{h}"
        else:
            size_text = "표시 이미지 크기: -"

        if self.preprocess_warnings:
            warning_text = " | ".join(self.preprocess_warnings)
            text = f"전처리: {steps} | {size_text} | 경고: {warning_text}"
        else:
            text = f"전처리: {steps} | {size_text}"

        self.preprocess_info_label.setText(text)

    def refresh_list(self):
        self.list_widget.clear()
        for obj in self.filtered_outputs:
            label = obj["label"]
            confidence = obj["confidence"]
            is_low = confidence < LOW_CONF_THRESHOLD
            if is_low:
                text = f"⚠  {label}  ({confidence:.2f})  [낮은 신뢰도]"
            else:
                text = f"✔  {label}  ({confidence:.2f})"

            item = QListWidgetItem(text)
            if is_low:
                item.setForeground(QBrush(QColor(180, 50, 50)))
                item.setBackground(QBrush(QColor(255, 240, 240)))
            self.list_widget.addItem(item)

    def show_all_masks(self):
        if self.display_image is None:
            return
        img = self.display_image.copy()
        overlay = img.copy()
        for obj in self.filtered_outputs:
            mask = obj.get("mask")
            if mask is None:
                continue
            overlay[mask] = [0, 0, 255] if obj["confidence"] < LOW_CONF_THRESHOLD else [0, 255, 0]
        blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
        self.update_image(blended)

    def show_single_mask(self, index: int):
        if self.display_image is None:
            return
        if index < 0 or index >= len(self.filtered_outputs):
            self.show_all_masks()
            return

        img = self.display_image.copy()
        overlay = img.copy()
        obj = self.filtered_outputs[index]
        mask = obj.get("mask")
        if mask is not None:
            color = [0, 0, 255] if obj["confidence"] < LOW_CONF_THRESHOLD else [0, 200, 255]
            overlay[mask] = color
        blended = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
        self.update_image(blended)

    def update_image(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.display_target_w,
            self.display_target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.last_display_pixmap_size = (scaled.width(), scaled.height())
        self.image_label.setPixmap(scaled)

    def delete_selected_item(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.filtered_outputs):
            QMessageBox.information(self, "알림", "삭제할 항목을 먼저 선택하세요.")
            return
        deleted_label = self.filtered_outputs[row]["label"]
        del self.filtered_outputs[row]
        self.refresh_list()
        self.show_all_masks()
        QMessageBox.information(self, "삭제 완료", f"'{deleted_label}' 항목을 제거했습니다.")

    def delete_mask_by_click(self, click_pos: QPoint):
        if self.display_image is None or not self.filtered_outputs or self.last_display_pixmap_size is None:
            return

        pix_w, pix_h = self.last_display_pixmap_size
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        offset_x = (label_w - pix_w) / 2
        offset_y = (label_h - pix_h) / 2
        x_in_pix = click_pos.x() - offset_x
        y_in_pix = click_pos.y() - offset_y

        if x_in_pix < 0 or y_in_pix < 0 or x_in_pix >= pix_w or y_in_pix >= pix_h:
            return

        img_h, img_w = self.display_image.shape[:2]
        img_x = int(x_in_pix * img_w / pix_w)
        img_y = int(y_in_pix * img_h / pix_h)

        target_index = self.find_topmost_mask_index(img_x, img_y)
        if target_index is None:
            QMessageBox.information(self, "알림", "해당 위치에 삭제할 항목이 없습니다.")
            return

        deleted_label = self.filtered_outputs[target_index]["label"]
        del self.filtered_outputs[target_index]
        self.refresh_list()
        self.show_all_masks()
        QMessageBox.information(self, "삭제 완료", f"'{deleted_label}' 항목을 제거했습니다.")

    def find_topmost_mask_index(self, x: int, y: int) -> int | None:
        candidates: list[tuple[int, float]] = []
        for idx, obj in enumerate(self.filtered_outputs):
            mask = obj.get("mask")
            if mask is None:
                continue
            h, w = mask.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                continue
            if mask[y, x]:
                candidates.append((idx, obj["confidence"]))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    def eventFilter(self, obj, event):
        if obj == self.list_widget.viewport():
            if event.type() == QEvent.MouseMove:
                item = self.list_widget.itemAt(event.pos())
                if item is not None:
                    self.show_single_mask(self.list_widget.row(item))
                else:
                    self.show_all_masks()
            elif event.type() == QEvent.Leave:
                self.show_all_masks()
        return super().eventFilter(obj, event)

    def next_screen(self):
        if not self.filtered_outputs or self.display_image is None:
            QMessageBox.information(self, "알림", "남아있는 항목이 없습니다.")
            return
        self.stack.layer_screen.set_data(self.display_image, self.filtered_outputs)
        self.stack.setCurrentIndex(3)
