from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QPushButton, QFileDialog,
    QLabel, QSizePolicy, QHBoxLayout
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


STYLE = """
QWidget {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}

QWidget#root {
    background-color: #F5F5F5;
}

QWidget#sidebar {
    background-color: #2C2C2C;
    min-width: 220px;
    max-width: 220px;
}

QLabel#appTitle {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: bold;
    padding: 20px 16px 8px 16px;
}

QLabel#appVersion {
    color: #F5A623;
    font-size: 11px;
    padding: 0px 16px 20px 16px;
}

QLabel#menuItem {
    color: #CCCCCC;
    font-size: 14px;
    padding: 12px 20px;
}

QLabel#menuItemActive {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: bold;
    padding: 12px 20px;
    background-color: #3A3A3A;
    border-left: 3px solid #F5A623;
}

QWidget#topbar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E0E0E0;
}

QWidget#content {
    background-color: #F5F5F5;
}

QLabel#pageTitle {
    color: #1A1A1A;
    font-size: 24px;
    font-weight: bold;
}

QWidget#card {
    background-color: #FFFFFF;
    border-radius: 8px;
}

QPushButton#backBtn {
    background-color: #FFFFFF;
    color: #1A1A1A;
    border: 1px solid #DDDDDD;
    border-radius: 6px;
    font-size: 13px;
    padding: 8px 18px;
}
QPushButton#backBtn:hover { background-color: #F5F5F5; }

QPushButton#addBtn {
    background-color: #F5A623;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: bold;
    padding: 10px 24px;
}
QPushButton#addBtn:hover { background-color: #E09510; }
QPushButton#addBtn:pressed { background-color: #C8840E; }

QLabel#imgThumb {
    border: 2px solid #E0E0E0;
    border-radius: 6px;
    background-color: #F8F8F8;
}
QLabel#imgThumb:hover {
    border: 2px solid #F5A623;
}
"""


class ClickableImageLabel(QLabel):
    def __init__(self, image_path, callback):
        super().__init__()
        self.image_path = image_path
        self.callback = callback
        self.setObjectName("imgThumb")
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        self.callback(self.image_path)


class PhotoScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.image_paths = []
        self.setObjectName("root")
        self.setStyleSheet(STYLE)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(0)

        title = QLabel("짐 싸기")
        title.setObjectName("appTitle")
        version = QLabel("v1.0.0")
        version.setObjectName("appVersion")
        sb_layout.addWidget(title)
        sb_layout.addWidget(version)

        div = QWidget(); div.setFixedHeight(1); div.setStyleSheet("background:#3D3D3D;")
        sb_layout.addWidget(div)

        for icon, name, active in [
            ("🏠", "홈", False),
            ("📷", "사진 입력", True),
            ("✂️", "세그멘테이션", False),
            ("🗂", "레이어 뷰", False),
        ]:
            lbl = QLabel(f"  {icon}  {name}")
            lbl.setObjectName("menuItemActive" if active else "menuItem")
            sb_layout.addWidget(lbl)

        sb_layout.addStretch()
        settings = QLabel("  ⚙️  설정")
        settings.setObjectName("menuItem")
        sb_layout.addWidget(settings)

        # ── Content ───────────────────────────────────
        content = QWidget()
        content.setObjectName("content")
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(0)

        # Topbar
        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(56)
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(24, 0, 24, 0)
        tb_layout.addStretch()
        admin = QLabel("admin  ▾")
        admin.setStyleSheet("color:#1A1A1A; font-size:14px;")
        tb_layout.addWidget(admin)

        # Body
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(32, 32, 32, 32)
        body_layout.setSpacing(20)

        # Header row
        header_row = QHBoxLayout()
        page_title = QLabel("사진 입력")
        page_title.setObjectName("pageTitle")
        self.back_button = QPushButton("← 뒤로가기")
        self.back_button.setObjectName("backBtn")
        self.back_button.clicked.connect(self.go_back)
        header_row.addWidget(page_title)
        header_row.addStretch()
        header_row.addWidget(self.back_button)

        # Card
        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet("QWidget#card{background:#FFFFFF;border-radius:8px;}")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        # Add button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.add_button = QPushButton("+ 사진 추가")
        self.add_button.setObjectName("addBtn")
        self.add_button.clicked.connect(self.add_photo)
        btn_row.addWidget(self.add_button)

        self.grid = QGridLayout()
        self.grid.setSpacing(16)

        card_layout.addLayout(btn_row)
        card_layout.addLayout(self.grid)

        body_layout.addLayout(header_row)
        body_layout.addWidget(card)
        body_layout.addStretch()

        c_layout.addWidget(topbar)
        c_layout.addWidget(body)

        root.addWidget(sidebar)
        root.addWidget(content)

    def go_back(self):
        self.stack.setCurrentIndex(0)

    def add_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        self.image_paths.insert(0, file_path)
        self.image_paths = self.image_paths[:4]
        self.refresh_grid()

    def refresh_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for idx, path in enumerate(self.image_paths):
            row = idx // 2
            col = idx % 2
            label = ClickableImageLabel(path, self.select_image)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                label.setPixmap(
                    pixmap.scaled(380, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                label.setText("이미지 로드 실패")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setMinimumHeight(240)
            self.grid.addWidget(label, row, col)

    def select_image(self, path):
        self.stack.segment_screen.set_image(path)
        self.stack.setCurrentIndex(2)
