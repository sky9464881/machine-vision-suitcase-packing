import os
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtGui import QPixmap, QFont, QColor, QPalette
from PySide6.QtCore import Qt


STYLE = """
QWidget#MainScreen {
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

QWidget#content {
    background-color: #F5F5F5;
}

QLabel#pageTitle {
    color: #1A1A1A;
    font-size: 26px;
    font-weight: bold;
    padding: 32px 32px 16px 32px;
}

QWidget#card {
    background-color: #FFFFFF;
    border-radius: 8px;
}

QLabel#cardTitle {
    color: #1A1A1A;
    font-size: 15px;
    font-weight: bold;
}

QLabel#cardDesc {
    color: #888888;
    font-size: 13px;
}

QPushButton#primaryBtn {
    background-color: #F5A623;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: bold;
    padding: 12px 28px;
}

QPushButton#primaryBtn:hover {
    background-color: #E09510;
}

QPushButton#primaryBtn:pressed {
    background-color: #C8840E;
}

QPushButton#secondaryBtn {
    background-color: #FFFFFF;
    color: #1A1A1A;
    border: 1px solid #DDDDDD;
    border-radius: 6px;
    font-size: 14px;
    padding: 12px 28px;
}

QPushButton#secondaryBtn:hover {
    background-color: #F5F5F5;
}
"""


class MainScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setObjectName("MainScreen")
        self.setStyleSheet(STYLE)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title_label = QLabel("캐리어 배치 최적화")
        title_label.setObjectName("appTitle")

        version_label = QLabel("v1.0.0")
        version_label.setObjectName("appVersion")

        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(version_label)

        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #3D3D3D;")
        sidebar_layout.addWidget(divider)

        menu_items = [
            ("🏠", "홈", True),
            ("📷", "사진 입력", False),
            ("✂️", "세그멘테이션", False),
            ("🗂", "레이어 뷰", False),
        ]

        for icon, name, active in menu_items:
            item = QLabel(f"  {icon}  {name}")
            item.setObjectName("menuItemActive" if active else "menuItem")
            sidebar_layout.addWidget(item)

        sidebar_layout.addStretch()

        bottom_label = QLabel("  ⚙️  설정")
        bottom_label.setObjectName("menuItem")
        sidebar_layout.addWidget(bottom_label)

        # ── Content area ─────────────────────────────
        content = QWidget()
        content.setObjectName("content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(56)
        topbar.setStyleSheet("background-color: #FFFFFF; border-bottom: 1px solid #E0E0E0;")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(24, 0, 24, 0)
        topbar_layout.addStretch()
        admin_label = QLabel("admin  ▾")
        admin_label.setStyleSheet("color: #1A1A1A; font-size: 14px;")
        topbar_layout.addWidget(admin_label)

        content_layout.addWidget(topbar)

        # Main body
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(32, 32, 32, 32)
        body_layout.setSpacing(24)

        page_title = QLabel("캐리어 배치 최적화")
        page_title.setObjectName("pageTitle")
        page_title.setContentsMargins(0, 0, 0, 0)

        # Card
        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet("QWidget#card { background-color: #FFFFFF; border-radius: 8px; }")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 32, 32, 32)
        card_layout.setSpacing(20)

        # Image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(340)
        self.image_label.setStyleSheet("background-color: #F0F0F0; border-radius: 6px;")

        base_dir = os.path.dirname(__file__)
        img_path = os.path.join(base_dir, "images", "start.jpg")
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(640, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.image_label.setText("시작 이미지를 찾을 수 없습니다.")
            self.image_label.setStyleSheet(
                "background-color: #F0F0F0; border-radius: 6px; color: #888; font-size: 14px;"
            )

        desc = QLabel("이미지를 불러와 객체를 자동으로 인식하고, 레이어별로 확인하세요.")
        desc.setObjectName("cardDesc")
        desc.setAlignment(Qt.AlignCenter)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        start_btn = QPushButton("시작하기")
        start_btn.setObjectName("primaryBtn")
        start_btn.setFixedWidth(160)
        start_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_row.addWidget(start_btn)
        btn_row.addStretch()

        card_layout.addWidget(self.image_label)
        card_layout.addWidget(desc)
        card_layout.addLayout(btn_row)

        body_layout.addWidget(page_title)
        body_layout.addWidget(card)
        body_layout.addStretch()

        content_layout.addWidget(body)

        root.addWidget(sidebar)
        root.addWidget(content)

    def mousePressEvent(self, event):
        self.stack.setCurrentIndex(1)

    def keyPressEvent(self, event):
        self.stack.setCurrentIndex(1)
