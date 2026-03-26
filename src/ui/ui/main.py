import sys

from PySide6.QtWidgets import QApplication, QStackedWidget

from ui_layer import LayerScreen
from ui_main import MainScreen
from ui_photo import PhotoScreen
from ui_segment import SegmentScreen


class App(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.main_screen = MainScreen(self)
        self.photo_screen = PhotoScreen(self)
        self.segment_screen = SegmentScreen(self)
        self.layer_screen = LayerScreen(self)

        self.addWidget(self.main_screen)
        self.addWidget(self.photo_screen)
        self.addWidget(self.segment_screen)
        self.addWidget(self.layer_screen)

        self.setCurrentIndex(0)
        self.setWindowTitle("캐리어 배치 최적화 도구")
        self.resize(1100, 820)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
