import sys
import math
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog)
import colour
from scipy import ndimage
import parse_cli_arguments
from inverter import load_raw_image, process_negative
from processing import convert_to_sRGB, process_all_adjustments


# derive from QWidget to create a custom updateable image class
class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageDisplayWidget, self).__init__(parent)
        self.setMinimumSize(1200, 800)

        # replace with QGraphicsView?
        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    @Slot()
    def update_image(self):
        """
        Modify numpy pixel data array and update the GUI to display the new
        image
        """
        display_image, sRGB_profile = convert_to_sRGB(self.image_array)
        display_image = np.clip(display_image, a_min=0, a_max=1)
        q_image = QImage(np.multiply(display_image, 255).astype(np.uint8),
                         display_image.shape[1],
                         display_image.shape[0],
                         3 * display_image.shape[1],
                         QImage.Format.Format_RGB888)
        # convert to QPixmap for display
        q_pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(q_pixmap.scaled(self.image_label.size(),
                                                Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))


class CentralPane(QWidget):
    def __init__(self, parent=None):
        super(CentralPane, self).__init__(parent)
        self.args = parse_cli_arguments.parse_user_arguments()
        # finagle the args in an ugly way to avoid preview breakage
        # I should really convert a_width/height to a float that insets
        # the analysis region automatically based on actual working_image size
        self.args.analysis_width = int(math.floor(self.args.analysis_width * 0.25))
        self.args.analysis_height = int(math.floor(self.args.analysis_height * 0.25))
        self.args.resize = 1.0
        
        self.load_layout = QHBoxLayout()
        self.current_file_label = QLabel("NO FILE LOADED")
        self.load_button = QPushButton("Load Image")
        self.load_layout.addWidget(self.current_file_label)
        self.load_layout.addWidget(self.load_button)

        self.layout = QVBoxLayout()
        self.image_preview = ImageDisplayWidget(self)
        self.preview_button = QPushButton("Preview Image")
        self.layout.addLayout(self.load_layout)
        self.layout.addWidget(self.image_preview)
        self.layout.addWidget(self.preview_button)
        self.setLayout(self.layout)

        self.load_button.clicked.connect(self.load_raw_file)
        self.preview_button.clicked.connect(self.preview_inverted_negative)

    def load_raw_file(self):
        self.current_raw, discarded_text = QFileDialog.getOpenFileName()
        self.source_image = load_raw_image(self.current_raw).astype(np.float32) / 65535.0
        self.source_image = ndimage.zoom(self.source_image, (0.25, 0.25, 1), order=3)
        self.current_file_label.setText(self.current_raw)
        self.image_preview.image_array = self.source_image
        self.image_preview.update_image()

    def preview_inverted_negative(self):
        # this absolutely, unquestionably needs to be threaded
        self.adjustments = process_negative(self.source_image,
                                            self.args)
        self.image_preview.image_array = process_all_adjustments(self.source_image,
                                                                    self.adjustments)
        self.image_preview.update_image()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Film Negative Inverter")
        self.resize(800, 600)
        self.central_pane = CentralPane()
        self.setCentralWidget(self.central_pane)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
