import sys
import math
import numpy as np
from PySide6.QtCore import Qt, Slot, QThreadPool
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog)
import colour
from scipy import ndimage
import parse_cli_arguments
from ui_classes import Worker
from inverter import load_raw_image, process_negative
from colour_management import convert_to_sRGB
from processing import process_all_adjustments


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
        # internal tracking vars
        self.source_image = []
        self.preview_image = []

        # load an initial set of user args just to populate the values
        self.args = parse_cli_arguments.parse_user_arguments()
        
        # create a thread pool for triggering image processing
        self.threadpool = QThreadPool()
        thread_count = self.threadpool.maxThreadCount()
        print(f"Multithreading active with {thread_count} threads.",
              file=sys.stderr)
        # hold threads to prevent garbage collection
        # and initialize a thread_id iterator
        self.active_threads = {}
        self.thread_id = 0

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

    def remove_thread(self, thread_id):
        self.active_threads[thread_id] = None

    def load_raw_file(self):
        # load raw image and resize it for the preview pane
        image_path, discard_text = QFileDialog.getOpenFileName()
        if image_path:
            # disable button while we load the new image
            self.current_raw = image_path
            self.load_button.setEnabled(False)
            self.current_file_label.setText("Loading your image...")
            # the function to be executed within a separate thread
            def init_func():
                raw = load_raw_image(image_path).astype(np.float32) / 65535.0
                raw = ndimage.zoom(raw, (0.25, 0.25, 1), order=3)
                return raw
            # the function to be executed upon thread completion
            def post_func(image_data):
                self.source_image = image_data
                self.image_preview.image_array = self.source_image
                self.image_preview.update_image()
                self.load_button.setEnabled(True)
            # instantiate and run a thread
            # should split this out into its own function, surely
            thread = self.instantiate_thread(init_func)
            thread.signals.result.connect(post_func)
            thread.signals.finished.connect(self.remove_thread)
            self.active_threads[thread.thread_id] = thread
            self.threadpool.start(thread)

    def preview_inverted_negative(self):
        if len(self.source_image) > 0:
            self.preview_button.setEnabled(False)
            def init_func():
                adjustments = process_negative(self.source_image,
                                               self.args)
                return process_all_adjustments(self.source_image,
                                               adjustments)
            def post_func(image_data):
                self.preview_image = image_data
                self.image_preview.image_array = self.preview_image
                self.image_preview.update_image()
                self.preview_button.setEnabled(True)
            thread = self.instantiate_thread(init_func)
            thread.signals.result.connect(post_func)
            thread.signals.finished.connect(self.remove_thread)
            self.active_threads[thread.thread_id] = thread
            self.threadpool.start(thread)

    def instantiate_thread(self, func, *args, **kwargs):
        self.thread_id += 1
        thread = Worker(func, thread_id=self.thread_id)
        return thread


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
