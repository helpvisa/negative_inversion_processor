import sys
import numpy as np
from PySide6.QtCore import Qt, Slot, QThreadPool
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog,
                               QGraphicsScene, QMessageBox, QSplitter)
from scipy import ndimage
import parse_cli_arguments
from ui_classes import Worker
from custom_widgets import ImageView, LabeledSlider
from inverter import load_raw_image, process_negative
from colour_management import convert_to_sRGB
from processing import process_all_adjustments


# derive from QWidget to create a custom updateable image class
class PrimaryImageView(QWidget):
    def __init__(self, parent=None):
        super(PrimaryImageView, self).__init__(parent)
        self.setMinimumSize(600, 400)

        # configure QGraphicsScene and ImageView
        self.layout = QHBoxLayout()
        self.scene = QGraphicsScene(0, 0, 1200, 800)
        self.view = ImageView(self.scene)
        # configure preview image itself
        self.preview_pixmap = self.scene.addPixmap(QPixmap(1200, 800))
        self.preview_pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.preview_pixmap.setPos(0, 0)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

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
        new_pixmap = QPixmap.fromImage(q_image)
        self.preview_pixmap.setPixmap(new_pixmap)
        # resize view to match image
        self.scene.setSceneRect(self.preview_pixmap.boundingRect())
        self.view.centerOn(self.preview_pixmap)


class ToolPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        #--- top-level tools layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        #--- inversion tools layout
        self.inversion_tools_header = QLabel("<b>Inversion</b>")
        self.inversion_layout = QVBoxLayout()
        self.red_ratio_slider = LabeledSlider("Red Ratio",
                                              0.0, 3.0, 1.36, 300)
        self.blue_ratio_slider = LabeledSlider("Blue Ratio",
                                               0.0, 3.0, 0.86, 300)
        self.green_exponent_slider = LabeledSlider("Green Exponent",
                                                   0.0, 5.0, 1.5, 500)
        self.inversion_layout.addWidget(self.red_ratio_slider)
        self.inversion_layout.addWidget(self.blue_ratio_slider)
        self.inversion_layout.addWidget(self.green_exponent_slider)
        #--- add all layouts
        self.layout.addWidget(self.inversion_tools_header)
        self.layout.addLayout(self.inversion_layout)
        self.setLayout(self.layout)


class EditingDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # internal tracking vars
        self.current_raw = None
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

        # image preview / canvas
        self.image_preview = PrimaryImageView(self)
        # controls
        # demo controls
        self.current_file_label = QLabel("NO FILE LOADED")
        self.load_button = QPushButton("Load Image")
        self.rotate_left_button = QPushButton("Rotate Left")
        self.rotate_right_button = QPushButton("Rotate Right")
        self.preview_button = QPushButton("Preview Image")
        # actual side panel
        self.tool_panel = ToolPanel()

        # define layouts
        # top-level layout
        self.layout = QVBoxLayout()
        # file layout
        self.load_layout = QHBoxLayout()
        self.load_layout.addWidget(self.current_file_label)
        self.load_layout.addWidget(self.load_button)
        # editing layout
        self.editing_layout = QSplitter(Qt.Horizontal)
        self.editing_layout.setHandleWidth(16)
        self.editing_layout.addWidget(self.image_preview)
        self.editing_layout.addWidget(self.tool_panel)
        # add all layouts and wrap it all up
        self.layout.addLayout(self.load_layout)
        self.layout.addWidget(self.editing_layout)
        self.setLayout(self.layout)

        # wiring up functions
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
            self.preview_button.setEnabled(False)
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
                self.current_file_label.setText(self.current_raw)
                self.load_button.setEnabled(True)
                self.preview_button.setEnabled(True)
            def error_func(e):
                exctype, value, error = e
                # tell user there's an issue
                QMessageBox.critical(self, "Error processing image!", error)
                # and re-enable the buttons
                self.current_file_label.setText("NO FILE LOADED")
                self.preview_button.setEnabled(True)
                self.load_button.setEnabled(True)
            # instantiate and run a thread
            # should split this out into its own function, surely
            thread = self.instantiate_thread(init_func)
            self.active_threads[thread.thread_id] = thread
            thread.signals.result.connect(post_func)
            thread.signals.error.connect(error_func)
            thread.signals.finished.connect(self.remove_thread)
            self.threadpool.start(thread)

    def preview_inverted_negative(self):
        if len(self.source_image) > 0:
            self.load_button.setEnabled(False)
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
                self.load_button.setEnabled(True)
            def error_func(e):
                exctype, value, error = e
                QMessageBox.critical(self, "Error processing image!", error)
                self.preview_button.setEnabled(True)
                self.load_button.setEnabled(True)
            thread = self.instantiate_thread(init_func)
            self.active_threads[thread.thread_id] = thread
            thread.signals.result.connect(post_func)
            thread.signals.error.connect(error_func)
            thread.signals.finished.connect(self.remove_thread)
            self.threadpool.start(thread)

    def instantiate_thread(self, func, *args, **kwargs):
        self.thread_id += 1
        thread = Worker(func, thread_id=self.thread_id)
        return thread


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Film Negative Inverter")
        self.editing_display = EditingDisplay()
        self.setCentralWidget(self.editing_display)
        self.resize(1200, 800)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
